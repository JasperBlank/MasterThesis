"""
Layout grid over endoscope distance D for the Ø 8 mm end-effector.

For each minimum D the layout optimizer is run and the resulting cross-section
is drawn in one panel, annotated with the achieved minimum wall thickness.

Internal holes: Ø 4 mm (working channel), 2 × Ø 1.2 mm (endoscopes),
Ø 1 mm (syringe/needle).

Reuses `optimize_layout`, `reposition_syringe`, `min_clearance` from
`end_effector_layout.py`. Output: figures/layout_grid_D_sweep.png
"""

import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Pull optimizer from the GUI script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_DIR / "figures"

spec = importlib.util.spec_from_file_location("m", str(SCRIPT_DIR / "end_effector_layout.py"))
plt.show = lambda *a, **k: None  # neutralize show() during import
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ----- Parameters -----
R_OUTER = 4.0                     # mm (Ø 8 mm shell)
RADII = [2.0, 0.6, 0.6, 0.5]      # working, endo A, endo B, syringe
PANEL_LABELS = ["Ø4", "Ø1.2", "Ø1.2", "Ø1"]
COLORS = ["#3b82f6", "#10b981", "#10b981", "#f59e0b"]
D_VALUES = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]  # minimum endoscope distance per panel

fig, axes = plt.subplots(2, 3, figsize=(13, 8.8))

for ax, D in zip(axes.flat, D_VALUES):
    pm, _ = mod.optimize_layout(
        RADII[:3], R_OUTER,
        n_restarts=30, seed=int(D * 1000),
        min_baseline=D, endo_idx=(1, 2), fix_center_idx=0,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if pm is None:
        ax.set_title("D = %.2f mm  |  infeasible" % D)
        continue

    pos = np.vstack([pm, np.array([[0.0, 0.0]])])
    pos, _ = mod.reposition_syringe(pos, RADII, R_OUTER, syringe_idx=3)
    wall = mod.min_clearance(pos.flatten(), RADII, R_OUTER)
    actual_D = float(np.linalg.norm(pos[1] - pos[2]))

    ax.add_patch(Circle((0, 0), R_OUTER, fill=False, lw=2, edgecolor="black"))
    for p, r, c, lbl in zip(pos, RADII, COLORS, PANEL_LABELS):
        ax.add_patch(Circle((p[0], p[1]), r, facecolor=c, alpha=0.7, edgecolor="black"))
        ax.text(p[0], p[1], lbl, ha="center", va="center", fontsize=8)

    lim = R_OUTER + 0.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title("D = %.2f mm  |  wall = %.3f mm" % (actual_D, wall))

fig.suptitle(
    "Ø 8 mm shell — layout vs minimum endoscope distance D\n"
    "Ø 4 working  |  2 × Ø 1.2 endoscopes  |  Ø 1 syringe",
    fontsize=14,
)
plt.tight_layout()
FIGURES_DIR.mkdir(exist_ok=True)
output_path = FIGURES_DIR / "layout_grid_D_sweep.png"
plt.savefig(str(output_path), dpi=140)
print("saved %s" % output_path)
