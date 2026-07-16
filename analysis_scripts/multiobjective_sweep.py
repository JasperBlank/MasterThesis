"""
Multi-objective sweep over endoscope distance D for the Ø 8 mm end-effector.

Objectives (all higher = better):
  1. Depth resolution score   S_depth = 1 / ΔZ(D, Z₀)
        with ΔZ = Z²·Δd / (f·D)   (stereo triangulation, sub-pixel matching limit)
  2. Binocular FOV overlap    S_fov = max(0, 1 - D / (2·Z₀·tan α))
        i.e. the fraction of each camera's view that overlaps the other at Z₀.
  3. Syringe reconstructability  S_rec = 1 - |y_syringe - y_endo_midline| / R_outer
        i.e. how close the syringe tip is to the line through both endoscopes,
        proxying how centred it is in the stereo overlap volume.

Hard constraint: min wall thickness w(D) ≥ w_min.

We run the existing layout optimizer for every D, then evaluate the three scores.
Output:
  - Per-D table
  - 3-panel figure of the three scores vs D
  - Pareto front in 2-D projections (depth vs FOV, depth vs reconstructability)
  - Combined weighted score with a recommended D
"""

import numpy as np
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pull optimizer from the GUI script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = PROJECT_DIR / "figures"

spec = importlib.util.spec_from_file_location("m", str(SCRIPT_DIR / "end_effector_layout.py"))
plt.show = lambda *a, **k: None  # neutralize show() during import
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# ----- Parameters -----
R_OUTER = 4.0           # mm
RADII = [2.0, 0.6, 0.6, 0.5]
F_PX = 800.0            # focal length in pixels
DD_SUBPIX = 0.25        # sub-pixel matching error
Z_NOMINAL = 40.0        # nominal working distance, mm
FOV_HALF_ANGLE = np.deg2rad(60.0)  # half of 120° diagonal FOV (muC112 spec)
W_MIN = 0.30            # required minimum wall thickness (mm)

Ds = np.arange(2.5, 6.01, 0.1)

rows = []
for D in Ds:
    pm, _ = mod.optimize_layout(
        RADII[:3], R_OUTER,
        n_restarts=25, seed=int(D * 1000),
        min_baseline=D, endo_idx=(1, 2), fix_center_idx=0,
    )
    if pm is None:
        continue
    p = np.vstack([pm, [[0.0, 0.0]]])
    p, _ = mod.reposition_syringe(p, RADII, R_OUTER, syringe_idx=3, n_restarts=15, seed=7)
    w = mod.min_clearance(p.flatten(), RADII, R_OUTER)
    actual_D = float(np.linalg.norm(p[1] - p[2]))

    dZ = Z_NOMINAL ** 2 * DD_SUBPIX / (F_PX * actual_D)
    S_depth = 1.0 / dZ                                # bigger = better
    overlap = max(0.0, 1.0 - actual_D / (2.0 * Z_NOMINAL * np.tan(FOV_HALF_ANGLE)))
    S_fov = overlap
    y_endo_mid = 0.5 * (p[1, 1] + p[2, 1])
    S_rec = max(0.0, 1.0 - abs(p[3, 1] - y_endo_mid) / R_OUTER)

    rows.append((actual_D, w, dZ, S_depth, S_fov, S_rec))

rows = np.array(rows)
feasible = rows[rows[:, 1] >= W_MIN]

# ----- Print table -----
print(f"\nf={F_PX} px, dd={DD_SUBPIX} px, Z0={Z_NOMINAL} mm, FOV half-angle={np.rad2deg(FOV_HALF_ANGLE):.0f} deg, wall >= {W_MIN} mm\n")
print("   D   |  wall  |   dZ    | S_depth | S_fov | S_rec | feasible")
print("-" * 70)
for r in rows:
    flag = "yes" if r[1] >= W_MIN else "no "
    print(f" {r[0]:5.2f} | {r[1]:5.3f} |  {r[2]:5.3f}  |  {r[3]:5.2f} | {r[4]:.3f} | {r[5]:.3f} |   {flag}")

# ----- Combined weighted score -----
def normalize(col):
    lo, hi = col.min(), col.max()
    return (col - lo) / (hi - lo) if hi > lo else np.zeros_like(col)

w_depth, w_fov, w_rec = 0.5, 0.3, 0.2
norm_depth = normalize(rows[:, 3])
norm_fov = normalize(rows[:, 4])
norm_rec = normalize(rows[:, 5])
combined = w_depth * norm_depth + w_fov * norm_fov + w_rec * norm_rec
combined_feas = np.where(rows[:, 1] >= W_MIN, combined, -np.inf)
best_idx = int(np.argmax(combined_feas))
print(f"\nWeighted best feasible (w_depth={w_depth}, w_fov={w_fov}, w_rec={w_rec}):")
print(f"  D = {rows[best_idx,0]:.2f} mm  |  wall = {rows[best_idx,1]:.3f} mm  |  dZ = {rows[best_idx,2]:.3f} mm  |  combined = {combined[best_idx]:.3f}")

# ----- Plots -----
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

ax = axes[0, 0]
ax.plot(rows[:, 0], rows[:, 2], color="C0")
ax.axvline(rows[best_idx, 0], color="black", ls="--", lw=0.8, label=f"best D={rows[best_idx,0]:.2f}")
ax.set_xlabel("D [mm]"); ax.set_ylabel("ΔZ at Z=40 mm [mm]")
ax.set_title("Depth resolution (lower = better)")
ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(rows[:, 0], rows[:, 4], color="C1")
ax.set_xlabel("D [mm]"); ax.set_ylabel("Binocular FOV overlap fraction")
ax.set_title("FOV overlap at Z=40 mm")
ax.grid(alpha=0.3)
ax.axvline(rows[best_idx, 0], color="black", ls="--", lw=0.8)

ax = axes[1, 0]
ax.plot(rows[:, 0], rows[:, 5], color="C2")
ax.set_xlabel("D [mm]"); ax.set_ylabel("Syringe reconstructability score")
ax.set_title("Syringe centred-ness in stereo (higher = better)")
ax.grid(alpha=0.3)
ax.axvline(rows[best_idx, 0], color="black", ls="--", lw=0.8)

ax = axes[1, 1]
infeasible = rows[rows[:, 1] < W_MIN]
ax.scatter(infeasible[:, 2], infeasible[:, 4], c="gray", alpha=0.4, s=18, label="infeasible (wall < 0.3)")
sc = ax.scatter(feasible[:, 2], feasible[:, 4], c=feasible[:, 0], cmap="viridis", s=30, label="feasible")
plt.colorbar(sc, ax=ax, label="D [mm]")
ax.set_xlabel("ΔZ [mm] (lower → better)")
ax.set_ylabel("FOV overlap fraction (higher → better)")
ax.set_title("Pareto view: depth-res vs FOV overlap")
ax.invert_xaxis()
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

fig.suptitle(
    f"Multi-objective sweep — Ø 8 mm shell, Ø 1.2 endos, Ø 4 working, Ø 1 syringe\n"
    f"wall_min = {W_MIN} mm, f={F_PX:.0f} px, Δd={DD_SUBPIX} px, Z={Z_NOMINAL:.0f} mm, FOV=120°",
    fontsize=11)
plt.tight_layout()
FIGURES_DIR.mkdir(exist_ok=True)
output_path = FIGURES_DIR / "multiobjective_sweep.png"
plt.savefig(str(output_path), dpi=140)
print(f"\nsaved {output_path}")
