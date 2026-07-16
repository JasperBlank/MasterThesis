"""
End-effector cross-section optimizer.

Outer shell: Ø 8 mm.
Internal holes: Ø 4 mm (working channel), Ø 1.2 mm (endoscope), Ø 1.2 mm (endoscope), Ø 1 mm (syringe).
Goal: place the four inner circles so that the minimum wall thickness
(between any two circles, and between each circle and the outer shell)
is maximized.

Interactive: drag the diameter sliders; the layout re-optimizes live.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
from scipy.optimize import minimize


LABELS = ["Ø4 (working)", "Ø1.2 (endo A)", "Ø1.2 (endo B)", "Ø1 (syringe)"]
COLORS = ["#3b82f6", "#10b981", "#10b981", "#f59e0b"]


def min_clearance(positions, radii, R_outer):
    """Smallest gap between any pair of circles and between each circle and the outer wall.
    Negative means overlap / sticking out."""
    n = len(radii)
    p = positions.reshape(n, 2)
    gaps = []
    for i in range(n):
        gaps.append(R_outer - np.linalg.norm(p[i]) - radii[i])
        for j in range(i + 1, n):
            gaps.append(np.linalg.norm(p[i] - p[j]) - radii[i] - radii[j])
    return min(gaps)


def optimize_layout(radii, R_outer, n_restarts=25, seed=0, min_baseline=0.0, endo_idx=(1, 2), fix_center_idx=0):
    """Maximize the minimum clearance using SLSQP with a slack variable.
    Variables: [x1,y1, x2,y2, ..., xn,yn, t]; maximize t s.t. t <= every gap."""
    n = len(radii)
    rng = np.random.default_rng(seed)

    def make_constraints():
        cons = []
        for i in range(n):
            def wall_con(z, i=i):
                p = z[:2 * n].reshape(n, 2)
                t = z[-1]
                return R_outer - np.linalg.norm(p[i]) - radii[i] - t
            cons.append({"type": "ineq", "fun": wall_con})
        for i in range(n):
            for j in range(i + 1, n):
                def pair_con(z, i=i, j=j):
                    p = z[:2 * n].reshape(n, 2)
                    t = z[-1]
                    return np.linalg.norm(p[i] - p[j]) - radii[i] - radii[j] - t
                cons.append({"type": "ineq", "fun": pair_con})
        if fix_center_idx is not None:
            k = fix_center_idx
            # Keep on the vertical centerline (x = 0); free to slide along y.
            cons.append({"type": "eq", "fun": lambda z, k=k: z[2 * k]})
        if min_baseline > 0.0:
            ia, ib = endo_idx
            def baseline_con(z, ia=ia, ib=ib):
                p = z[:2 * n].reshape(n, 2)
                return np.linalg.norm(p[ia] - p[ib]) - min_baseline
            cons.append({"type": "ineq", "fun": baseline_con})
        return cons

    cons = make_constraints()
    best = None
    best_t = -np.inf

    for _ in range(n_restarts):
        x0 = []
        for idx, r in enumerate(radii):
            if idx == fix_center_idx:
                y0 = rng.uniform(-(R_outer - r), R_outer - r)
                x0.extend([0.0, y0])
                continue
            rho = rng.uniform(0, max(R_outer - r, 0.0))
            theta = rng.uniform(0, 2 * np.pi)
            x0.extend([rho * np.cos(theta), rho * np.sin(theta)])
        x0.append(0.0)
        x0 = np.array(x0)

        res = minimize(
            lambda z: -z[-1],
            x0,
            method="SLSQP",
            constraints=cons,
            options={"maxiter": 300, "ftol": 1e-9},
        )
        if res.success or res.status in (0, 9):
            t = res.x[-1]
            pos = res.x[:2 * n].reshape(n, 2)
            actual = min_clearance(pos, radii, R_outer)
            if actual > best_t:
                best_t = actual
                best = pos

    return best, best_t


def reposition_syringe(positions, radii, R_outer, syringe_idx=3, n_restarts=40, seed=0):
    """Phase 2: holding all other circles fixed, move the syringe to maximize
    its own minimum clearance (to the outer wall and to every other circle).
    Returns updated positions and the syringe's min clearance."""
    rng = np.random.default_rng(seed)
    rs = radii[syringe_idx]
    others = [(positions[i], radii[i]) for i in range(len(radii)) if i != syringe_idx]

    # Lock syringe to the y-axis (x = 0); only y is free. Use a dense 1-D scan
    # then refine — robust because the objective is a piecewise-min, not smooth.
    def min_gap_at_y(y):
        xy = np.array([0.0, float(y)])
        gaps = [R_outer - np.linalg.norm(xy) - rs]
        for (c, r) in others:
            gaps.append(np.linalg.norm(xy - c) - r - rs)
        return min(gaps)

    y_lim = max(R_outer - rs, 0.0)
    ys = np.linspace(-y_lim, y_lim, 4001)
    vals = np.array([min_gap_at_y(y) for y in ys])
    k = int(np.argmax(vals))
    best_y = float(ys[k])
    best_val = float(vals[k])

    # Local refine around the best grid point
    lo = ys[max(0, k - 2)]
    hi = ys[min(len(ys) - 1, k + 2)]
    res = minimize(lambda yv: -min_gap_at_y(yv[0]), [best_y],
                   method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 200})
    if -res.fun > best_val:
        best_val = -res.fun
        best_y = float(res.x[0])
    best_xy = np.array([0.0, best_y])

    new_positions = positions.copy()
    new_positions[syringe_idx] = best_xy
    return new_positions, best_val


# ---------- UI ----------
fig, ax = plt.subplots(figsize=(8, 8.8))
plt.subplots_adjust(bottom=0.36)
ax.set_aspect("equal")
ax.set_title("End-effector layout — max wall thickness")

R_outer_init = 4.0  # Ø 8 mm
diameters_init = [4.0, 1.2, 1.2, 1.0]

outer_patch = Circle((0, 0), R_outer_init, fill=False, lw=2, edgecolor="black")
ax.add_patch(outer_patch)

inner_patches = []
inner_labels = []
for d, c, lbl in zip(diameters_init, COLORS, LABELS):
    p = Circle((0, 0), d / 2, fill=True, facecolor=c, alpha=0.55, edgecolor="black")
    ax.add_patch(p)
    inner_patches.append(p)
    inner_labels.append(ax.text(0, 0, lbl, ha="center", va="center", fontsize=7))

info_text = ax.text(
    0, -R_outer_init - 0.5, "", ha="center", va="top", fontsize=10, family="monospace"
)

ax.set_xlim(-R_outer_init - 1, R_outer_init + 1)
ax.set_ylim(-R_outer_init - 1.5, R_outer_init + 1)
ax.grid(True, alpha=0.3)

# Sliders
slider_axes = [plt.axes([0.18, 0.26 - i * 0.04, 0.65, 0.025]) for i in range(6)]
s_outer = Slider(slider_axes[0], "Ø outer", 4.0, 14.0, valinit=8.0, valstep=0.1)
s_d1 = Slider(slider_axes[1], "Ø working", 1.0, 6.0, valinit=4.0, valstep=0.1)
s_d2 = Slider(slider_axes[2], "Ø endo A", 0.5, 4.0, valinit=1.2, valstep=0.1)
s_d3 = Slider(slider_axes[3], "Ø endo B", 0.5, 4.0, valinit=1.2, valstep=0.1)
s_d4 = Slider(slider_axes[4], "Ø syringe", 0.3, 3.0, valinit=1.0, valstep=0.1)
s_bmin = Slider(slider_axes[5], "min endoscope distance D", 0.0, 12.0, valinit=5.0, valstep=0.1)

ax_btn = plt.axes([0.45, 0.005, 0.1, 0.04])
btn = Button(ax_btn, "Re-solve")


def redraw(_=None):
    R_outer = s_outer.val / 2.0
    diameters = [s_d1.val, s_d2.val, s_d3.val, s_d4.val]
    radii = [d / 2.0 for d in diameters]

    if sum(np.pi * r * r for r in radii) > np.pi * R_outer ** 2 * 0.95:
        info_text.set_text("Too tight — total area exceeds outer envelope")
        info_text.set_color("red")
        fig.canvas.draw_idle()
        return

    min_baseline = s_bmin.val
    # Phase 1: optimize ONLY working + 2 endoscopes (syringe excluded entirely)
    radii_main = [radii[0], radii[1], radii[2]]
    pos_main, _ = optimize_layout(radii_main, R_outer, n_restarts=30,
                                   seed=int(1e6 * (R_outer + sum(diameters) + min_baseline)) % (2**31),
                                   min_baseline=min_baseline,
                                   endo_idx=(1, 2),
                                   fix_center_idx=0)
    if pos_main is None:
        info_text.set_text("No feasible packing found")
        info_text.set_color("red")
        fig.canvas.draw_idle()
        return
    # Stitch syringe placeholder, then Phase 2 places it on the y-axis
    pos = np.vstack([pos_main, np.array([[0.0, 0.0]])])
    pos, syringe_gap = reposition_syringe(pos, radii, R_outer, syringe_idx=3, n_restarts=40,
                                           seed=int(1e6 * (R_outer + sum(diameters) + min_baseline + 7)) % (2**31))
    t = min_clearance(pos.flatten(), radii, R_outer)

    outer_patch.set_radius(R_outer)
    ax.set_xlim(-R_outer - 1, R_outer + 1)
    ax.set_ylim(-R_outer - 1.5, R_outer + 1)

    new_labels = [f"Ø{diameters[0]:.1f} working",
                  f"Ø{diameters[1]:.1f} endo A",
                  f"Ø{diameters[2]:.1f} endo B",
                  f"Ø{diameters[3]:.1f} syringe"]

    for patch, lbl_obj, r, p, lbl in zip(inner_patches, inner_labels, radii, pos, new_labels):
        patch.set_center((p[0], p[1]))
        patch.set_radius(r)
        lbl_obj.set_position((p[0], p[1]))
        lbl_obj.set_text(lbl)
    baseline = float(np.linalg.norm(pos[1] - pos[2]))
    info_text.set_position((0, -R_outer - 0.5))
    info_text.set_text(
        f"min wall = {t:.3f} mm   D (endo A↔B) = {baseline:.3f} mm   syringe clearance = {syringe_gap:.3f} mm"
    )
    info_text.set_color("black" if t > 0.2 else "darkorange")

    fig.canvas.draw_idle()


for s in (s_outer, s_d1, s_d2, s_d3, s_d4, s_bmin):
    s.on_changed(redraw)
btn.on_clicked(redraw)

redraw()
plt.show()
