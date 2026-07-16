# Dual-Endoscope End Effector: Impact of Inter-Endoscope Width (Baseline) on Depth Resolution

**Date:** 2026-05-18
**Author:** Prepared for J. Blank (Master's project)

---

## 1. Introduction

A dual-endoscope end effector mounts two endoscopic cameras side-by-side on a single distal tool (typically at the tip of a robotic manipulator or laparoscopic instrument). The two views form a **stereo pair**, allowing 3D reconstruction of the surgical scene through triangulation. This report focuses on how the **inter-endoscope spacing — the stereo baseline `B`** — drives **depth resolution**, and on the design trade-offs this imposes for an end-effector-mounted system.

---

## 2. Stereo Geometry Fundamentals

For a rectified stereo pair with parallel optical axes:

```
Z = (f · B) / d
```

| Symbol | Meaning | Typical units |
|--------|---------|---------------|
| `Z` | Depth to the scene point | mm |
| `f` | Focal length (in pixels) | px |
| `B` | Baseline — distance between the two optical centers | mm |
| `d` | Disparity (pixel shift between left/right images) | px |

### 2.1 Depth resolution

Differentiating the depth equation with respect to disparity gives the **depth resolution** (the smallest depth difference the system can resolve):

```
ΔZ = (Z² / (f · B)) · Δd
```

where `Δd` is the disparity resolution (sub-pixel matching error, typically 0.1–1 px).

**Key consequence:** `ΔZ` scales with **1/B** and with **Z²**.

- Doubling the baseline → halves the depth uncertainty at a given range.
- Doubling the working distance → quadruples the depth uncertainty.

---

## 3. Impact of Inter-Endoscope Width on Depth Resolution

### 3.1 Wider baseline → better depth resolution

A larger separation between the two endoscopes increases the disparity for a given depth, so a fixed sub-pixel matching error translates into a smaller absolute depth error.

**Worked example** (typical endoscopic camera, `f` ≈ 800 px, `Δd` ≈ 0.25 px, working distance `Z` = 50 mm):

| Baseline `B` | ΔZ at Z = 50 mm | ΔZ at Z = 100 mm |
|--------------|-----------------|------------------|
| 1.6 mm | 0.49 mm | 1.95 mm |
| 4 mm   | 0.20 mm | 0.78 mm |
| 8 mm   | 0.10 mm | 0.39 mm |
| 15 mm  | 0.052 mm | 0.21 mm |
| 25 mm  | 0.031 mm | 0.125 mm |

So an end effector that can physically separate the two endoscopes by, say, 15 mm rather than 1.6 mm provides roughly an **order-of-magnitude** improvement in depth resolution.

### 3.2 Trade-offs of a wider baseline

Increasing `B` is not free. The key drawbacks:

1. **Larger overall tool diameter / footprint.** Endoscopes for minimally invasive surgery (MIS) must pass through trocars (typically 5–12 mm). A wide baseline directly conflicts with miniaturization; this is why a 4-mm endoscope barrel cannot host a traditional two-camera stereo head at all.
2. **Reduced field-of-view overlap.** Depth can only be computed where both endoscopes see the same surface. As `B` grows relative to `Z`, the binocular overlap region shrinks, especially at close range.
3. **Larger disparity search range** → higher computational cost and a greater risk of mismatches in textureless tissue regions.
4. **More occlusion / parallax artifacts.** Wider separation means surfaces visible to one endoscope but hidden to the other (specular highlights, fluid pools, and tissue folds aggravate this).
5. **Calibration sensitivity.** Mechanical flex between two independently mounted endoscopes changes `B` and the relative rotation, causing depth drift. The wider the baseline arm, the more leverage small deflections have.
6. **Near-range limit.** Very close objects (within a few `B`) produce disparities so large they fall outside the overlap, or beyond the matcher's search window.

### 3.3 Narrower baseline → worse depth resolution, but other advantages

A small baseline (e.g., 1.6 mm in miniature binocular endoscopes) preserves the slim form factor needed for MIS, keeps the views nearly identical (easier stereo matching, fewer occlusions), and works well at very short range — at the cost of coarse depth resolution.

### 3.4 Focal length interacts with baseline

Empirical work on stereo endoscopes has shown that **focal length has a larger effect on depth accuracy than baseline alone** — the product `f·B` is what matters in the depth equation. A long-focal-length (narrow-FOV) endoscope can partially compensate for a small baseline, but at the cost of a narrower field of view.

---

## 4. Design Guidelines for a Dual-Endoscope End Effector

| Design parameter | Recommendation |
|------------------|----------------|
| Baseline `B` | Set to ≈ 1/10 to 1/5 of the **nominal working distance** `Z`. For MIS at `Z` ≈ 30–80 mm this suggests `B` ≈ 5–15 mm. |
| Mechanical stability | Rigid, thermally stable mount; both scopes referenced to a common machined frame to minimize calibration drift. |
| Synchronization | Hardware-triggered capture so left/right frames are time-aligned (essential for moving tissue). |
| Calibration | Bundle-adjusted intrinsic + extrinsic calibration with a sub-mm target; re-check after sterilization cycles. |
| Convergence | Slight toe-in (verged) optical axes can extend useful overlap at close range, at the cost of more complex rectification. |
| Sub-pixel matching | Use sub-pixel-accurate stereo matching (e.g., SGBM with parabolic refinement, or learning-based methods); `Δd ≈ 0.1–0.25 px` is achievable on textured tissue. |
| Lighting | Coaxial illumination per channel to reduce specularities that defeat stereo correspondence. |

### 4.1 If a wide baseline is infeasible

Alternatives reported in the literature for ultra-thin endoscopic 3D imaging:
- **Single-lens, dual-aperture** systems (e.g., 3-D-MARVEL 4-mm endoscope) — replace the geometric baseline with a pupil split.
- **Structured light** projected through one channel, imaged through the other.
- **Monocular learning-based depth** (NeRF/diffusion priors, e.g., EndoPerfect, EndoMUST) — useful when geometric baseline is essentially zero.

---

## 5. Practical Example — Sizing the Baseline for a Surgical End Effector

Assume:
- Working distance `Z` = 40 mm
- Required depth resolution `ΔZ` ≤ 0.1 mm (for sub-mm tissue registration / suturing)
- Camera focal length `f` = 1000 px (narrow-FOV chip-on-tip)
- Sub-pixel disparity error `Δd` = 0.2 px

Solve for `B`:

```
B ≥ (Z² · Δd) / (f · ΔZ)
  = (40² · 0.2) / (1000 · 0.1)
  = 3.2 mm
```

A baseline of ≥ **3.2 mm** is sufficient — comfortably achievable in a 10–12 mm trocar-compatible end effector. To retain margin against calibration drift and matching noise, a designed baseline of 6–8 mm is a sensible target.

---

## 5b. Effect of an 8 mm Probe Outer Diameter

If the end effector itself is constrained to **Ø 8 mm** (e.g., to pass through a standard 8–10 mm trocar or cannula), the baseline is **geometrically capped** by what fits inside the housing.

### 5b.1 Maximum achievable baseline

Two endoscopes of diameter `d` packed side-by-side inside a Ø 8 mm tube (wall thickness `w`) give a maximum center-to-center spacing:

```
B_max = 8 − 2w − d
```

Typical chip-on-tip endoscope diameters and the resulting baseline:

| Endoscope Ø `d` | Wall `w` | `B_max` | Notes |
|-----------------|----------|---------|-------|
| 1.0 mm (fiber/chip-on-tip) | 0.3 mm | ≈ 5.4 mm | Very small sensors, lower image resolution |
| 1.7 mm | 0.3 mm | ≈ 4.7 mm | Common micro-endoscope size |
| 2.7 mm (standard arthroscope chip) | 0.3 mm | ≈ 3.7 mm | Good image quality, modest baseline |
| 3.5 mm | 0.3 mm | ≈ 2.9 mm | Larger sensor, but baseline is squeezed |

So in practical terms an 8 mm probe limits the **realistic baseline to roughly 3–5 mm**, depending on the endoscope chosen.

### 5b.2 Depth-resolution consequences

Re-running the depth-resolution numbers from §3.1 (`f` = 800 px, `Δd` = 0.25 px):

| Baseline `B` | ΔZ at Z = 30 mm | ΔZ at Z = 50 mm | ΔZ at Z = 80 mm |
|--------------|-----------------|-----------------|-----------------|
| 3 mm (large scopes) | 0.094 mm | 0.26 mm | 0.67 mm |
| 4 mm | 0.070 mm | 0.20 mm | 0.50 mm |
| 5 mm (small scopes) | 0.056 mm | 0.16 mm | 0.40 mm |

At a typical surgical working distance of 30–50 mm, an 8 mm probe can still deliver **sub-millimeter depth resolution**, but you should not expect the 0.05 mm precision available to larger stereo rigs.

### 5b.3 Practical implications of the 8 mm constraint

1. **Endoscope size becomes the dominant lever.** Each extra mm of endoscope diameter directly costs ~1 mm of baseline. Choosing the smallest endoscopes that still meet image-quality requirements is the single most effective design decision.
2. **Wall and packaging budget matters.** Sterilizable housing, illumination fibers, irrigation/suction channels, and the working channel for the end-effector tool all eat into the 8 mm envelope. A realistic internal volume for the two scopes may be closer to Ø 5–6 mm.
3. **Working distance should be kept short.** Because `ΔZ ∝ Z²`, halving the working distance gives the same depth-precision improvement as quadrupling the baseline — easier to achieve than enlarging the probe.
4. **Compensate optically.** With baseline capped, the remaining levers are **focal length** (narrower FOV, higher `f` in pixels), **sensor resolution** (more pixels per disparity), and **sub-pixel matching quality**.
5. **Consider verged (toe-in) optics.** With only 3–5 mm of baseline, slightly converging the two optical axes onto the nominal working point keeps the binocular overlap large at close range.
6. **Calibration is critical.** With a small baseline, a 50 µm mechanical shift between the two scopes represents ~1–2% of `B` and produces a comparable depth error. The mount must be rigid and thermally stable, ideally machined as one piece.
7. **Plan for illumination and occlusion.** With both scopes close together, shadows and specularities are very similar in both views — generally good for matching, but it also means a single specular highlight can blind both channels simultaneously. Polarized or alternating illumination can help.

### 5b.4 Twist / rotational misalignment: B = 4 mm vs B = 6 mm

"Twist" here means a small **relative angular misalignment** between the two endoscope optical axes — either residual from assembly or induced by mechanical flex of the housing in use. Three modes matter:

1. **Yaw twist `Δφ`** (rotation about the vertical axis perpendicular to the baseline) — biases the matched disparity, so it shows up directly as a depth error.
2. **Roll twist `Δθ`** (rotation about each scope's own optical axis) — produces vertical disparity, which breaks epipolar matching unless rectified.
3. **Baseline-length twist** — a lateral shift `δB` of one scope relative to the other directly changes `B`, scaling depth linearly.

Assume `f ≈ 800 px`, working distance `Z = 40 mm`, image half-height `r ≈ 400 px`.

**Yaw twist `Δφ` → depth error**
Induced disparity error `Δd ≈ f · Δφ`. Plugging into `ΔZ = Z²·Δd/(f·B) = Z²·Δφ/B`:

| `Δφ` | ΔZ at B = 4 mm | ΔZ at B = 6 mm | Improvement |
|------|----------------|----------------|-------------|
| 0.05° (0.87 mrad) | 0.35 mm | 0.23 mm | −33 % |
| 0.10° (1.75 mrad) | 0.70 mm | 0.47 mm | −33 % |
| 0.20° (3.49 mrad) | 1.40 mm | 0.93 mm | −33 % |
| 0.50° (8.73 mrad) | 3.49 mm | 2.33 mm | −33 % |

Depth error from yaw scales as **1/B**, so a 6 mm baseline is **1.5× more tolerant** than 4 mm to the same yaw twist. Equivalently, to keep `ΔZ ≤ 0.1 mm` at Z = 40 mm, yaw must be held to ≤ 0.014° at B = 4 mm but can relax to ≤ 0.021° at B = 6 mm.

**Roll twist `Δθ` → vertical disparity (matching failure)**
Vertical disparity at image edge: `v_err ≈ r · Δθ`. This is **baseline-independent**:

| `Δθ` | v_err at r = 400 px |
|------|---------------------|
| 0.05° | 0.35 px |
| 0.10° | 0.70 px |
| 0.20° | 1.40 px |
| 0.50° | 3.49 px |

Beyond ~0.5 px vertical disparity, dense stereo matchers start to fail. Roll is the more dangerous twist mode because the 6 mm baseline gives you **no** mechanical advantage — both 4 mm and 6 mm need the same sub-0.1° roll alignment, or software rectification to compensate. The wider 6 mm mount is actually slightly **harder** to keep aligned because the longer separation gives a longer lever arm for the roll error to accumulate from a non-rigid frame.

**Baseline-length twist `δB` → depth scaling bias**
If mechanical play shifts one scope laterally by `δB`, depth is biased by the fractional change `δB / B`:

| `δB` | fractional error at B = 4 mm | at B = 6 mm |
|------|------------------------------|-------------|
| 20 µm | 0.50 % | 0.33 % |
| 50 µm | 1.25 % | 0.83 % |
| 100 µm | 2.50 % | 1.67 % |

At Z = 40 mm, a 50 µm shift biases depth by ≈ 0.5 mm at B = 4 mm versus ≈ 0.33 mm at B = 6 mm. Again, the wider baseline is more forgiving by a factor of `B_new / B_old = 1.5`.

**Net comparison**

| Twist mode | 4 mm vs 6 mm | Practical takeaway |
|------------|--------------|--------------------|
| Yaw (depth bias) | 6 mm 1.5× more tolerant | Favors 6 mm |
| Roll (matching failure) | Equal in optics; 6 mm slightly worse mechanically | Push to rigid one-piece mount regardless |
| Baseline shift | 6 mm 1.5× more tolerant | Favors 6 mm |

Going from B = 4 mm to B = 6 mm gives you a roughly **33 % reduction in depth error** for the same yaw/lateral twist — useful but modest. It does **not** help with roll twist, which has to be handled by mechanical rigidity or post-capture rectification. For an 8 mm probe with ≤ 2.7 mm endoscopes (B ≈ 3.7 mm achievable), the practical question is whether a redesign to reach B ≈ 6 mm — likely requiring smaller (≤ 1.7 mm) endoscopes — is worth the optical penalty for the extra twist margin.

### 5b.5 Sizing check against the §5 example

The §5 example required `B ≥ 3.2 mm` at `Z` = 40 mm for `ΔZ` ≤ 0.1 mm. An 8 mm probe with ≤ 2.7 mm endoscopes meets this comfortably (`B ≈ 3.7 mm`). If the application demands `ΔZ` ≤ 0.05 mm at the same range, the 8 mm form factor is **not sufficient** with a passive stereo pair — you would need to either shrink the working distance, increase focal length, or move to structured-light/active depth augmentation.

---

## 6. Summary

- Depth resolution of a dual-endoscope stereo system scales as **ΔZ ∝ Z² / (f·B)**.
- Wider inter-endoscope spacing **linearly** improves depth resolution but **trades against** tool diameter, FOV overlap, occlusion, calibration stability, and minimum working distance.
- The "correct" baseline is the smallest `B` that meets the depth-resolution requirement at the maximum working distance — typically 5–15 mm for laparoscopic working ranges.
- Focal length and sub-pixel matching quality are co-equal levers; the design problem is to optimize the product `f·B` against the form-factor and FOV constraints of the end effector.

---

## Sources

- [A Miniature Binocular Endoscope with Local Feature Matching and Stereo Matching for 3D Measurement and 3D Reconstruction (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6069142/)
- [4-mm-diameter three-dimensional imaging endoscope with steerable camera for minimally invasive surgery (3-D-MARVEL)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5054215/)
- [Design and Characterisation of Stereo Endoscope for Polyp Size Measurement (IEEE)](https://ieeexplore.ieee.org/document/10596887/)
- [Stereo endoscopy as a 3-D measurement tool (PubMed)](https://pubmed.ncbi.nlm.nih.gov/19963650/)
- [Method for measuring stereo camera depth accuracy based on stereoscopic vision](https://www.researchgate.net/publication/235349563_Method_for_measuring_stereo_camera_depth_accuracy_based_on_stereoscopic_vision)
- [Multi-scale, multi-dimensional binocular endoscopic image depth estimation network (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0010482523007709)
- [Dense Depth Estimation from Stereo Endoscopy Videos Using Unsupervised Optical Flow Methods (Springer)](https://link.springer.com/chapter/10.1007/978-3-030-80432-9_26)
- [EndoPerfect: A Hybrid NeRF-Stereo Vision Approach (arXiv)](https://arxiv.org/html/2410.04041v3)
- [EndoMUST: Monocular Depth Estimation for Robotic Endoscopy (arXiv)](https://arxiv.org/html/2506.16017v1)
- [Designing a New Endoscope for Panoramic-View with Focus-Area 3D-Vision in MIS (Springer)](https://link.springer.com/article/10.1007/s40846-019-00503-9)
- [Lens, Focal Length and Stereo Baseline Calculator (Nerian)](https://nerian.com/support/calculator/)
- [Configuring Stereo Depth (Luxonis docs)](https://docs.luxonis.com/hardware/platform/depth/configuring-stereo-depth)
- [How to choose the optimal baseline and resolution for a stereo vision camera (LinkedIn)](https://www.linkedin.com/advice/1/how-do-you-choose-optimal-baseline-resolution-stereo)
