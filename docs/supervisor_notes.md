# Supervisor Notes — Project Specs & Hardware

## Updated Specifications

- **Vasculature range:** 100–500 µm (revised to stay realistic with current hardware).
- **Probe diameter:** 8–12 mm (standard endoscopy range).
- Forwarded to Sarthak.

## Hardware

### Endoscopic Camera
- **Model in use:** [Comedia muC112 Series](https://www.comedia.com.hk/products/muc112) micro camera.
  - Sensor: OCHFA10 CMOS, 720 × 720 @ 30 fps, raw RGB analog output.
  - Tip: 1.05 × 1.05 mm, outer diameter 2.0 / 2.2 mm; cable Ø 0.5 mm, 1–3 m.
  - FoV: 120° diagonal; working distance 5–50 mm.
  - Power: 3.3 VDC, 25 mW.
  - Variants: with two LEDs or without LED (illumination via embedded LEDs or external source).
- **Previously considered (available in lab):** V1000LH (MD-B1000) — dimmer not functional; superseded by the muC112 above.

### Two-Photon Endomicroscopy Probe (out of scope, but consider its dimensions)
- Cylinder: **Ø 4 mm × 35 mm length**

### Working Channel
- Diameter must accommodate a needle that can deliver vasculatures of **100–500 µm**.
- Reference: needle gauge concept.
- **Syringe pump (available):** [NE-300 Just Infusion Syringe Pump (SyringePump.com)](https://www.syringepump.com/NE-300.php) — for pumping the micro-agent solution.

### Linear Stages — Thorlabs Instrumentation

| Item | Part # | Qty | Unit (€) | Total (€) |
|---|---|---|---|---|
| 25 mm Travel, DC Servo Motor Actuator, Ø3/8" Mounting Barrel | [Z925B](https://www.thorlabs.com/thorproduct.cfm?partnumber=Z925B) | 3 | 873.03 | 2,619.09 |
| K-Cube Brushed DC Servo Motor Controller (PSU not incl.) | [KDC101](https://www.thorlabs.com/thorproduct.cfm?partnumber=KDC101) | 3 | 751.52 | 2,254.56 |
| Ethernet & USB Controller Hub + PSU for Six K-Cubes | [KEH6](https://www.thorlabs.com/thorproduct.cfm?partnumber=KEH6) | 1 | 727.20 | 727.20 |
| 25 mm Travel Linear Translation Stage, M6 × 1.0 Taps (no actuator/mount) | [XR25X/M](https://www.thorlabs.com/thorproduct.cfm?partnumber=XR25X/M) | 3 | 389.05 | 1,167.15 |
| XZ Stage Assembly Kit for 3" Dovetails, M6 × 1.0 Taps | [XR25-XZ/M](https://www.thorlabs.com/thorproduct.cfm?partnumber=XR25-XZ/M) | 1 | 161.34 | 161.34 |
| Actuator Side-Mounting Kit for XR25 / XR50 Series Stages | [XR25-C1](https://www.thorlabs.com/thorproduct.cfm?partnumber=XR25-C1) | 3 | 43.37 | 130.11 |
| Base Plate for Stages with 3" Dovetails, Metric Slot Spacing | [XR25-B1/M](https://www.thorlabs.com/thorproduct.cfm?partnumber=XR25-B1/M) | 2 | 37.31 | 74.62 |
| Perpendicular Mounting Adapter (Male 2" / Female 3" Dovetail) | [XRN-A1](https://www.thorlabs.com/thorproduct.cfm?partnumber=XRN-A1) | 1 | 56.65 | 56.65 |

**Action:** check datasheets and driver interfaces for stage control.

## Action Items

- [ ] Identify project objectives
- [ ] Draft Gantt chart / timeline
- [ ] Prepare short presentation for next meeting
- [ ] Optional online meeting: Thursday or Friday, 14:00–17:00

## Notes

- Supervisor positive on attached plots — curious to see results.
