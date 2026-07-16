# Project notes

**Read `AGENTS.md` first — it is the single source of truth** for this project:
hardware quirks (endoscope LED protocol, USB flakiness, motor homing), key scripts,
the probe-pose pipeline, standard commands, conventions, and open work. It is shared
with other AI assistants (Codex reads it natively); keep it updated instead of this
file when project facts change.

Claude-specific notes only below.

## Hard rules (duplicated here because they gate every edit)
- **Python 3.8 only** — no `match`, no `X | Y` annotations, no `dict | dict`,
  no `list[int]` runtime generics. Use `typing.Union` / `typing.List`.
- **D = endoscope center-to-center distance** — never "baseline" in new code/labels.
- **Both tip LEDs at "min"** on the bench; only via MSMF streams (see AGENTS.md).

## Claude workflow notes
- PowerShell 5.1 mangles nested quotes in `python -c` one-liners — write a temp
  .py file in the scratchpad instead.
- Long-running previews/GUIs: launch as background tasks; stop them before any
  script needs the cameras (DSHOW capture is exclusive; MSMF is shared).
- After changing `digital_twin/twin_wasd_jog.py`, verify with
  `--test-render out.png` (headless) before handing back.
- Auto-memory holds the same hardware facts as AGENTS.md — when they conflict,
  AGENTS.md wins (it is newer and user-visible).
