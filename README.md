# smooth-motion-mfg

Multi frame generation (3x and up) for NVIDIA Smooth Motion, the driver-level
frame generation that works in games with no DLSS or temporal upscaling.

**Status: research. Nothing here changes a game or the driver yet.**
First target: Ghost Recon Breakpoint (DX11), RTX 40-series, 3x.

## Where things are

| Path | What |
|---|---|
| `docs/DESIGN.md` | What the existing mods prove, the plan, and the open questions |
| `tools/get_nvpresent.cmd` | Double-click: copies your installed `NvPresent64.dll` into `ref/` |
| `docs/findings/` | What inspection found, per driver build |
| `tools/inspect_nvpresent.py` | Offline inspector: exports, imports, CUDA kernels, pacing hooks |
| `tools/analyze_capture.py` | Tests the flow reading of the network output on F9 captures |
| `tools/cubin_info.py` | Parameter layout and parameter reads of one dumped kernel |
| `src/tracer/`, `build.sh` | Phase 1 tracer: log-only `version.dll` |
| `tests/test_inspect.py` | Self-test for the inspector on synthetic data |
| `ref/`, `out/` | NVIDIA files and decoded kernels. Local only, git-ignored |

## Inspecting your own driver

1. Double-click `tools\get_nvpresent.cmd` (copies `NvPresent64.dll` into `ref\`).
2. Run:

   ```
   python tools\inspect_nvpresent.py ref\NvPresent64.dll --dump out\kernels > out\report.txt
   ```

## Progress

- [x] Phase 0: static inspection of driver 616.56 ([findings](docs/findings/616.56.md))
- [ ] Phase 1: runtime tracer (log-only). Built, see [docs/TRACER.md](docs/TRACER.md); first trace done ([findings](docs/findings/616.56-trace.md)); network output decoded ([findings](docs/findings/616.56-network-output.md)): flows can be rescaled to other times
- [ ] Phase 2: frames at time t. Prototype 0.3.2 shifts NVIDIA's generated frame in time at 2x (confirmed in game at t = 0; 0.25/0.75 not yet verified by capture)
- [ ] Phase 3: presenting the extra frames. Presentation path traced (0.4 to 0.6); prototype 0.7 is the first 3x attempt (F10), in-game check pending

## Rules for this project

- No NVIDIA binaries, or data decoded from them, are committed or released.
- Patches are applied in memory only, to builds the mod recognises, and are
  restored on unload. Unknown builds are left untouched.
- Single-player use only. Do not use with anti-cheat-protected online play.
