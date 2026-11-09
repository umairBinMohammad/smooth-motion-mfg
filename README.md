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
- [ ] Phase 1: runtime tracer (log-only). Built, see [docs/TRACER.md](docs/TRACER.md); first trace pending
- [ ] Phase 2+: see `docs/DESIGN.md`

## Rules for this project

- No NVIDIA binaries, or data decoded from them, are committed or released.
- Patches are applied in memory only, to builds the mod recognises, and are
  restored on unload. Unknown builds are left untouched.
- Single-player use only. Do not use with anti-cheat-protected online play.
