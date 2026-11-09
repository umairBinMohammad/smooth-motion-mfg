# Smooth Motion MFG: design notes

Last updated 2026-11-10. Status: phase 0 (static inspection) done for driver
616.56, see [findings/616.56.md](findings/616.56.md). Phase 1 (runtime tracer)
is next.

**Phase 1 so far** ([trace](findings/616.56-trace.md)): NvPresent replays a
recorded CUDA graph per frame, the network runs at a fixed 1024x576 and emits
a 256x144 6-channel field, and the warp parameter at +48 is only a resolution
ratio. Multi-frame now depends on decoding that field and writing our own
warp kernel in PTX.

**Phase 0 in short:** kernels are compiled only (no PTX), there is no visible
time input, and NvPresent already paces with hardware flip metering in fixed
batches of two. S1 as planned is out; S1b below is the new lead, and the
tracer decides between S1b and S2.

## 1. Goal

Smooth Motion inserts one generated frame between two rendered frames. This
mod makes it insert two or more (3x, 4x), evenly spaced in time and evenly
paced on screen, at close to the cost of the stock 2x mode.

First target: Ghost Recon Breakpoint, DX11, RTX 4060 laptop, 144 Hz VRR
panel. 3x from a 48 fps base fills that panel exactly, so 3x comes first.

## 2. What the existing mods establish

All three are MIT licensed. Commits are the ones read for these notes.

| Project | Commit | Relevant to us |
|---|---|---|
| [ItsAdeline/NVSmooth30](https://github.com/ItsAdeline/NVSmooth30) | 21fa435 (2026-09-21) | How to hook NvPresent |
| [dashdogy/MFG-Unlocker](https://github.com/dashdogy/MFG-Unlocker) | bb9fb80 (2026-11-06) | Temporal fix (original), pacing |
| [mavismmg/MFGAdaUnlock-RenoDx](https://github.com/mavismmg/MFGAdaUnlock-RenoDx) | 1226d4f (2026-11-07) | Temporal fix (compact form), pacing policy |

### 2.1 Frames at the right time (the "midpoint fix")

Source: `MFGAdaUnlock-RenoDx/src/addons/mfgunlock/midpoint.hpp`, technique from
`MFG-Unlocker/source/native/midpoint_fix.cpp`.

- Unlocking 3x/4x in DLSS-G gave the right number of frames but identical
  content: the interpolation kernel blends with a compiled-in 0.5, so every
  generated frame landed at the midpoint.
- The kernel is `main_kernel` (renamed `Kernel_EstimateIntermMvecsScatter` in
  DLSS-G 310.9). It takes a 144-byte parameter struct whose float at offset 32
  is the temporal position, which the stock code ignores for blending.
- The fatbin ships the kernel as LZ4-compressed PTX (text) plus a compiled
  sm_89 cubin. The fix decompresses the PTX, loads the float at `param_0+32`,
  replaces the 104 `mul.ftz.f32 ..., 0.5` weights with `t` and `1 - t`, and
  re-emits the fatbin uncompressed and truncated after the PTX entry so the
  driver JIT-compiles the edited text instead of loading the cubin.
- Fatbin entry layout used there: kind u16 @0, header size u32 @4, payload size
  u64 @8, compressed size u32 @16, arch u32 @28, flags u64 @40, raw size u64 @56.

**Why this matters for us:** if NvPresent's kernels come from the same family
(same shape, PTX present, a float temporal input), the hardest part of
multi-frame, generating at arbitrary `t`, is a text edit. If they ship as
cubins only, or the network has no temporal input, it is not.

### 2.2 The NvPresent hook surface

Source: `NVSmooth30/src/nvpresent.cpp`, `dxgi_hooks.cpp`, `d3d11_bridge.cpp`.

- `NvPresent64.dll` lives in the driver store next to the display driver and
  exports `NVP_Init_D3D`.
- An architecture gate (`cmp dword [rcx+14h], 2|3` then `setge`) and a config
  block reached from `NVP_Init_D3D` decide whether Smooth Motion runs. We do
  not need to touch either on RTX 40, where the driver enables it officially.
- NvPresent reaches CUDA through `cuModuleLoadData` in `nvcuda.dll`. Depending
  on the build that pointer sits in the import table, the delay-import table,
  or a private dispatch slot filled via `GetProcAddress`. NVSmooth30 handles
  all four cases; our tracer can reuse the same discovery order.
- NvPresent takes over presentation either by wrapping the swapchain object
  (wrapper pointer historically at +0x18 behind the public swapchain) or by
  patching `Present`/`Present1` in the shared DXGI vtable (slots 8 and 22).
- For D3D11 games the reference path copies the back buffer through a shared
  texture into a D3D12 device. That suggests NvPresent's own presentation is
  D3D12 even for DX11 games, which matters for pacing (2.3). To be confirmed.

One difference from NVSmooth30: on RTX 30 the mod has to load and initialise
NvPresent itself. On RTX 40 the driver does that, so we only need to notice
the module arriving (loader notification) and attach.

### 2.3 Pacing

Sources: `MFG-Unlocker/source/native/xess/sdk_pacing.{h,cpp}`,
`sdk_pacing_meter.h`, `ampere_flip_metering.h`; `NVSmooth30/src/pacer.cpp`.

Three techniques exist, from best to simplest:

1. **Driver flip metering.** `NvAPI_D3D12_SetFlipConfig`, resolved through
   `nvapi_QueryInterface(0xF3148C42)` (capability query `0x6194B19D`). The
   caller submits a whole batch of frames early and tells the driver
   `nFramesPerBatch` and `timePerBatchMs`; the display hardware spaces the
   flips. Parameter block (24 bytes, version `0x20018`): `version,
   nFramesPerBatch, flipConfigFeedbackMask (out), nPresentedFramesPerBatch,
   timePerBatchMs, minControlFrameTimeMs`. Feedback bit 1 means no independent
   flip, in which case metering is not in effect. MFG-Unlocker's notes say
   Streamline uses this path on RTX 40, and that consumer RTX 30 drivers accept
   the call. It needs a D3D12 presentation queue.
2. **Gated flip queue.** Each output's `Present` waits on a GPU gate that a
   meter thread opens on a clock locked to the game's hand-over rate.
   `sdk_pacing_meter.h` documents the hard-won rules: interval = smoothed batch
   hand-over time / outputs; isolated long frames are hitches and are not
   averaged; the first output of a batch flips at readiness (at least 0.9 of an
   interval after the previous flip) rather than behind a jitter buffer; guard
   against lock-in, where the meter ends up pacing the game.
3. **CPU pacer.** `QueryPerformanceCounter`, sleep to within about 0.7 ms, then
   spin. Good enough for a base-frame cap, not for spacing generated frames.

Plan: cap base frames with (3), space outputs with (1) when the driver accepts
it, fall back to (2).

## 3. Open questions (phase 0 answers 1 to 4)

1. Do NvPresent's fatbins carry PTX for sm_89, or cubins only?
2. Is there a kernel that blends with a compiled-in 0.5 and also receives a
   float parameter that could act as `t`?
3. How does this build reach CUDA (import, delay import, `GetProcAddress`)?
4. Does NvPresent contain the flip-metering interface ids, i.e. does stock
   Smooth Motion already pace through the driver?
5. Runtime only: how many kernel launches happen per generated frame, which of
   them are motion estimation (run once per pair) and which are synthesis
   (run once per output)?
6. Runtime only: where does the generated frame get presented, and can that
   path be driven more than once per rendered frame?
7. Does Windows NvPresent honour `NVPRESENT_LOG_LEVEL` (documented for the
   Linux Vulkan layer)? If so we get NVIDIA's own trace for free.

## 4. Generation strategies

| | S1: re-run synthesis at t = k/N | S2: cascade |
|---|---|---|
| Needs | PTX present and a usable temporal input (Q1, Q2) | Nothing from the kernels |
| Multipliers | Any (3x, 4x, ...) | 4x only |
| Cost per rendered frame | 1 motion estimate + (N-1) synthesis passes | 3 full passes |
| Quality | Each output derived from real frames | Outputs derived from generated frames |

S1 is the goal. S2 is the fallback if Q1 or Q2 comes back negative. If both
fail on the 4060's budget, the project stops there.

Phase 0 answered Q1 and Q2 negatively for driver 616.56, which rules out S1
as a parameter change. New candidate:

**S1b: scaled warp.** Run the network once per rendered pair as now, then run
`warp_coarse_kernel` N-1 times with its displacement scaled by 2t. Cheap (the
warp kernel is a few hundred instructions; the network is the cost), but only
correct for straight-line motion, and only possible if the parameter at
warp_coarse +48 really is a displacement scale. The tracer checks this.

## 5. Planned architecture

```
game -> DXGI Present
          |
   [our module: version.dll proxy or .asi]
          |-- loader notification: NvPresent64.dll arrives
          |-- CUDA slots in NvPresent:
          |     cuModuleLoadData  -> swap in the temporal-corrected kernel
          |     cuLaunchKernel    -> set t per output; trace or replay
          |-- present scheduler: N-1 generated outputs + the real frame,
          |     spaced by flip metering or the gated clock
          v
   NvPresent (unchanged on disk) -> display
```

Everything is in memory, recognised builds only, restored on unload.

## 6. Phases

| Phase | Deliverable | Proves |
|---|---|---|
| 0 | Inspector report on the installed DLL (done, 616.56) | Q1 to Q4 answered: no PTX, no time input, flip metering present |
| 1 | Log-only tracer build: module loads, kernel launches and presents per frame | Q5 to Q7; nothing is modified |
| 2 | Temporal patch with a fixed `t` other than 0.5 at stock 2x | The temporal input works. At t = 0.25 the generated frame should sit visibly early |
| 3 | Multi-frame generation, unpaced | Extra frames exist and differ from each other |
| 4 | Pacing: flip metering, then gated-clock fallback | Flat frame-time graph (PresentMon `MsBetweenDisplayChange`) |
| 5 | Hardening: build recognition, fail closed, config, base-frame cap | Safe to hand to other people |

Measure with PresentMon or FrameView display-change timings, not an in-game
FPS counter: present-based counters do not see how flips are spaced.

## 7. Constraints

- No NVIDIA binaries or decoded kernels in the repo or in releases. The mod
  works from the user's installed driver.
- Driver updates will break recognition; the mod must refuse cleanly.
- NvPresent builds differ a lot between driver branches (file size alone has
  ranged from about 7 MB to 76 MB), so every finding is per build.
