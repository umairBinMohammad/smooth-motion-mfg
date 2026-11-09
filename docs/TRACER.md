# Tracer and phase 2 prototype

A `version.dll` that records what `NvPresent64.dll` asks CUDA to do. It
patches nothing in NvPresent or the game. Up to 0.2 it was log-only; from 0.3
it also inserts one kernel of its own (below).

## What it logs (`smm_trace.log`, beside the game executable)

- `module #N load rva=+0x...`: a kernel module NvPresent loaded. The rva
  matches the fatbin list in the inspector report, which gives the kernel name.
- `function <name> ... total=<bytes>`: a kernel looked up, with the size of
  its parameter block.
- `launch <seq> <name> grid=... params[..]=<hex>`: a kernel launch with its
  first parameter block. Written for the first 800 launches, then for 80
  launches out of every 6000.
- `warp +48 new value`: `warp_coarse_kernel`'s suspected displacement scale
  changed to a value not seen before.
- `capture begin/end`, `graph launch`: NvPresent recorded its launches into a
  CUDA graph and is replaying it. If this appears, per-frame launches will not.
- `array create`, `external image`: sizes of the images NvPresent works on.
- `summary launches/s: ...` every 5 seconds: launch rate per kernel.

## Capture (tracer 0.2)

Press **F9** in game to dump two consecutive generated frames' network data
into `smm_capture\` beside the log: the network's 256x144 output and the
256x144 inputs. Up to 8 presses per run. Each press also logs how long the
generation graph took on the GPU.

## Time shift (prototype 0.3)

While NvPresent records its per-frame graph, the DLL inserts one small kernel
(hand-written PTX) before `warp_coarse_kernel`. It scales the network's flow
channels so NVIDIA's warp builds the frame for time t instead of the midpoint.
Still 2x: one generated frame per real frame, just at a different moment.

**F10** cycles t, with a beep (higher pitch each step):

| t | Expect |
|---|---|
| 0.5 (start) | Normal Smooth Motion; the data is unchanged |
| 0.25 | Generated frame sits close to the previous real frame: uneven, slightly stuttery motion |
| 0.75 | Close to the current real frame: uneven the other way |
| 0 | Generated frame equals the previous real frame: looks like Smooth Motion is off |

Blend weights are not adjusted yet, so some ghosting at 0.25/0.75 is expected.
If the kernel fails to compile, the log says so and nothing is inserted.

## Use

1. Copy `build\version.dll` next to the game's main executable.
2. Make sure Smooth Motion is on for the game, as usual.
3. Play for a minute or two. Move the camera slowly, then fast. If the game
   lets you, change resolution once.
4. Quit, and copy `smm_trace.log` into this project's `out\` folder.
5. Delete `version.dll` from the game folder.

If no `smm_trace.log` appears, the game does not load `version.dll` from its
own folder and the tracer needs a different proxy name.

Setting the environment variable `SMM_LOG_DIR` writes the log there instead.

## Build

`./build.sh` with [llvm-mingw](https://github.com/mstorsjo/llvm-mingw) on the
PATH (or `TOOLCHAIN=/path/to/llvm-mingw/bin ./build.sh`). Hooking uses
[MinHook](https://github.com/TsudaKageyu/minhook), vendored in `third_party/`.

Same caution as any injected DLL: single-player sessions only.
