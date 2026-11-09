# Notices

This project contains no NVIDIA software. It works from the NVIDIA driver
already installed on the user's machine. NVIDIA, GeForce, CUDA and Smooth
Motion are trademarks of NVIDIA Corporation; this project is not affiliated
with or endorsed by NVIDIA.

## Prior work

Techniques and file-format knowledge come from these MIT-licensed projects:

- [ItsAdeline/NVSmooth30](https://github.com/ItsAdeline/NVSmooth30): the
  NvPresent gate and configuration signatures in `tools/inspect_nvpresent.py`
  are a port of its `tools/inspect_nvp.py`, and its hook design informs
  `docs/DESIGN.md`.
- [dashdogy/MFG-Unlocker](https://github.com/dashdogy/MFG-Unlocker): the
  midpoint (temporal) fix and its flip-metering and pacing research.
- [mavismmg/MFGAdaUnlock-RenoDx](https://github.com/mavismmg/MFGAdaUnlock-RenoDx):
  fatbin entry layout and the compact form of the temporal fix.

## Third-party code

- `third_party/minhook`: [MinHook](https://github.com/TsudaKageyu/minhook) by
  Tsuda Kageyu, BSD 2-Clause (see `third_party/minhook/LICENSE.txt`).
