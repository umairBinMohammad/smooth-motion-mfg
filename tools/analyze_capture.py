#!/usr/bin/env python3
"""Check what the Smooth Motion network outputs, from tracer F9 captures.

Reads smm_capture/setN_P_output.bin (256x144, 16 half floats per pixel) and
setN_P_coarse.bin (256x144, 8 half floats: previous RGB, current RGB, 2 unused).

Tests the reading "channels 0,1 = flow from the generated frame into the
previous frame, 2,3 = flow into the current frame (coarse pixels)" by warping
both real frames with those flows and measuring how well they agree, at the
midpoint and at other times t with the flows rescaled by 2t and 2(1-t).

Needs numpy.  Usage: python analyze_capture.py out/smm_capture
"""
import glob
import os
import sys

import numpy as np

W, H = 256, 144


def load(path, channels):
    return np.fromfile(path, dtype=np.float16)[:W * H * channels].reshape(H, W, channels).astype(np.float64)


def sample(img, x, y):
    x = np.clip(x, 0, W - 1.001)
    y = np.clip(y, 0, H - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "out/smm_capture"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    inner = (slice(8, -8), slice(8, -8))
    print("set   motion  unaligned  t=0.50  t=0.33  t=0.67  t=0.25  t=0.75   prev(next)==cur")
    for out_path in sorted(glob.glob(os.path.join(folder, "set*_0_output.bin"))):
        stem = out_path[:-len("_0_output.bin")]
        o = load(out_path, 16)
        c = load(stem + "_0_coarse.bin", 8)
        prev, cur = c[..., 0:3], c[..., 3:6]
        row = ["%-5s" % os.path.basename(stem), "%6.2f" % np.abs(o[..., 0:2]).mean(),
               "%9.4f" % np.abs(prev - cur)[inner].mean()]
        for t in (0.5, 1 / 3, 2 / 3, 0.25, 0.75):
            a = sample(prev, xx + 2 * t * o[..., 0], yy + 2 * t * o[..., 1])
            b = sample(cur, xx + 2 * (1 - t) * o[..., 2], yy + 2 * (1 - t) * o[..., 3])
            row.append("%6.4f" % np.abs(a - b)[inner].mean())
        nxt = stem + "_1_coarse.bin"
        if os.path.exists(nxt):
            row.append("  %s" % bool(np.array_equal(load(nxt, 8)[..., 0:3], cur)))
        print("  ".join(row))


if __name__ == "__main__":
    main()
