#!/usr/bin/env python3
"""Offline inspector for NVIDIA's NvPresent64.dll (the Smooth Motion runtime).

Reads the file as data. It never loads, executes or modifies the DLL.
Python 3.8+, standard library only (zstd-compressed kernels need the `zstd`
command line tool; LZ4 is handled here).

What it answers, in order:
  1. Which build is this (hash, size, timestamp)?
  2. What does it export and import (the hook surface)?
  3. Which CUDA kernels does it carry, for which GPU architectures, and is
     their PTX source present (PTX is editable text; a bare cubin is not)?
  4. For every PTX kernel: its parameters, and how many compiled-in 0.5
     blend weights it has. A kernel that blends two frames with a constant
     0.5 while also taking a float parameter is the multi-frame candidate.
  5. Does NvPresent use NVIDIA's driver flip metering for pacing?

Usage:
    python inspect_nvpresent.py ref/NvPresent64.dll --dump out/kernels

The gate/config signatures are a port of tools/inspect_nvp.py from
ItsAdeline/NVSmooth30 (MIT). The fatbin entry layout follows midpoint.hpp in
mavismmg/MFGAdaUnlock-RenoDx and dashdogy/MFG-Unlocker (both MIT).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

FATBIN_MAGIC = b"\x50\xed\x55\xba"  # 0xBA55ED50, little endian
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
ELF_MAGIC = b"\x7fELF"
KIND_NAMES = {1: "PTX", 2: "cubin", 8: "NVVM-IR", 16: "mercury", 32: "obfuscated-PTX", 64: "compressed-PTX"}

IMAGE_SCN_MEM_EXECUTE = 0x20000000
IMAGE_SCN_MEM_WRITE = 0x80000000

# Reference build NVSmooth30 was written against.
NVSMOOTH30_REFERENCE_SHA256 = "cd395d58f41c6e393c31a9898f2be3da83f7bc109a2228935c23e8c89b944c15"
# DLSS-G temporal kernels that the MFG unlock mods rewrite (entry name, PTX bytes).
DLSSG_TEMPORAL_ENTRIES = {"main_kernel": 99362, "Kernel_EstimateIntermMvecsScatter": 99626}
# nvapi_QueryInterface ids used by Streamline's flip metering (from MFG-Unlocker).
NVAPI_IDS = {"NvAPI_D3D12_SetFlipConfig": 0xF3148C42, "flip-config caps (V2)": 0x6194B19D}

STRING_KEYWORDS = [
    "cuModule", "cuLaunch", "cuCtx", "cuMem", "cuGraphics", "cuD3D", "cuStream", "nvcuda",
    "NVPRESENT", "NVP_", "Smooth", "smooth", "FlipConfig", "flip", "Flip", "meter",
    "interp", "Interp", "Mvec", "mvec", "flow", "Flow", "warp", "Warp", "inpaint", "Inpaint",
    "kernel", "Kernel", "dlfg", "DLFG", "FrameGen", "framegen", "numFrames", "MultiFrame",
    "pacing", "Pacing", "pacer", "Pacer", "latency", "Latency", "Reflex", "VSync", "vsync",
    "SwapChain", "Present", "Vulkan", "D3D11", "D3D12", "LOG_LEVEL", "debug", "Debug",
]


def u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def cstr(b, o, limit=512):
    end = b.find(b"\0", o, o + limit)
    if end < 0:
        end = min(len(b), o + limit)
    return b[o:end].decode("latin-1")


# --------------------------------------------------------------------------- PE

class PE:
    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 0x40 or data[:2] != b"MZ":
            raise ValueError("not a PE image")
        pe = u32(data, 0x3C)
        if data[pe:pe + 4] != b"PE\0\0":
            raise ValueError("not a PE image")
        self.machine = u16(data, pe + 4)
        nsections = u16(data, pe + 6)
        self.timestamp = u32(data, pe + 8)
        opt_size = u16(data, pe + 20)
        self.opt = pe + 24
        if u16(data, self.opt) != 0x20B:
            raise ValueError("not PE32+ (64-bit)")
        self.image_base = u64(data, self.opt + 24)
        self.size_of_image = u32(data, self.opt + 56)
        self.sections = []
        table = self.opt + opt_size
        for i in range(nsections):
            s = table + 40 * i
            self.sections.append({
                "name": data[s:s + 8].rstrip(b"\0").decode("latin-1"),
                "vsize": u32(data, s + 8), "va": u32(data, s + 12),
                "rsize": u32(data, s + 16), "raw": u32(data, s + 20),
                "flags": u32(data, s + 36),
            })

    def directory(self, index):
        o = self.opt + 112 + 8 * index
        return u32(self.data, o), u32(self.data, o + 4)

    def section_of_rva(self, rva):
        for s in self.sections:
            if s["va"] <= rva < s["va"] + max(s["vsize"], s["rsize"]):
                return s
        return None

    def off(self, rva):
        s = self.section_of_rva(rva)
        if s is None or rva - s["va"] >= s["rsize"]:
            raise ValueError("unmapped RVA 0x%x" % rva)
        return s["raw"] + (rva - s["va"])

    def rva_of_offset(self, offset):
        for s in self.sections:
            if s["raw"] <= offset < s["raw"] + s["rsize"]:
                return s["va"] + (offset - s["raw"])
        return None

    def exports(self):
        rva, size = self.directory(0)
        if not rva:
            return {}
        d = self.off(rva)
        (_, _, _, _, _, _, nfuncs, nnames, funcs, names, ordinals) = struct.unpack_from("<IIHHIIIIIII", self.data, d)
        out = {}
        for i in range(nnames):
            name = cstr(self.data, self.off(u32(self.data, self.off(names) + 4 * i)))
            ordinal = u16(self.data, self.off(ordinals) + 2 * i)
            if ordinal < nfuncs:
                out[name] = u32(self.data, self.off(funcs) + 4 * ordinal)
        return out

    def _thunks(self, rva):
        names = []
        if not rva:
            return names
        o = self.off(rva)
        while o + 8 <= len(self.data):
            v = u64(self.data, o)
            if v == 0:
                break
            if v >> 63:
                names.append("#%d" % (v & 0xFFFF))
            else:
                try:
                    names.append(cstr(self.data, self.off(v & 0x7FFFFFFF) + 2))
                except ValueError:
                    names.append("?")
            o += 8
        return names

    def imports(self):
        out = {}
        rva, _ = self.directory(1)
        if rva:
            o = self.off(rva)
            while True:
                oft, _, _, name, ft = struct.unpack_from("<IIIII", self.data, o)
                if not name and not ft:
                    break
                out.setdefault(cstr(self.data, self.off(name)), []).extend(self._thunks(oft or ft))
                o += 20
        return out

    def delay_imports(self):
        out = {}
        rva, _ = self.directory(13)
        if rva:
            o = self.off(rva)
            while True:
                _, name, _, _, int_rva, _, _, _ = struct.unpack_from("<IIIIIIII", self.data, o)
                if not name:
                    break
                out.setdefault(cstr(self.data, self.off(name)), []).extend(self._thunks(int_rva))
                o += 32
        return out


# ------------------------------------------------------------------ NVSmooth30

def find_gates(pe: PE):
    """cmp dword [rcx+14h], 2|3 followed within 40 bytes by SETGE r8 (arch gate)."""
    hits = []
    for s in pe.sections:
        if not s["flags"] & IMAGE_SCN_MEM_EXECUTE:
            continue
        blob = pe.data[s["raw"]:s["raw"] + s["rsize"]]
        for imm in (b"\x03", b"\x02"):
            start = 0
            while True:
                i = blob.find(b"\x83\x79\x14" + imm, start)
                if i < 0:
                    break
                start = i + 1
                for j in range(4, 40):
                    t = blob[i + j:i + j + 4]
                    if len(t) < 4:
                        break
                    plain = t[0] == 0x0F and t[1] == 0x9D and (t[2] & 0xF8) == 0xC0
                    rex = (t[0] & 0xF0) == 0x40 and t[1] == 0x0F and t[2] == 0x9D and (t[3] & 0xF8) == 0xC0
                    if plain or rex:
                        hits.append((s["va"] + i + 3, s["va"] + i + j, imm[0]))
                        break
    return hits


def find_config(pe: PE, init_rva):
    """First `lea rcx, [rip+disp]` in NVP_Init_D3D that targets writable data."""
    base = pe.off(init_rva)
    for i in range(0, 90):
        if pe.data[base + i:base + i + 3] != b"\x48\x8d\x0d":
            continue
        disp = struct.unpack_from("<i", pe.data, base + i + 3)[0]
        target = init_rva + i + 7 + disp
        s = pe.section_of_rva(target)
        if s and s["flags"] & IMAGE_SCN_MEM_WRITE and target + 0x12A6 <= s["va"] + s["vsize"]:
            return target
    return None


# ---------------------------------------------------------------------- fatbin

def lz4_block_decompress(src: bytes, dst_size: int) -> bytes:
    out = bytearray()
    i, n = 0, len(src)
    while i < n:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                if i >= n:
                    raise ValueError("lz4: truncated literal length")
                e = src[i]
                i += 1
                lit += e
                if e != 255:
                    break
        if lit > n - i:
            raise ValueError("lz4: literal run past end")
        out += src[i:i + lit]
        i += lit
        if i == n:
            break
        if n - i < 2:
            raise ValueError("lz4: truncated offset")
        back = src[i] | (src[i + 1] << 8)
        i += 2
        if back == 0 or back > len(out):
            raise ValueError("lz4: bad offset")
        match = 4 + (token & 15)
        if (token & 15) == 15:
            while True:
                if i >= n:
                    raise ValueError("lz4: truncated match length")
                e = src[i]
                i += 1
                match += e
                if e != 255:
                    break
        start = len(out) - back
        if back >= match:
            out += out[start:start + match]
        else:  # overlapping copy
            for k in range(match):
                out.append(out[start + k])
        if len(out) > dst_size:
            raise ValueError("lz4: output larger than declared")
    if len(out) != dst_size:
        raise ValueError("lz4: got %d bytes, header says %d" % (len(out), dst_size))
    return bytes(out)


def zstd_decompress(src: bytes) -> bytes:
    try:
        r = subprocess.run(["zstd", "-d", "-c"], input=src, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        raise ValueError("zstd-compressed entry; install the `zstd` command line tool to read it")
    except subprocess.CalledProcessError as e:
        raise ValueError("zstd failed: " + e.stderr.decode("latin-1", "replace").strip())
    return r.stdout


def _plausible_entry(data, p, end):
    if p + 0x20 > end:
        return False
    hdr, size = u32(data, p + 4), u64(data, p + 8)
    return u16(data, p) in KIND_NAMES and 0x20 <= hdr <= 0x400 and p + hdr + size <= end


def parse_fatbin(data: bytes, start: int):
    """Returns (total_size, entries) or None when `start` is not a fatbin."""
    if start + 16 > len(data):
        return None
    hdr_size, payload = u16(data, start + 6), u64(data, start + 8)
    if not 0x10 <= hdr_size <= 0x100 or payload == 0 or payload > 64 << 20:
        return None
    end = start + hdr_size + payload
    if end > len(data):
        return None
    entries, p = [], start + hdr_size
    while p < end:
        if not _plausible_entry(data, p, end):
            break
        hdr, size = u32(data, p + 4), u64(data, p + 8)
        e = {
            "offset": p, "kind": u16(data, p), "header": hdr, "size": size,
            "compressed": u32(data, p + 16), "minor": u16(data, p + 24), "major": u16(data, p + 26),
            "arch": u32(data, p + 28),
            "flags": u64(data, p + 40) if hdr >= 48 else 0,
            "raw_size": u64(data, p + 56) if hdr >= 64 else 0,
            "payload": p + hdr,
        }
        entries.append(e)
        nxt = p + hdr + size
        if nxt < end and not _plausible_entry(data, nxt, end):
            aligned = (nxt + 7) & ~7
            if _plausible_entry(data, aligned, end):
                nxt = aligned
        p = nxt
    if not entries:
        return None
    return hdr_size + payload, entries


def entry_bytes(data: bytes, e) -> bytes:
    raw = data[e["payload"]:e["payload"] + e["size"]]
    if not e["compressed"]:
        return raw
    packed = raw[:e["compressed"]]
    if packed[:4] == ZSTD_MAGIC:
        return zstd_decompress(packed)
    return lz4_block_decompress(packed, e["raw_size"])


def find_fatbins(data: bytes):
    found, cursor, skip_until = [], 0, 0
    while True:
        cursor = data.find(FATBIN_MAGIC, cursor)
        if cursor < 0:
            return found
        if cursor >= skip_until:
            parsed = parse_fatbin(data, cursor)
            if parsed:
                total, entries = parsed
                found.append({"offset": cursor, "size": total, "entries": entries})
                skip_until = cursor + total
        cursor += 4


# ------------------------------------------------------------------- PTX / ELF

ENTRY_RE = re.compile(r"^\s*(?:\.(?:visible|weak|extern)\s+)*\.entry\s+([A-Za-z_$][\w$]*)\s*\(", re.M)
PARAM_RE = re.compile(r"\.param\s+(?:\.align\s+\d+\s+)?\.(\w+)\s+([A-Za-z_$][\w$]*)(?:\[(\d+)\])?")
LDPARAM_RE = re.compile(r"ld\.param\.(\w+)\s+%\w+\s*,\s*\[([A-Za-z_$][\w$]*)(?:\+(\d+))?\]")


def analyze_ptx(text: str):
    kernels = []
    matches = list(ENTRY_RE.finditer(text))
    for n, m in enumerate(matches):
        body_end = matches[n + 1].start() if n + 1 < len(matches) else len(text)
        close = text.find(")", m.end())
        params = []
        for p in PARAM_RE.finditer(text, m.end(), close if close > 0 else m.end()):
            params.append((p.group(2), p.group(1), int(p.group(3)) if p.group(3) else None))
        body = text[m.end():body_end]
        midpoint_mul = 0
        for line in body.splitlines():
            line = line.strip()
            if line.startswith("mul.ftz.f32") and line.endswith("0f3F000000;"):
                midpoint_mul += 1
        loads = {}
        for l in LDPARAM_RE.finditer(body):
            loads.setdefault((l.group(2), int(l.group(3) or 0)), set()).add(l.group(1))
        kernels.append({
            "name": m.group(1), "params": params, "bytes": len(body),
            "midpoint_mul": midpoint_mul, "half_constants": body.count("0f3F000000"),
            "float_loads": sorted((k[0], k[1]) for k, t in loads.items() if "f32" in t),
            "loads": len(loads),
        })
    return kernels


def analyze_cubin(blob: bytes):
    if blob[:4] != ELF_MAGIC or len(blob) < 0x40 or blob[4] != 2:
        return None
    shoff, flags = u64(blob, 0x28), u32(blob, 0x30)
    shentsize, shnum, shstrndx = u16(blob, 0x3A), u16(blob, 0x3C), u16(blob, 0x3E)
    if not shoff or shentsize < 0x40 or shoff + shnum * shentsize > len(blob) or shstrndx >= shnum:
        return {"flags": flags, "kernels": {}}
    stroff = u64(blob, shoff + shstrndx * shentsize + 0x18)
    kernels = {}
    for i in range(shnum):
        s = shoff + i * shentsize
        name = cstr(blob, stroff + u32(blob, s))
        size = u64(blob, s + 0x20)
        for prefix, key in ((".text.", "text"), (".nv.constant0.", "const0"), (".nv.shared.", "shared")):
            if name.startswith(prefix):
                kernels.setdefault(name[len(prefix):], {})[key] = size
    return {"flags": flags, "kernels": {k: v for k, v in kernels.items() if "text" in v}}


# --------------------------------------------------------------------- strings

def interesting_strings(data: bytes, exclude):
    def excluded(o):
        return any(a <= o < b for a, b in exclude)

    seen, out = set(), []
    for m in re.finditer(rb"[\x20-\x7e]{6,}", data):
        if not excluded(m.start()):
            out.append((m.start(), m.group().decode("latin-1")))
    for m in re.finditer(rb"(?:[\x20-\x7e]\x00){6,}", data):
        if not excluded(m.start()):
            out.append((m.start(), m.group().decode("utf-16-le")))
    keep = []
    for offset, s in sorted(out):
        if s in seen or not any(k in s for k in STRING_KEYWORDS):
            continue
        seen.add(s)
        keep.append((offset, s))
    return keep


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dll", type=Path)
    ap.add_argument("--dump", type=Path, help="write decoded PTX/cubins and the full string list here")
    ap.add_argument("--max-strings", type=int, default=120)
    args = ap.parse_args()

    data = args.dll.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    print("file      %s" % args.dll)
    print("size      %d bytes (%.1f MB)" % (len(data), len(data) / 1048576))
    print("sha256    %s" % sha)
    if sha == NVSMOOTH30_REFERENCE_SHA256:
        print("          = the reference build NVSmooth30 was written against")

    pe = None
    try:
        pe = PE(data)
    except ValueError as e:
        print("pe        %s; scanning as raw data" % e)

    if pe:
        print("built     %s UTC (PE timestamp)" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pe.timestamp)))
        print("\n== sections")
        for s in pe.sections:
            perms = ("x" if s["flags"] & IMAGE_SCN_MEM_EXECUTE else "-") + ("w" if s["flags"] & IMAGE_SCN_MEM_WRITE else "-")
            print("  %-8s rva=+0x%-8x vsize=0x%-8x raw=0x%-8x %s" % (s["name"], s["va"], s["vsize"], s["rsize"], perms))

        exports = pe.exports()
        print("\n== exports (%d)" % len(exports))
        for name, rva in sorted(exports.items(), key=lambda kv: kv[1]):
            print("  +0x%-8x %s" % (rva, name))

        for title, table in (("imports", pe.imports()), ("delay imports", pe.delay_imports())):
            print("\n== %s (%d DLLs)" % (title, len(table)))
            for dll, names in sorted(table.items(), key=lambda kv: kv[0].lower()):
                cuda = [n for n in names if n.startswith("cu")] if "cuda" in dll.lower() else []
                if cuda:
                    print("  %s: %s" % (dll, ", ".join(cuda)))
                else:
                    print("  %s (%d)" % (dll, len(names)))
            hooks = [n for names in table.values() for n in names
                     if n in ("GetProcAddress", "LoadLibraryW", "LoadLibraryA", "LoadLibraryExW", "LoadLibraryExA")]
            if hooks:
                print("  dynamic resolution via: %s" % ", ".join(sorted(set(hooks))))

        print("\n== NVSmooth30 signatures")
        gates = find_gates(pe)
        print("  arch gate candidates: %d %s" % (len(gates), " ".join(
            "cmp_imm=+0x%x(imm=%d),setge=+0x%x" % (c, imm, s) for c, s, imm in gates)))
        init = exports.get("NVP_Init_D3D")
        print("  NVP_Init_D3D: %s" % ("+0x%x" % init if init else "missing"))
        if init:
            cfg = find_config(pe, init)
            print("  config block: %s" % ("+0x%x" % cfg if cfg else "not found"))

        print("\n== NVAPI interface ids in code (driver flip metering)")
        for name, value in NVAPI_IDS.items():
            needle, hits = struct.pack("<I", value), []
            for s in pe.sections:
                if s["flags"] & IMAGE_SCN_MEM_EXECUTE:
                    blob, i = data[s["raw"]:s["raw"] + s["rsize"]], 0
                    while True:
                        i = blob.find(needle, i)
                        if i < 0:
                            break
                        hits.append(s["va"] + i)
                        i += 1
            print("  0x%08X %-28s %s" % (value, name, ("%d site(s): " % len(hits) + " ".join("+0x%x" % h for h in hits[:8])) if hits else "absent"))

    fatbins = find_fatbins(data)
    print("\n== CUDA fatbins (%d)" % len(fatbins))
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
    arch_totals, candidates = {}, []
    for index, fb in enumerate(fatbins):
        rva = pe.rva_of_offset(fb["offset"]) if pe else None
        print("\n  fatbin #%d  file+0x%x%s  %d bytes  %d entries" % (
            index, fb["offset"], "  rva=+0x%x" % rva if rva is not None else "", fb["size"], len(fb["entries"])))
        for e in fb["entries"]:
            kind = KIND_NAMES.get(e["kind"], "kind%d" % e["kind"])
            arch_totals[(kind, e["arch"])] = arch_totals.get((kind, e["arch"]), 0) + 1
            packing = "stored"
            if e["compressed"]:
                head = data[e["payload"]:e["payload"] + 4]
                packing = ("zstd" if head == ZSTD_MAGIC else "lz4") + " %d->%d" % (e["compressed"], e["raw_size"])
            print("    %-6s sm_%-3d v%d.%d  %7d bytes  %s  flags=0x%x" % (
                kind, e["arch"], e["major"], e["minor"], e["size"], packing, e["flags"]))
            try:
                blob = entry_bytes(data, e)
            except ValueError as err:
                print("      (could not decode: %s)" % err)
                continue
            stem = "fatbin%02d_sm%d" % (index, e["arch"])
            if e["kind"] == 1:
                text = blob.rstrip(b"\0").decode("latin-1")
                if args.dump:
                    (args.dump / (stem + ".ptx")).write_text(text, encoding="latin-1", newline="")
                for k in analyze_ptx(text):
                    struct_params = ["%s[%d]" % (n, c) for n, t, c in k["params"] if c]
                    print("      .entry %s  params=%d %s  body=%d bytes  ld.param sites=%d" % (
                        k["name"], len(k["params"]), " ".join(struct_params), k["bytes"], k["loads"]))
                    if k["float_loads"]:
                        print("        f32 parameter loads: %s" % ", ".join("%s+%d" % fl for fl in k["float_loads"][:24]))
                    if k["half_constants"]:
                        print("        0.5 constants: %d, of which blend-style `mul.ftz.f32 ..., 0.5`: %d" % (
                            k["half_constants"], k["midpoint_mul"]))
                    known = DLSSG_TEMPORAL_ENTRIES.get(k["name"])
                    if known:
                        print("        ** same entry name as DLSS-G's temporal kernel (its sm_89 PTX is %d bytes; this entry's PTX is %d)" % (known, len(blob)))
                    if k["midpoint_mul"]:
                        candidates.append((index, e["arch"], k["name"], k["midpoint_mul"], len(k["float_loads"])))
            elif e["kind"] == 2:
                if args.dump:
                    (args.dump / (stem + ".cubin")).write_bytes(blob)
                info = analyze_cubin(blob)
                if info is None:
                    print("      (payload is not an ELF cubin)")
                else:
                    for name, v in sorted(info["kernels"].items()):
                        print("      kernel %s  text=%d  constant0=%d" % (name, v.get("text", 0), v.get("const0", 0)))

    print("\n== summary")
    if arch_totals:
        print("  entries by type/arch: " + ", ".join("%s sm_%d x%d" % (k, a, n) for (k, a), n in sorted(arch_totals.items())))
        have_ptx = sorted({a for (k, a) in arch_totals if k == "PTX"})
        print("  PTX source present for: %s" % (", ".join("sm_%d" % a for a in have_ptx) if have_ptx else
              "none (kernels ship as compiled cubins only; text rewriting is not available)"))
    else:
        print("  no fatbins found: kernels may be stored another way, or in another file")
    if candidates:
        print("  multi-frame candidates (kernels blending with a compiled-in 0.5):")
        for index, arch, name, count, floats in candidates:
            print("    fatbin #%d sm_%d %s: %d midpoint multiplies, %d f32 parameter loads" % (index, arch, name, count, floats))
    else:
        print("  no PTX kernel with a compiled-in 0.5 blend was found")

    exclude = [(fb["offset"], fb["offset"] + fb["size"]) for fb in fatbins]
    strings = interesting_strings(data, exclude)
    print("\n== strings of interest (%d; showing %d)" % (len(strings), min(len(strings), args.max_strings)))
    for offset, s in strings[:args.max_strings]:
        print("  file+0x%-8x %s" % (offset, s[:160]))
    if args.dump:
        (args.dump / "strings.txt").write_text(
            "\n".join("file+0x%x\t%s" % (o, s) for o, s in strings), encoding="utf-8")
        print("\nfull string list and decoded kernels written to %s" % args.dump)


if __name__ == "__main__":
    sys.exit(main())
