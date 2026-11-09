#!/usr/bin/env python3
"""Summarise a CUDA cubin (ELF) written out by inspect_nvpresent.py --dump.

For each kernel: its sections, the parameter block from .nv.info
(EIATTR_KPARAM_INFO / CBANK_PARAM_SIZE), and a heuristic SASS scan that lists
every instruction reading the parameter block, and every 32-bit float
immediate. The scan only decodes the operand fields of the sm_8x/sm_9x
128-bit encoding; opcodes are shown as raw numbers. Without nvdisasm this is
a map of where to look, not a disassembly.

Usage: python cubin_info.py out/kernels/fatbin36_sm89.cubin [...]
"""
import struct
import sys

EIATTR = {0x0A: "PARAM_CBANK", 0x17: "KPARAM_INFO", 0x19: "CBANK_PARAM_SIZE",
          0x1C: "EXIT_INSTR_OFFSETS", 0x1B: "MAXREG_COUNT"}


def u16(b, o): return struct.unpack_from("<H", b, o)[0]
def u32(b, o): return struct.unpack_from("<I", b, o)[0]
def u64(b, o): return struct.unpack_from("<Q", b, o)[0]


def sections(b):
    shoff, ent, n, si = u64(b, 0x28), u16(b, 0x3A), u16(b, 0x3C), u16(b, 0x3E)
    strings = u64(b, shoff + si * ent + 0x18)
    out = {}
    for i in range(n):
        s = shoff + i * ent
        start = strings + u32(b, s)
        out[b[start:b.index(b"\0", start)].decode()] = (u64(b, s + 0x18), u64(b, s + 0x20))
    return out


def nv_info(b, off, size):
    p = off
    while p < off + size:
        fmt, attr = b[p], b[p + 1]
        if fmt == 4:
            n = u16(b, p + 2)
            yield attr, b[p + 4:p + 4 + n]
            p += 4 + n
        else:
            yield attr, b[p + 2:p + 4]
            p += 4


def main():
    for path in sys.argv[1:]:
        b = open(path, "rb").read()
        if b[:4] != b"\x7fELF":
            print(path, ": not an ELF cubin")
            continue
        flags = u32(b, 0x30)
        print("== %s  sm_%d" % (path, flags & 0xFF))
        secs = sections(b)
        for name, (off, size) in secs.items():
            if not name.startswith(".text."):
                continue
            kernel = name[6:]
            cbank = 0x160
            print("  kernel %s  %d instructions" % (kernel, size // 16))
            info = secs.get(".nv.info." + kernel)
            if info:
                for attr, pay in nv_info(b, *info):
                    if attr == 0x17:
                        _, ordinal, offset, v = struct.unpack("<IHHI", pay)
                        print("    param #%d at +%d, %d bytes" % (ordinal, offset, (v >> 18) & 0x3FFF))
                    elif attr == 0x0A:
                        cbank = u16(pay, 4)
                        print("    parameters live at c[0][%#x]" % cbank)
                    elif attr == 0x1C:
                        print("    exits at", " ".join(hex(u32(pay, i)) for i in range(0, len(pay), 4)))
            reads, floats = {}, []
            for i in range(0, size, 16):
                lo = u64(b, off + i)
                op = lo & 0xFFF
                if (op & 0xF00) == 0xA00 or op in (0xB82, 0xAB9, 0x7AB):
                    bank, coff = (lo >> 54) & 0x1F, ((lo >> 40) & 0x3FFF) * 4
                    if bank == 0 and coff >= cbank:
                        reads.setdefault(coff - cbank, []).append("%#x(op %03x)" % (i, op))
                if (op & 0xF00) == 0x800:
                    imm = (lo >> 32) & 0xFFFFFFFF
                    if 0x30000000 < imm < 0x48000000 or 0xB0000000 < imm < 0xC8000000:
                        floats.append("%#x:%g" % (i, struct.unpack("<f", struct.pack("<I", imm))[0]))
            for p in sorted(reads):
                print("    reads param+%-3d at %s" % (p, " ".join(reads[p][:8])))
            if floats:
                print("    float immediates:", " ".join(floats[:40]))


if __name__ == "__main__":
    main()
