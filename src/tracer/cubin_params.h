// Kernel parameter layout from a CUDA fatbin's cubin. Shared by the tracer and its test.
// SPDX-License-Identifier: MIT
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Provided by the includer: true when [address, address + bytes) can be read.
bool readable(const void* address, size_t bytes);

template <class T>
T rd(const uint8_t* p) {
    T v;
    memcpy(&v, p, sizeof(v));
    return v;
}

// ------------------------------------------------- kernel parameter sizes

struct ParamLayout {
    uint32_t count = 0;
    uint32_t total = 0;       // CBANK_PARAM_SIZE
    uint16_t offset[8] = {};  // per kernel parameter
    uint16_t size[8] = {};
};

// Reads EIATTR_KPARAM_INFO / CBANK_PARAM_SIZE for `kernel` from the cubin inside a fatbin.
bool param_layout(const uint8_t* fat, const char* kernel, ParamLayout& out) {
    if (!readable(fat, 16) || rd<uint32_t>(fat) != 0xBA55ED50u) return false;
    const size_t header = rd<uint16_t>(fat + 6);
    const uint64_t payload = rd<uint64_t>(fat + 8);
    if (header < 16 || header > 0x100 || payload > (64u << 20)) return false;
    const size_t total = header + size_t(payload);
    if (!readable(fat, total)) return false;

    const uint8_t* elf = nullptr;
    size_t elf_size = 0;
    for (size_t p = header; p + 0x40 <= total;) {
        const uint16_t kind = rd<uint16_t>(fat + p);
        const uint32_t eh = rd<uint32_t>(fat + p + 4);
        const uint64_t size = rd<uint64_t>(fat + p + 8);
        const uint32_t arch = rd<uint32_t>(fat + p + 28);
        if (eh < 0x20 || eh > 0x400 || size > total - p - eh) break;
        if (kind == 2 && (!elf || arch == 89)) {
            elf = fat + p + eh;
            elf_size = size_t(size);
        }
        p = (p + eh + size_t(size) + 7) & ~size_t(7);
    }
    if (!elf || elf_size < 0x40 || memcmp(elf, "\x7f" "ELF", 4) != 0) return false;

    const uint64_t shoff = rd<uint64_t>(elf + 0x28);
    const uint16_t shentsize = rd<uint16_t>(elf + 0x3a), shnum = rd<uint16_t>(elf + 0x3c),
                   shstrndx = rd<uint16_t>(elf + 0x3e);
    if (shentsize < 0x40 || shstrndx >= shnum || shoff > elf_size ||
        uint64_t(shnum) * shentsize > elf_size - shoff)
        return false;
    const uint64_t names = rd<uint64_t>(elf + shoff + uint64_t(shstrndx) * shentsize + 0x18);

    char wanted[96];
    snprintf(wanted, sizeof(wanted), ".nv.info.%s", kernel);
    const size_t wanted_len = strlen(wanted) + 1;
    for (uint16_t i = 0; i < shnum; ++i) {
        const uint8_t* sh = elf + shoff + uint64_t(i) * shentsize;
        const uint64_t name = names + rd<uint32_t>(sh);
        if (name > elf_size || wanted_len > elf_size - name || memcmp(elf + name, wanted, wanted_len) != 0)
            continue;
        const uint64_t off = rd<uint64_t>(sh + 0x18), size = rd<uint64_t>(sh + 0x20);
        if (off > elf_size || size > elf_size - off) return false;
        for (uint64_t p = off; p + 4 <= off + size;) {
            const uint8_t fmt = elf[p], attr = elf[p + 1];
            if (fmt == 4) {
                const uint16_t len = rd<uint16_t>(elf + p + 2);
                if (p + 4 + len > off + size) break;
                if (attr == 0x17 && len >= 12 && out.count < 8) {
                    out.offset[out.count] = rd<uint16_t>(elf + p + 4 + 6);
                    out.size[out.count] = uint16_t((rd<uint32_t>(elf + p + 4 + 8) >> 18) & 0x3fff);
                    ++out.count;
                }
                p += 4 + len;
            } else {
                if (attr == 0x19) out.total = rd<uint16_t>(elf + p + 2);
                p += 4;
            }
        }
        return out.count != 0;
    }
    return false;
}
