// version.dll proxy: every export jumps to the real System32 version.dll.
// SPDX-License-Identifier: MIT
#include <windows.h>

extern "C" void* smm_real[17] = {};

#define SMM_FWD(n) \
    __asm__(".globl smm_fwd_" #n "\n" \
            "smm_fwd_" #n ":\n" \
            "  jmp *(smm_real + 8*" #n ")(%rip)\n");
SMM_FWD(0) SMM_FWD(1) SMM_FWD(2) SMM_FWD(3) SMM_FWD(4) SMM_FWD(5) SMM_FWD(6) SMM_FWD(7) SMM_FWD(8)
SMM_FWD(9) SMM_FWD(10) SMM_FWD(11) SMM_FWD(12) SMM_FWD(13) SMM_FWD(14) SMM_FWD(15) SMM_FWD(16)

static const char* const kNames[17] = {
    "GetFileVersionInfoA", "GetFileVersionInfoByHandle", "GetFileVersionInfoExA", "GetFileVersionInfoExW",
    "GetFileVersionInfoSizeA", "GetFileVersionInfoSizeExA", "GetFileVersionInfoSizeExW",
    "GetFileVersionInfoSizeW", "GetFileVersionInfoW", "VerFindFileA", "VerFindFileW", "VerInstallFileA",
    "VerInstallFileW", "VerLanguageNameA", "VerLanguageNameW", "VerQueryValueA", "VerQueryValueW"};

bool smm_load_real_version() {
    wchar_t path[MAX_PATH + 16];
    const UINT n = GetSystemDirectoryW(path, MAX_PATH);
    if (!n || n >= MAX_PATH) return false;
    lstrcatW(path, L"\\version.dll");
    const HMODULE real = LoadLibraryW(path);
    if (!real) return false;
    for (int i = 0; i < 17; ++i) {
        smm_real[i] = reinterpret_cast<void*>(GetProcAddress(real, kNames[i]));
        if (!smm_real[i]) return false;
    }
    return true;
}
