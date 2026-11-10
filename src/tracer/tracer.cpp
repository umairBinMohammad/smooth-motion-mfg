// Smooth Motion tracer and phase 2 prototype.
// SPDX-License-Identifier: MIT
//
// Records what NvPresent64.dll asks CUDA to do: which kernel modules it loads,
// which kernels it launches per frame, and the parameter block of each launch.
// Every hooked call is forwarded with its original arguments. From 0.3 it also
// inserts one kernel of its own (see "time shift" below); at t = 0.5, the
// default, that kernel leaves the data unchanged.
//
// Output: smm_trace.log beside the game executable (or in %SMM_LOG_DIR%).

#include <windows.h>

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "MinHook.h"

bool smm_load_real_version();

// ------------------------------------------------------------ safe reads

bool readable(const void* address, size_t bytes) {
    const auto* p = static_cast<const uint8_t*>(address);
    const uint8_t* end = p + bytes;
    if (!address || end < p) return false;
    while (p < end) {
        MEMORY_BASIC_INFORMATION info;
        if (!VirtualQuery(p, &info, sizeof(info)) || info.State != MEM_COMMIT ||
            (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)))
            return false;
        p = static_cast<const uint8_t*>(info.BaseAddress) + info.RegionSize;
    }
    return true;
}

#include "cubin_params.h"

namespace {

// One test key for every build: F10. kTestAction selects what it does in this build.
enum class TestAction { Timeline, CaptureBuffers, CycleTime };
constexpr TestAction kTestAction = TestAction::Timeline;
constexpr int kTestKey = VK_F10;

#define SMM_VERSION "prototype 0.4.1"

// ---------------------------------------------------------------- logging

CRITICAL_SECTION g_lock;
HANDLE g_log = INVALID_HANDLE_VALUE;
LARGE_INTEGER g_qpf{}, g_qpc0{};

double now_ms() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return 1000.0 * double(t.QuadPart - g_qpc0.QuadPart) / double(g_qpf.QuadPart);
}

void logf(const char* format, ...) {
    if (g_log == INVALID_HANDLE_VALUE) return;
    char line[2048];
    int n = snprintf(line, sizeof(line), "%10.3f ", now_ms());
    va_list args;
    va_start(args, format);
    const int m = vsnprintf(line + n, sizeof(line) - n - 2, format, args);
    va_end(args);
    if (m < 0) return;
    n += m < int(sizeof(line)) - n - 2 ? m : int(sizeof(line)) - n - 3;
    line[n++] = '\r';
    line[n++] = '\n';
    DWORD written;
    WriteFile(g_log, line, DWORD(n), &written, nullptr);
}

wchar_t g_dir_early[MAX_PATH * 2];

void open_log() {
    wchar_t path[MAX_PATH * 2];
    DWORD n = GetEnvironmentVariableW(L"SMM_LOG_DIR", path, MAX_PATH);
    if (!n || n >= MAX_PATH) {
        n = GetModuleFileNameW(nullptr, path, MAX_PATH);
        while (n && path[n - 1] != L'\\') --n;
        path[n] = 0;
    } else if (path[n - 1] != L'\\') {
        path[n++] = L'\\';
        path[n] = 0;
    }
    lstrcpyW(g_dir_early, path);
    lstrcatW(path, L"smm_trace.log");
    g_log = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL, nullptr);
}

// ------------------------------------------------------------ CUDA state

using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUgraph = void*;

struct Module {
    CUmodule handle;
    const uint8_t* image;
    int index;
};
struct Function {
    CUfunction handle;
    char name[48];
    ParamLayout layout;
    uint64_t launches;        // total
    uint32_t window;          // since last summary
};

Module g_modules[256];
int g_module_count = 0;
Function g_functions[256];
int g_function_count = 0;

uint8_t* g_nvp = nullptr;
size_t g_nvp_size = 0;
uint64_t g_foreign_calls = 0;

uint64_t g_launch_seq = 0;
uint32_t g_window_graph_launches = 0, g_window_unknown = 0;
int g_capturing = 0;
double g_last_summary = 0;

constexpr uint64_t kDetailFirst = 800;     // launches logged in full at the start
constexpr uint64_t kDetailEvery = 6000;    // then a burst of kDetailBurst this often
constexpr uint64_t kDetailBurst = 80;
constexpr double kSummaryMs = 5000.0;

// warp_coarse_kernel: the 4 bytes at +48 (two half floats) are the suspected displacement scale.
uint32_t g_warp_values[32];
int g_warp_value_count = 0;

bool from_nvpresent(void* return_address) {
    if (!g_nvp) {
        const HMODULE m = GetModuleHandleW(L"NvPresent64.dll");
        if (!m) return false;
        auto* base = reinterpret_cast<uint8_t*>(m);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(
            base + reinterpret_cast<const IMAGE_DOS_HEADER*>(base)->e_lfanew);
        g_nvp_size = nt->OptionalHeader.SizeOfImage;
        g_nvp = base;
        wchar_t path[MAX_PATH];
        GetModuleFileNameW(m, path, MAX_PATH);
        logf("NvPresent64.dll at %p size 0x%zx path %ls", base, g_nvp_size, path);
    }
    const auto* ra = static_cast<const uint8_t*>(return_address);
    return ra >= g_nvp && ra < g_nvp + g_nvp_size;
}

void note_foreign(const char* api, void* return_address) {
    if (++g_foreign_calls > 8) return;
    HMODULE m = nullptr;
    wchar_t path[MAX_PATH] = L"?";
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(return_address), &m))
        GetModuleFileNameW(m, path, MAX_PATH);
    logf("other-caller %s from %ls (not traced)", api, path);
}

bool from_nvpresent_quick(void* return_address) {
    EnterCriticalSection(&g_lock);
    const bool result = from_nvpresent(return_address);
    LeaveCriticalSection(&g_lock);
    return result;
}

Function* find_function(CUfunction handle) {
    for (int i = 0; i < g_function_count; ++i)
        if (g_functions[i].handle == handle) return &g_functions[i];
    return nullptr;
}

void hex(char* out, size_t out_size, const void* data, size_t bytes) {
    static const char digits[] = "0123456789abcdef";
    const auto* p = static_cast<const uint8_t*>(data);
    size_t o = 0;
    for (size_t i = 0; i < bytes && o + 10 < out_size; ++i) {
        if (i && i % 4 == 0) out[o++] = ' ';
        out[o++] = digits[p[i] >> 4];
        out[o++] = digits[p[i] & 15];
    }
    out[o] = 0;
}

void summary_if_due() {
    const double t = now_ms();
    if (t - g_last_summary < kSummaryMs) return;
    const double seconds = (t - g_last_summary) / 1000.0;
    g_last_summary = t;
    char line[1800];
    int n = 0;
    for (int i = 0; i < g_function_count && n < int(sizeof(line)) - 80; ++i) {
        if (!g_functions[i].window) continue;
        n += snprintf(line + n, sizeof(line) - n, " %s=%.1f", g_functions[i].name,
                      g_functions[i].window / seconds);
        g_functions[i].window = 0;
    }
    line[n] = 0;
    logf("summary launches/s:%s | graph=%.1f unknown=%.1f total=%llu", n ? line : " none",
         g_window_graph_launches / seconds, g_window_unknown / seconds,
         static_cast<unsigned long long>(g_launch_seq));
    g_window_graph_launches = g_window_unknown = 0;
}

// ---------------------------------------------------------------- capture
//
// F9 dumps two consecutive generated frames' worth of network data to
// smm_capture\ beside the log: the network's final output (what
// warp_coarse_kernel reads) and the 256x144 inputs the downscale kernel wrote.
// Read-only: one cuStreamSynchronize and two device-to-host copies per dump.

int (*cu_MemGetAddressRange)(uint64_t*, size_t*, uint64_t);
int (*cu_MemcpyDtoH)(void*, uint64_t, size_t);
int (*cu_StreamSynchronize)(CUstream);

wchar_t g_dir[MAX_PATH * 2];
uint64_t g_ptr_output = 0, g_ptr_coarse = 0;  // from the most recent graph capture
uint64_t g_graph_frames = 0;
int g_dump_pending = 0, g_dump_sets = 0;
bool g_key_down = false;
constexpr size_t kDumpBytes = 2u << 20;
constexpr int kMaxDumpSets = 8;

void dump_buffer(const char* tag, uint64_t pointer, int set, int part) {
    if (!pointer || !cu_MemGetAddressRange || !cu_MemcpyDtoH) return;
    uint64_t base = 0;
    size_t size = 0;
    const int range_rc = cu_MemGetAddressRange(&base, &size, pointer);
    size_t bytes = kDumpBytes;
    if (range_rc == 0 && base + size > pointer && base + size - pointer < bytes) bytes = size_t(base + size - pointer);
    void* host = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!host) return;
    const int rc = cu_MemcpyDtoH(host, pointer, bytes);
    wchar_t path[MAX_PATH * 2 + 64];
    swprintf(path, MAX_PATH * 2 + 64, L"%lssmm_capture\\set%d_%d_%hs.bin", g_dir, set, part, tag);
    DWORD written = 0;
    if (rc == 0) {
        const HANDLE file = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            WriteFile(file, host, DWORD(bytes), &written, nullptr);
            CloseHandle(file);
        }
    }
    VirtualFree(host, 0, MEM_RELEASE);
    logf("dump set%d_%d %s ptr=0x%llx bytes=%zu alloc_base=0x%llx alloc_size=%zu range_rc=%d copy_rc=%d written=%lu",
         set, part, tag, static_cast<unsigned long long>(pointer), bytes, static_cast<unsigned long long>(base), size,
         range_rc, rc, written);
}

// ------------------------------------------------------------------ hooks

int (*real_cuModuleLoadData)(CUmodule*, const void*);
int (*real_cuModuleUnload)(CUmodule);
int (*real_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
int (*real_cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                           unsigned, CUstream, void**, void**);
int (*real_cuGraphLaunch)(CUgraph, CUstream);
int (*real_cuStreamBeginCapture_v2)(CUstream, int);
int (*real_cuStreamEndCapture)(CUstream, CUgraph*);
int (*real_cuArrayCreate_v2)(void**, const void*);
int (*real_cuExternalMemoryGetMappedMipmappedArray)(void**, void*, const void*);

// ------------------------------------------------------------ time shift
//
// Phase 2 experiment. While NvPresent records its per-frame graph, one extra
// kernel is launched just before warp_coarse_kernel. It multiplies the flow
// channels of the network output in place: flows into the previous frame
// (channels 0,1,4,5) by 2t and flows into the current frame (2,3,6,7) by
// 2(1-t). NVIDIA's own warp then builds the frame for time t instead of the
// midpoint. At t = 0.5 both factors are 1 and the data is unchanged.
// The factors live in device memory, so F10 can change t without re-recording.

const char kScalePtx[] =
    ".version 7.0\n"
    ".target sm_80\n"
    ".address_size 64\n"
    "\n"
    ".visible .entry smm_scale_flow(\n"
    "    .param .u64 smm_buf,\n"
    "    .param .u64 smm_scales,\n"
    "    .param .u32 smm_count\n"
    ")\n"
    "{\n"
    "    .reg .pred %p<2>;\n"
    "    .reg .b16 %h<2>;\n"
    "    .reg .f32 %f<5>;\n"
    "    .reg .b32 %r<6>;\n"
    "    .reg .b64 %rd<6>;\n"
    "    ld.param.u64 %rd1, [smm_buf];\n"
    "    ld.param.u64 %rd5, [smm_scales];\n"
    "    ld.param.u32 %r1, [smm_count];\n"
    "    mov.u32 %r2, %ctaid.x;\n"
    "    mov.u32 %r3, %ntid.x;\n"
    "    mov.u32 %r4, %tid.x;\n"
    "    mad.lo.u32 %r5, %r2, %r3, %r4;\n"
    "    setp.ge.u32 %p1, %r5, %r1;\n"
    "    @%p1 bra L_end;\n"
    "    ld.global.f32 %f1, [%rd5];\n"
    "    ld.global.f32 %f2, [%rd5+4];\n"
    "    cvt.u64.u32 %rd2, %r5;\n"
    "    shl.b64 %rd3, %rd2, 5;\n"
    "    add.u64 %rd4, %rd1, %rd3;\n"
    "ld.global.u16 %h0, [%rd4+0];\n"
    "cvt.f32.f16 %f3, %h0;\n"
    "mul.f32 %f4, %f3, %f1;\n"
    "cvt.rn.f16.f32 %h1, %f4;\n"
    "st.global.u16 [%rd4+0], %h1;\n"
    "ld.global.u16 %h0, [%rd4+2];\n"
    "cvt.f32.f16 %f3, %h0;\n"
    "mul.f32 %f4, %f3, %f1;\n"
    "cvt.rn.f16.f32 %h1, %f4;\n"
    "st.global.u16 [%rd4+2], %h1;\n"
    "ld.global.u16 %h0, [%rd4+4];\n"
    "cvt.f32.f16 %f3, %h0;\n"
    "mul.f32 %f4, %f3, %f2;\n"
    "cvt.rn.f16.f32 %h1, %f4;\n"
    "st.global.u16 [%rd4+4], %h1;\n"
    "ld.global.u16 %h0, [%rd4+6];\n"
    "cvt.f32.f16 %f3, %h0;\n"
    "mul.f32 %f4, %f3, %f2;\n"
    "cvt.rn.f16.f32 %h1, %f4;\n"
    "st.global.u16 [%rd4+6], %h1;\n"
    "ld.global.u16 %h0, [%rd4+8];\n"
    "cvt.f32.f16 %f3, %h0;\n"
    "mul.f32 %f4, %f3, %f1;\n"
    "cvt.rn.f16.f32 %h1, %f4;\n"
    "st.global.u16 [%rd4+8], %h1;\n"
    "ld.global.u16 %h0, [%rd4+10];\n"
    "cvt.f32.f16 %f3, %h0;\n"
    "mul.f32 %f4, %f3, %f1;\n"
    "cvt.rn.f16.f32 %h1, %f4;\n"
    "st.global.u16 [%rd4+10], %h1;\n"
    "ld.global.u16 %h0, [%rd4+12];\n"
    "cvt.f32.f16 %f3, %h0;\n"
    "mul.f32 %f4, %f3, %f2;\n"
    "cvt.rn.f16.f32 %h1, %f4;\n"
    "st.global.u16 [%rd4+12], %h1;\n"
    "ld.global.u16 %h0, [%rd4+14];\n"
    "cvt.f32.f16 %f3, %h0;\n"
    "mul.f32 %f4, %f3, %f2;\n"
    "cvt.rn.f16.f32 %h1, %f4;\n"
    "st.global.u16 [%rd4+14], %h1;\n"
    "L_end:\n"
    "    ret;\n"
    "}\n";

int (*cu_ModuleLoadDataEx)(CUmodule*, const void*, unsigned, int*, void**);
int (*cu_MemAlloc)(uint64_t*, size_t);
int (*cu_MemcpyHtoD)(uint64_t, const void*, size_t);

constexpr float kTimes[] = {0.5f, 0.25f, 0.75f, 0.0f};
int g_time_index = 0;
bool g_shift_failed = false, g_time_key_down = false;
CUfunction g_scale_function = nullptr;
uint64_t g_scale_memory = 0;

int upload_time() {
    const float t = kTimes[g_time_index];
    const float scales[2] = {2.0f * t, 2.0f * (1.0f - t)};
    return cu_MemcpyHtoD(g_scale_memory, scales, sizeof(scales));
}

int (*cu_CtxGetCurrent)(void**);
int (*cu_CtxPushCurrent)(void*);
int (*cu_CtxPopCurrent)(void**);

struct SetupJob {
    void* context;
    int rc;
    const char* step;
    char errors[1024];
};

void step_log(const char* text) {
    EnterCriticalSection(&g_lock);
    logf("time shift: %s", text);
    LeaveCriticalSection(&g_lock);
}

// Runs on its own thread with a large stack: the driver's PTX compiler is not
// safe to run on NvPresent's worker thread (0.3 died inside this call there).
DWORD WINAPI setup_thread(void* argument) {
    auto* job = static_cast<SetupJob*>(argument);
    job->step = "push context";
    step_log("setup thread started");
    job->rc = cu_CtxPushCurrent(job->context);
    if (job->rc != 0) return 0;
    int options[2] = {5, 6};  // CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
    void* values[2] = {job->errors, reinterpret_cast<void*>(sizeof(job->errors))};
    CUmodule module = nullptr;
    job->step = "compile PTX";
    step_log("compiling PTX");
    job->rc = cu_ModuleLoadDataEx(&module, kScalePtx, 2, options, values);
    step_log(job->rc == 0 ? "compile ok" : "compile FAILED");
    if (job->rc == 0) {
        job->step = "get function";
        CUfunction function = nullptr;
        job->rc = real_cuModuleGetFunction(&function, module, "smm_scale_flow");
        step_log("get function returned");
        if (job->rc == 0) {
            job->step = "allocate";
            job->rc = cu_MemAlloc(&g_scale_memory, 8);
            step_log("allocate returned");
        }
        if (job->rc == 0) {
            job->step = "upload";
            job->rc = upload_time();
            step_log("upload returned");
        }
        if (job->rc == 0) {
            job->step = "done";
            g_scale_function = function;
        }
    }
    void* popped = nullptr;
    cu_CtxPopCurrent(&popped);
    step_log("context popped");
    return 0;
}

// Compiles the scale kernel. Called on NvPresent's thread when it loads its first kernel
// module: the CUDA context is current and no graph capture is in progress. (NvPresent captures
// in global mode, where module loads and allocations are not allowed until the capture ends;
// 0.3 and 0.3.1 did this work mid-capture and the game died.)
void setup_time_shift() {
    static bool attempted = false;
    if (attempted) return;
    attempted = true;
    if (!cu_ModuleLoadDataEx || !cu_MemAlloc || !cu_MemcpyHtoD || !cu_CtxGetCurrent || !cu_CtxPushCurrent ||
        !cu_CtxPopCurrent) {
        g_shift_failed = true;
        return;
    }
    static SetupJob job;
    job = {};
    job.rc = cu_CtxGetCurrent(&job.context);
    job.step = "get context";
    if (job.rc == 0 && job.context) {
        job.rc = -1;
        job.step = "start thread";
        const HANDLE thread = CreateThread(nullptr, 16u << 20, setup_thread, &job, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
        if (thread) {
            WaitForSingleObject(thread, INFINITE);
            CloseHandle(thread);
        }
    }
    EnterCriticalSection(&g_lock);
    logf("time shift: kernel setup rc=%d at step '%s' capturing=%d %s", job.rc, job.step, g_capturing, job.errors);
    LeaveCriticalSection(&g_lock);
    if (job.rc != 0 || !g_scale_function) g_shift_failed = true;
}

// Called on NvPresent's thread right before it launches warp_coarse_kernel.
void inject_time_shift(const uint8_t* warp_params, CUstream stream) {
    if (g_shift_failed || !g_scale_function) return;
    uint64_t buffer = rd<uint64_t>(warp_params + 16);
    const uint32_t width = rd<uint32_t>(warp_params + 32), height = rd<uint32_t>(warp_params + 36);
    uint32_t count = width * height;
    if (!buffer || !width || !height || width > 1024 || height > 1024) {
        EnterCriticalSection(&g_lock);
        logf("time shift: unexpected coarse size %ux%u; not injected", width, height);
        LeaveCriticalSection(&g_lock);
        return;
    }
    void* params[3] = {&buffer, &g_scale_memory, &count};
    const int rc = real_cuLaunchKernel(g_scale_function, (count + 255) / 256, 1, 1, 256, 1, 1, 0, stream, params, nullptr);
    EnterCriticalSection(&g_lock);
    logf("time shift: scale kernel inserted before warp (buffer=0x%llx pixels=%u t=%.2f) rc=%d%s",
         static_cast<unsigned long long>(buffer), count, kTimes[g_time_index], rc, g_capturing ? " [recorded into graph]" : "");
    LeaveCriticalSection(&g_lock);
}

// F10 cycles t. Called from the graph-launch hook, between frames.
void poll_time_key() {
    const bool key = kTestAction == TestAction::CycleTime && (GetAsyncKeyState(kTestKey) & 0x8000) != 0;
    if (key && !g_time_key_down && g_scale_function) {
        g_time_index = (g_time_index + 1) % int(sizeof(kTimes) / sizeof(kTimes[0]));
        const int rc = upload_time();
        logf("time shift: F10 -> t=%.2f (upload rc=%d)", kTimes[g_time_index], rc);
        Beep(500 + 250 * g_time_index, 70);
    }
    g_time_key_down = key;
}

int hook_cuModuleLoadData(CUmodule* module, const void* image) {
    const int rc = real_cuModuleLoadData(module, image);
    void* ra = __builtin_return_address(0);
    bool ours = false;
    EnterCriticalSection(&g_lock);
    if (!from_nvpresent(ra)) {
        note_foreign("cuModuleLoadData", ra);
    } else {
        ours = !g_capturing;
        const auto* p = static_cast<const uint8_t*>(image);
        const bool inside = p >= g_nvp && p < g_nvp + g_nvp_size;
        uint64_t size = 0;
        if (readable(p, 16) && rd<uint32_t>(p) == 0xBA55ED50u) size = 16 + rd<uint64_t>(p + 8);
        const int index = g_module_count;
        if (rc == 0 && module && g_module_count < 256) g_modules[g_module_count++] = {*module, p, index};
        if (inside)
            logf("module #%d load rva=+0x%zx bytes=%llu rc=%d handle=%p", index, size_t(p - g_nvp),
                 static_cast<unsigned long long>(size), rc, module ? *module : nullptr);
        else
            logf("module #%d load image=%p (outside NvPresent) bytes=%llu rc=%d handle=%p", index, image,
                 static_cast<unsigned long long>(size), rc, module ? *module : nullptr);
    }
    LeaveCriticalSection(&g_lock);
    if (ours) setup_time_shift();
    return rc;
}

int hook_cuModuleUnload(CUmodule module) {
    void* ra = __builtin_return_address(0);
    EnterCriticalSection(&g_lock);
    if (from_nvpresent(ra)) {
        for (int i = 0; i < g_module_count; ++i)
            if (g_modules[i].handle == module) {
                logf("module #%d unload", g_modules[i].index);
                g_modules[i].handle = nullptr;
            }
    }
    LeaveCriticalSection(&g_lock);
    return real_cuModuleUnload(module);
}

int hook_cuModuleGetFunction(CUfunction* function, CUmodule module, const char* name) {
    const int rc = real_cuModuleGetFunction(function, module, name);
    void* ra = __builtin_return_address(0);
    EnterCriticalSection(&g_lock);
    if (!from_nvpresent(ra)) {
        note_foreign("cuModuleGetFunction", ra);
    } else if (rc == 0 && function && name) {
        Function* f = find_function(*function);
        if (!f && g_function_count < 256) f = &g_functions[g_function_count++];
        int module_index = -1;
        if (f) {
            *f = {};
            f->handle = *function;
            snprintf(f->name, sizeof(f->name), "%s", name);
            for (int i = 0; i < g_module_count; ++i)
                if (g_modules[i].handle == module) {
                    module_index = g_modules[i].index;
                    param_layout(g_modules[i].image, name, f->layout);
                }
            char sizes[128];
            int n = 0;
            sizes[0] = 0;
            for (uint32_t i = 0; i < f->layout.count; ++i)
                n += snprintf(sizes + n, sizeof(sizes) - n, " #%u@%u:%u", i, f->layout.offset[i], f->layout.size[i]);
            logf("function %s module #%d handle=%p params=%u total=%u%s", name, module_index, *function,
                 f->layout.count, f->layout.total, sizes);
        }
    } else {
        logf("function %s lookup rc=%d", name ? name : "?", rc);
    }
    LeaveCriticalSection(&g_lock);
    return rc;
}

int hook_cuLaunchKernel(CUfunction function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                        unsigned bz, unsigned shared, CUstream stream, void** params, void** extra) {
    void* ra = __builtin_return_address(0);
    bool is_warp = false;
    uint8_t warp_block[136];
    EnterCriticalSection(&g_lock);
    if (!from_nvpresent(ra)) {
        note_foreign("cuLaunchKernel", ra);
    } else {
        const uint64_t seq = g_launch_seq++;
        Function* f = find_function(function);
        if (f) {
            ++f->launches;
            ++f->window;
        } else {
            ++g_window_unknown;
        }
        is_warp = f && strcmp(f->name, "warp_coarse_kernel") == 0;
        bool detail = seq < kDetailFirst || (seq % kDetailEvery) < kDetailBurst;

        // First kernel parameter, when its size is known and the memory is readable.
        uint8_t block[256];
        size_t block_size = 0;
        if (f && f->layout.count && params && readable(params, sizeof(void*)) && params[0]) {
            block_size = f->layout.size[0] < sizeof(block) ? f->layout.size[0] : sizeof(block);
            if (readable(params[0], block_size)) memcpy(block, params[0], block_size);
            else block_size = 0;
        }
        if (f && block_size >= 24 && strcmp(f->name, "warp_coarse_kernel") == 0) g_ptr_output = rd<uint64_t>(block + 16);
        if (f && block_size >= 112 && strcmp(f->name, "downscale_kernel") == 0) g_ptr_coarse = rd<uint64_t>(block + 104);
        if (f && block_size >= 52 && strcmp(f->name, "warp_coarse_kernel") == 0) {
            const uint32_t value = rd<uint32_t>(block + 48);
            bool seen = false;
            for (int i = 0; i < g_warp_value_count; ++i) seen |= g_warp_values[i] == value;
            if (!seen && g_warp_value_count < 32) {
                g_warp_values[g_warp_value_count++] = value;
                logf("warp +48 new value 0x%08x (distinct so far: %d)", value, g_warp_value_count);
                detail = true;
            }
        }
        if (detail) {
            char text[700];
            text[0] = 0;
            if (block_size) hex(text, sizeof(text), block, block_size);
            logf("launch %llu %s tid=%lu grid=%ux%ux%u block=%ux%ux%u shmem=%u stream=%p%s%s params[%zu]=%s",
                 static_cast<unsigned long long>(seq), f ? f->name : "(unknown)", GetCurrentThreadId(), gx, gy,
                 gz, bx, by, bz, shared, stream, g_capturing ? " CAPTURING" : "", extra ? " extra" : "",
                 block_size, text);
            if (f && f->layout.count > 1 && params && readable(params, f->layout.count * sizeof(void*))) {
                for (uint32_t i = 1; i < f->layout.count; ++i) {
                    const size_t size = f->layout.size[i] < 64 ? f->layout.size[i] : 64;
                    if (params[i] && readable(params[i], size)) {
                        hex(text, sizeof(text), params[i], size);
                        logf("    param #%u [%zu]=%s", i, size, text);
                    }
                }
            }
        }
        if (is_warp && block_size >= sizeof(warp_block)) memcpy(warp_block, block, sizeof(warp_block));
        else is_warp = false;
        summary_if_due();
    }
    LeaveCriticalSection(&g_lock);
    if (is_warp) inject_time_shift(warp_block, stream);
    return real_cuLaunchKernel(function, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

// ------------------------------------------------------------- timeline
//
// How the generated frame reaches the display. F8 (and, once, the 300th frame
// of a graph) records the next 400 events with timestamps: graph launches,
// CUDA/D3D12 hand-over semaphores, copies, and the driver's flip-metering calls.

int g_timeline = 0;
bool g_timeline_auto_done = false, g_timeline_key_down = false;

void tl(const char* format, ...) {
    if (g_timeline <= 0) return;
    char text[400];
    va_list args;
    va_start(args, format);
    vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    --g_timeline;
    logf("TL tid=%lu %s%s", GetCurrentThreadId(), text, g_timeline == 0 ? "  [timeline end]" : "");
}

int (*real_cuSignalExternalSemaphoresAsync)(void* const*, const void*, unsigned, CUstream);
int (*real_cuWaitExternalSemaphoresAsync)(void* const*, const void*, unsigned, CUstream);
int (*real_cuMemcpy2DAsync_v2)(const void*, CUstream);
int (*real_cuStreamSynchronize)(CUstream);
int (*real_cuImportExternalMemory)(void**, const void*);
int (*real_cuImportExternalSemaphore)(void**, const void*);
void* (*real_nvapi_QueryInterface)(uint32_t);
int (*real_SetFlipConfig)(void*, void*, void*, void*);

int hook_cuSignalExternalSemaphoresAsync(void* const* semaphores, const void* params, unsigned count, CUstream stream) {
    if (from_nvpresent_quick(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        tl("cuda signal sem=%p value=%llu n=%u stream=%p", count && readable(semaphores, 8) ? semaphores[0] : nullptr,
           readable(params, 8) ? static_cast<unsigned long long>(rd<uint64_t>(static_cast<const uint8_t*>(params))) : 0ull,
           count, stream);
        LeaveCriticalSection(&g_lock);
    }
    return real_cuSignalExternalSemaphoresAsync(semaphores, params, count, stream);
}

int hook_cuWaitExternalSemaphoresAsync(void* const* semaphores, const void* params, unsigned count, CUstream stream) {
    if (from_nvpresent_quick(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        tl("cuda wait   sem=%p value=%llu n=%u stream=%p", count && readable(semaphores, 8) ? semaphores[0] : nullptr,
           readable(params, 8) ? static_cast<unsigned long long>(rd<uint64_t>(static_cast<const uint8_t*>(params))) : 0ull,
           count, stream);
        LeaveCriticalSection(&g_lock);
    }
    return real_cuWaitExternalSemaphoresAsync(semaphores, params, count, stream);
}

int hook_cuMemcpy2DAsync_v2(const void* copy, CUstream stream) {
    if (from_nvpresent_quick(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        char text[300] = "";
        if (readable(copy, 128)) hex(text, sizeof(text), copy, 128);
        tl("cuda memcpy2D stream=%p %s", stream, text);
        LeaveCriticalSection(&g_lock);
    }
    return real_cuMemcpy2DAsync_v2(copy, stream);
}

int hook_cuStreamSynchronize(CUstream stream) {
    const bool ours = from_nvpresent_quick(__builtin_return_address(0));
    const double t0 = now_ms();
    const int rc = real_cuStreamSynchronize(stream);
    if (ours) {
        EnterCriticalSection(&g_lock);
        tl("cuda stream sync stream=%p took %.3f ms", stream, now_ms() - t0);
        LeaveCriticalSection(&g_lock);
    }
    return rc;
}

int hook_cuImportExternalMemory(void** memory, const void* descriptor) {
    const int rc = real_cuImportExternalMemory(memory, descriptor);
    if (from_nvpresent_quick(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        char text[200] = "";
        if (readable(descriptor, 40)) hex(text, sizeof(text), descriptor, 40);
        logf("external memory import rc=%d handle=%p desc=%s", rc, memory ? *memory : nullptr, text);
        LeaveCriticalSection(&g_lock);
    }
    return rc;
}

int hook_cuImportExternalSemaphore(void** semaphore, const void* descriptor) {
    const int rc = real_cuImportExternalSemaphore(semaphore, descriptor);
    if (from_nvpresent_quick(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        char text[200] = "";
        if (readable(descriptor, 24)) hex(text, sizeof(text), descriptor, 24);
        logf("external semaphore import rc=%d handle=%p desc=%s", rc, semaphore ? *semaphore : nullptr, text);
        LeaveCriticalSection(&g_lock);
    }
    return rc;
}

// NvAPI_D3D12_SetFlipConfig as NvPresent calls it. Forwarded unchanged; parameters logged
// before and after (the driver writes feedback into the block).
int observed_SetFlipConfig(void* a, void* b, void* params, void* d) {
    static uint64_t calls = 0;
    static uint8_t last[24];
    uint8_t before[24] = {};
    const bool ok = readable(params, 24);
    if (ok) memcpy(before, params, 24);
    const double t0 = now_ms();
    const int rc = real_SetFlipConfig(a, b, params, d);
    EnterCriticalSection(&g_lock);
    const bool changed = ok && memcmp(before, last, 24) != 0;
    if (ok) memcpy(last, before, 24);
    char in[80] = "", out[80] = "";
    if (ok) {
        hex(in, sizeof(in), before, 24);
        hex(out, sizeof(out), params, 24);
    }
    if (calls++ < 20 || changed)
        logf("SetFlipConfig #%llu a=%p b=%p d=%p rc=%d in=%s out=%s (%.3f ms)", static_cast<unsigned long long>(calls), a, b,
             d, rc, in, out, now_ms() - t0);
    tl("SetFlipConfig rc=%d in=%s out=%s", rc, in, out);
    LeaveCriticalSection(&g_lock);
    return rc;
}

void* hook_nvapi_QueryInterface(uint32_t id) {
    void* result = real_nvapi_QueryInterface(id);
    if (!from_nvpresent_quick(__builtin_return_address(0))) return result;
    EnterCriticalSection(&g_lock);
    static uint32_t seen[96];
    static int seen_count = 0;
    bool known = false;
    for (int i = 0; i < seen_count; ++i) known |= seen[i] == id;
    if (!known && seen_count < 96) {
        seen[seen_count++] = id;
        logf("nvapi interface 0x%08X -> %p%s", id, result,
             id == 0xF3148C42u ? " (SetFlipConfig)" : id == 0x6194B19Du ? " (flip-config caps)" : "");
    }
    LeaveCriticalSection(&g_lock);
    if (id == 0xF3148C42u && result) {
        real_SetFlipConfig = reinterpret_cast<decltype(real_SetFlipConfig)>(result);
        return reinterpret_cast<void*>(&observed_SetFlipConfig);
    }
    return result;
}

int hook_cuGraphLaunch(CUgraph graph, CUstream stream) {
    void* ra = __builtin_return_address(0);
    if (!from_nvpresent_quick(ra)) return real_cuGraphLaunch(graph, stream);
    const double t0 = now_ms();
    const int rc = real_cuGraphLaunch(graph, stream);
    EnterCriticalSection(&g_lock);
    const uint64_t frame = g_graph_frames++;
    if (frame < 40) logf("graph launch graph=%p stream=%p tid=%lu rc=%d", graph, stream, GetCurrentThreadId(), rc);
    ++g_window_graph_launches;

    const bool tl_key = kTestAction == TestAction::Timeline && (GetAsyncKeyState(kTestKey) & 0x8000) != 0;
    if ((tl_key && !g_timeline_key_down) || (frame == 300 && !g_timeline_auto_done)) {
        g_timeline_auto_done = true;
        g_timeline = 400;
        logf("timeline: recording the next 400 events (graph frame %llu)", static_cast<unsigned long long>(frame));
        if (tl_key) Beep(900, 60);
    }
    g_timeline_key_down = tl_key;
    tl("graph launch returned after %.3f ms (frame %llu)", now_ms() - t0, static_cast<unsigned long long>(frame));
    const bool key = kTestAction == TestAction::CaptureBuffers && (GetAsyncKeyState(kTestKey) & 0x8000) != 0;
    if (key && !g_key_down && !g_dump_pending && g_dump_sets < kMaxDumpSets) {
        g_dump_pending = 2;
        ++g_dump_sets;
    }
    g_key_down = key;
    poll_time_key();
    if (g_dump_pending && cu_StreamSynchronize) {
        const int sync_rc = (real_cuStreamSynchronize ? real_cuStreamSynchronize : cu_StreamSynchronize)(stream);
        const double t1 = now_ms();
        const int part = 2 - g_dump_pending;
        logf("capture set%d_%d graph frame %llu: launch+sync %.3f ms (sync_rc=%d)", g_dump_sets, part,
             static_cast<unsigned long long>(frame), t1 - t0, sync_rc);
        dump_buffer("output", g_ptr_output, g_dump_sets, part);
        dump_buffer("coarse", g_ptr_coarse, g_dump_sets, part);
        --g_dump_pending;
    }
    summary_if_due();
    LeaveCriticalSection(&g_lock);
    return rc;
}

int hook_cuStreamBeginCapture_v2(CUstream stream, int mode) {
    void* ra = __builtin_return_address(0);
    EnterCriticalSection(&g_lock);
    if (from_nvpresent(ra)) {
        ++g_capturing;
        logf("capture begin stream=%p mode=%d (launches until 'capture end' are recorded into a graph)", stream, mode);
    }
    LeaveCriticalSection(&g_lock);
    return real_cuStreamBeginCapture_v2(stream, mode);
}

int hook_cuStreamEndCapture(CUstream stream, CUgraph* graph) {
    const int rc = real_cuStreamEndCapture(stream, graph);
    void* ra = __builtin_return_address(0);
    EnterCriticalSection(&g_lock);
    if (from_nvpresent(ra)) {
        if (g_capturing) --g_capturing;
        g_graph_frames = 0;
        logf("capture end stream=%p graph=%p rc=%d", stream, graph ? *graph : nullptr, rc);
    }
    LeaveCriticalSection(&g_lock);
    return rc;
}

int hook_cuArrayCreate_v2(void** array, const void* descriptor) {
    const int rc = real_cuArrayCreate_v2(array, descriptor);
    void* ra = __builtin_return_address(0);
    EnterCriticalSection(&g_lock);
    if (from_nvpresent(ra) && readable(descriptor, 24)) {
        const auto* d = static_cast<const uint8_t*>(descriptor);
        logf("array create %llux%llu format=0x%x channels=%u rc=%d handle=%p",
             static_cast<unsigned long long>(rd<uint64_t>(d)), static_cast<unsigned long long>(rd<uint64_t>(d + 8)),
             rd<uint32_t>(d + 16), rd<uint32_t>(d + 20), rc, array ? *array : nullptr);
    }
    LeaveCriticalSection(&g_lock);
    return rc;
}

int hook_cuExternalMemoryGetMappedMipmappedArray(void** mipmap, void* memory, const void* descriptor) {
    const int rc = real_cuExternalMemoryGetMappedMipmappedArray(mipmap, memory, descriptor);
    void* ra = __builtin_return_address(0);
    EnterCriticalSection(&g_lock);
    if (from_nvpresent(ra) && readable(descriptor, 48)) {
        // { u64 offset; { size_t W, H, D; u32 format, channels, flags; } ; u32 levels; }
        const auto* d = static_cast<const uint8_t*>(descriptor);
        logf("external image %llux%llux%llu format=0x%x channels=%u flags=0x%x levels=%u rc=%d handle=%p",
             static_cast<unsigned long long>(rd<uint64_t>(d + 8)), static_cast<unsigned long long>(rd<uint64_t>(d + 16)),
             static_cast<unsigned long long>(rd<uint64_t>(d + 24)), rd<uint32_t>(d + 32), rd<uint32_t>(d + 36),
             rd<uint32_t>(d + 40), rd<uint32_t>(d + 44), rc, mipmap ? *mipmap : nullptr);
    }
    LeaveCriticalSection(&g_lock);
    return rc;
}

template <class T>
void hook(const char* name, T detour, T* original) {
    const MH_STATUS status = MH_CreateHookApi(L"nvcuda.dll", name, reinterpret_cast<void*>(detour),
                                              reinterpret_cast<void**>(original));
    logf("hook %-44s %s", name, MH_StatusToString(status));
}

// Logs the first few fatal exceptions (where, and in which module) without handling them.
LONG CALLBACK crash_logger(EXCEPTION_POINTERS* info) {
    static LONG count = 0;
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_STACK_OVERFLOW && code != EXCEPTION_ILLEGAL_INSTRUCTION &&
        code != EXCEPTION_INT_DIVIDE_BY_ZERO && code != 0xC0000409u)
        return EXCEPTION_CONTINUE_SEARCH;
    if (InterlockedIncrement(&count) > 4 || code == EXCEPTION_STACK_OVERFLOW) return EXCEPTION_CONTINUE_SEARCH;
    void* address = info->ExceptionRecord->ExceptionAddress;
    HMODULE module = nullptr;
    wchar_t path[MAX_PATH] = L"?";
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(address), &module))
        GetModuleFileNameW(module, path, MAX_PATH);
    const ULONG_PTR* p = info->ExceptionRecord->ExceptionInformation;
    logf("EXCEPTION 0x%08lx at %p (%ls +0x%llx) tid=%lu access=%llu target=0x%llx", code, address, path,
         static_cast<unsigned long long>(static_cast<uint8_t*>(address) - reinterpret_cast<uint8_t*>(module)),
         GetCurrentThreadId(), static_cast<unsigned long long>(p[0]), static_cast<unsigned long long>(p[1]));
    return EXCEPTION_CONTINUE_SEARCH;
}

DWORD WINAPI bootstrap(void*) {
    open_log();
    AddVectoredExceptionHandler(1, crash_logger);
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    logf("smooth-motion-mfg " SMM_VERSION " in %ls (test key: F10 = record a timeline)", exe);

    // NvPresent imports nvcuda.dll by name; loading it first lets us hook its exports
    // before NvPresent makes its first call.
    if (!LoadLibraryW(L"nvcuda.dll")) {
        logf("nvcuda.dll could not be loaded (error %lu); nothing will be traced", GetLastError());
        return 0;
    }
    if (MH_Initialize() != MH_OK) {
        logf("MinHook failed to initialise; nothing will be traced");
        return 0;
    }
    lstrcpyW(g_dir, g_dir_early);
    wchar_t capture_dir[MAX_PATH * 2 + 16];
    lstrcpyW(capture_dir, g_dir);
    lstrcatW(capture_dir, L"smm_capture");
    CreateDirectoryW(capture_dir, nullptr);
    const HMODULE cuda = GetModuleHandleW(L"nvcuda.dll");
    cu_MemGetAddressRange = reinterpret_cast<decltype(cu_MemGetAddressRange)>(reinterpret_cast<void*>(GetProcAddress(cuda, "cuMemGetAddressRange_v2")));
    cu_MemcpyDtoH = reinterpret_cast<decltype(cu_MemcpyDtoH)>(reinterpret_cast<void*>(GetProcAddress(cuda, "cuMemcpyDtoH_v2")));
    cu_ModuleLoadDataEx = reinterpret_cast<decltype(cu_ModuleLoadDataEx)>(reinterpret_cast<void*>(GetProcAddress(cuda, "cuModuleLoadDataEx")));
    cu_MemAlloc = reinterpret_cast<decltype(cu_MemAlloc)>(reinterpret_cast<void*>(GetProcAddress(cuda, "cuMemAlloc_v2")));
    cu_MemcpyHtoD = reinterpret_cast<decltype(cu_MemcpyHtoD)>(reinterpret_cast<void*>(GetProcAddress(cuda, "cuMemcpyHtoD_v2")));
    cu_CtxGetCurrent = reinterpret_cast<decltype(cu_CtxGetCurrent)>(reinterpret_cast<void*>(GetProcAddress(cuda, "cuCtxGetCurrent")));
    cu_CtxPushCurrent = reinterpret_cast<decltype(cu_CtxPushCurrent)>(reinterpret_cast<void*>(GetProcAddress(cuda, "cuCtxPushCurrent_v2")));
    cu_CtxPopCurrent = reinterpret_cast<decltype(cu_CtxPopCurrent)>(reinterpret_cast<void*>(GetProcAddress(cuda, "cuCtxPopCurrent_v2")));
    cu_StreamSynchronize = reinterpret_cast<decltype(cu_StreamSynchronize)>(reinterpret_cast<void*>(GetProcAddress(cuda, "cuStreamSynchronize")));
    logf("capture: F9 dumps two consecutive frames (max %d sets); helpers %s", kMaxDumpSets,
         cu_MemGetAddressRange && cu_MemcpyDtoH && cu_StreamSynchronize ? "ok" : "MISSING");
    hook("cuModuleLoadData", &hook_cuModuleLoadData, &real_cuModuleLoadData);
    hook("cuModuleUnload", &hook_cuModuleUnload, &real_cuModuleUnload);
    hook("cuModuleGetFunction", &hook_cuModuleGetFunction, &real_cuModuleGetFunction);
    hook("cuLaunchKernel", &hook_cuLaunchKernel, &real_cuLaunchKernel);
    hook("cuGraphLaunch", &hook_cuGraphLaunch, &real_cuGraphLaunch);
    hook("cuStreamBeginCapture_v2", &hook_cuStreamBeginCapture_v2, &real_cuStreamBeginCapture_v2);
    hook("cuStreamEndCapture", &hook_cuStreamEndCapture, &real_cuStreamEndCapture);
    hook("cuArrayCreate_v2", &hook_cuArrayCreate_v2, &real_cuArrayCreate_v2);
    hook("cuExternalMemoryGetMappedMipmappedArray", &hook_cuExternalMemoryGetMappedMipmappedArray,
         &real_cuExternalMemoryGetMappedMipmappedArray);
    hook("cuSignalExternalSemaphoresAsync", &hook_cuSignalExternalSemaphoresAsync, &real_cuSignalExternalSemaphoresAsync);
    hook("cuWaitExternalSemaphoresAsync", &hook_cuWaitExternalSemaphoresAsync, &real_cuWaitExternalSemaphoresAsync);
    hook("cuMemcpy2DAsync_v2", &hook_cuMemcpy2DAsync_v2, &real_cuMemcpy2DAsync_v2);
    hook("cuStreamSynchronize", &hook_cuStreamSynchronize, &real_cuStreamSynchronize);
    hook("cuImportExternalMemory", &hook_cuImportExternalMemory, &real_cuImportExternalMemory);
    hook("cuImportExternalSemaphore", &hook_cuImportExternalSemaphore, &real_cuImportExternalSemaphore);
    if (LoadLibraryW(L"nvapi64.dll")) {
        const MH_STATUS nvapi = MH_CreateHookApi(L"nvapi64.dll", "nvapi_QueryInterface",
                                                 reinterpret_cast<void*>(&hook_nvapi_QueryInterface),
                                                 reinterpret_cast<void**>(&real_nvapi_QueryInterface));
        logf("hook nvapi_QueryInterface                     %s", MH_StatusToString(nvapi));
    } else {
        logf("nvapi64.dll not loadable (error %lu); flip-metering calls will not be logged", GetLastError());
    }
    const MH_STATUS status = MH_EnableHook(MH_ALL_HOOKS);
    logf("hooks enabled: %s. NvPresent already loaded: %s", MH_StatusToString(status),
         GetModuleHandleW(L"NvPresent64.dll") ? "yes (earlier module loads were missed)" : "no");
    return 0;
}

}  // namespace

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(instance);
        if (!smm_load_real_version()) return FALSE;
        InitializeCriticalSection(&g_lock);
        QueryPerformanceFrequency(&g_qpf);
        QueryPerformanceCounter(&g_qpc0);
        if (const HANDLE thread = CreateThread(nullptr, 0, bootstrap, nullptr, 0, nullptr)) CloseHandle(thread);
    }
    return TRUE;
}
