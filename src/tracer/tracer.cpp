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
#include <d3d11.h>
#include <dxgi.h>

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
enum class TestAction { Timeline, CaptureBuffers, CycleTime, Mfg };
constexpr TestAction kTestAction = TestAction::Mfg;
constexpr int kTestKey = VK_F10;

// ---- multi-frame (3x) state, filled in by the CUDA and D3D11 hooks below ----
struct Link {
    void* key;
    void* value;
};
struct LinkTable {
    Link items[64];
    unsigned next = 0;
    void add(void* key, void* value) { items[next++ % 64] = {key, value}; }
    void* find(void* key) const {  // most recent entry wins
        for (unsigned i = 0; i < 64; ++i) {
            const Link& l = items[(next - 1 - i) % 64];
            if (l.key == key && key) return l.value;
        }
        return nullptr;
    }
};
LinkTable g_extmem_to_handle, g_mip_to_extmem, g_array_to_mip, g_surf_to_array, g_handle_to_texture, g_texture_to_device;

bool g_mfg_on = false, g_mfg_ready = false, g_mfg_pending = false, g_mfg_key_down = false;
DWORD g_mfg_tid = 0;
void* g_warp_function = nullptr;
unsigned g_warp_dims[7] = {};
uint8_t g_warp_block[136] = {};
void* g_mfg_stream = nullptr;
uint64_t g_mfg_buffer = 0, g_scale_memory2 = 0;
uint32_t g_mfg_count = 0;
void* g_out_texture = nullptr;  // ID3D11Texture2D on NvPresent's device that CUDA writes the generated frame into
void* g_out_device = nullptr;
double g_frame_ms = 0, g_last_launch_ms = 0, g_extra_cost_ms = 3.0;
uint64_t g_extra_frames = 0, g_extra_skipped = 0;
bool g_real_pending = false;
double g_real_target_ms = 0, g_ready_ms = 0;

#define SMM_VERSION "prototype 0.8.1"

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

// True when the caller is NvPresent64.dll. The module can be unloaded and loaded again at a
// different address (seen when nvapi64.dll is loaded early), so a miss re-checks by address.
bool from_nvpresent(void* return_address) {
    const auto* ra = static_cast<const uint8_t*>(return_address);
    if (g_nvp && ra >= g_nvp && ra < g_nvp + g_nvp_size) return true;
    HMODULE m = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(return_address), &m) || !m)
        return false;
    auto* base = reinterpret_cast<uint8_t*>(m);
    if (base == g_nvp) return false;
    wchar_t path[MAX_PATH];
    const DWORD n = GetModuleFileNameW(m, path, MAX_PATH);
    static const wchar_t kName[] = L"NvPresent64.dll";
    const DWORD name_len = sizeof(kName) / sizeof(wchar_t) - 1;
    if (n < name_len || lstrcmpiW(path + n - name_len, kName) != 0) return false;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(
        base + reinterpret_cast<const IMAGE_DOS_HEADER*>(base)->e_lfanew);
    g_nvp_size = nt->OptionalHeader.SizeOfImage;
    g_nvp = base;
    logf("NvPresent64.dll at %p size 0x%zx path %ls", base, g_nvp_size, path);
    return true;
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

int upload_scales(float t) {
    const float scales[2] = {2.0f * t, 2.0f * (1.0f - t)};
    return cu_MemcpyHtoD(g_scale_memory, scales, sizeof(scales));
}

int upload_time() { return upload_scales(kTimes[g_time_index]); }

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
            // Second pass of 3x: the buffer already holds t = 1/3 flows; x2 and x0.5 turn them into t = 2/3.
            job->step = "allocate 2";
            job->rc = cu_MemAlloc(&g_scale_memory2, 8);
            const float second[2] = {2.0f, 0.5f};
            if (job->rc == 0) job->rc = cu_MemcpyHtoD(g_scale_memory2, second, sizeof(second));
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
    g_mfg_buffer = buffer;
    g_mfg_count = count;
    g_mfg_stream = stream;
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
    if (is_warp) {
        inject_time_shift(warp_block, stream);
        g_warp_function = function;
        const unsigned dims[7] = {gx, gy, gz, bx, by, bz, shared};
        memcpy(g_warp_dims, dims, sizeof(dims));
        memcpy(g_warp_block, warp_block, sizeof(g_warp_block));
    }
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

int (*real_cuMipmappedArrayGetLevel)(void**, void*, unsigned);
int (*real_cuSurfObjectCreate)(uint64_t*, const void*);

int hook_cuMipmappedArrayGetLevel(void** array, void* mipmap, unsigned level) {
    const int rc = real_cuMipmappedArrayGetLevel(array, mipmap, level);
    if (rc == 0 && array && from_nvpresent_quick(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        g_array_to_mip.add(*array, mipmap);
        LeaveCriticalSection(&g_lock);
    }
    return rc;
}

int hook_cuSurfObjectCreate(uint64_t* surface, const void* descriptor) {
    const int rc = real_cuSurfObjectCreate(surface, descriptor);
    if (rc == 0 && surface && from_nvpresent_quick(__builtin_return_address(0)) && readable(descriptor, 16)) {
        EnterCriticalSection(&g_lock);
        void* array = reinterpret_cast<void*>(rd<uint64_t>(static_cast<const uint8_t*>(descriptor) + 8));
        g_surf_to_array.add(reinterpret_cast<void*>(*surface), array);
        logf("surface object 0x%llx on array %p", static_cast<unsigned long long>(*surface), array);
        LeaveCriticalSection(&g_lock);
    }
    return rc;
}

int hook_cuImportExternalMemory(void** memory, const void* descriptor) {
    const int rc = real_cuImportExternalMemory(memory, descriptor);
    if (rc == 0 && memory && readable(descriptor, 16) && from_nvpresent_quick(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        g_extmem_to_handle.add(*memory, reinterpret_cast<void*>(rd<uint64_t>(static_cast<const uint8_t*>(descriptor) + 8)));
        LeaveCriticalSection(&g_lock);
    }
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

    if (g_last_launch_ms > 0 && t0 - g_last_launch_ms < 250.0) {
        const double dt = t0 - g_last_launch_ms;
        g_frame_ms = g_frame_ms > 0 ? 0.9 * g_frame_ms + 0.1 * dt : dt;
    }
    g_last_launch_ms = t0;
    if (kTestAction == TestAction::Mfg) {
        const bool mfg_key = (GetAsyncKeyState(kTestKey) & 0x8000) != 0;
        if (mfg_key && !g_mfg_key_down) {
            if (g_mfg_ready) {
                g_mfg_on = !g_mfg_on;
                const int upload_rc = upload_scales(g_mfg_on ? 1.0f / 3.0f : 0.5f);
                logf("3x: F10 -> %s (upload rc=%d, base frame %.2f ms)", g_mfg_on ? "ON" : "off", upload_rc, g_frame_ms);
                g_timeline = 900;
                Beep(g_mfg_on ? 1200 : 500, 80);
            } else {
                logf("3x: F10 pressed but not ready");
                Beep(250, 250);
            }
        }
        g_mfg_key_down = mfg_key;
        g_mfg_pending = g_mfg_on && g_mfg_ready && rc == 0;
        g_mfg_tid = GetCurrentThreadId();
    }
    const bool tl_key = kTestAction == TestAction::Timeline && (GetAsyncKeyState(kTestKey) & 0x8000) != 0;
    if ((tl_key && !g_timeline_key_down) || (frame == 300 && !g_timeline_auto_done)) {
        g_timeline_auto_done = true;
        g_timeline = 700;
        logf("timeline: recording the next 700 events (graph frame %llu)", static_cast<unsigned long long>(frame));
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

// Finds the D3D11 texture behind the surface object that warp_coarse_kernel writes to:
// surface object (in the warp parameters) -> CUDA array -> mipmapped array -> external
// memory -> shared handle -> texture opened on NvPresent's device.
void resolve_output_texture() {
    void* texture = nullptr;
    for (size_t offset = 0; offset + 8 <= sizeof(g_warp_block) && !texture; offset += 8) {
        void* candidate = reinterpret_cast<void*>(rd<uint64_t>(g_warp_block + offset));
        void* array = g_surf_to_array.find(candidate);
        if (!array) continue;
        void* mip = g_array_to_mip.find(array);
        void* memory = g_mip_to_extmem.find(mip);
        void* handle = g_extmem_to_handle.find(memory);
        texture = g_handle_to_texture.find(handle);
        logf("3x: warp param +%zu surface=%p array=%p mip=%p extmem=%p handle=%p texture=%p", offset, candidate, array, mip,
             memory, handle, texture);
    }
    if (texture != g_out_texture) {
        if (texture) static_cast<IUnknown*>(texture)->AddRef();
        if (g_out_texture) static_cast<IUnknown*>(g_out_texture)->Release();
        g_out_texture = texture;
        g_out_device = g_texture_to_device.find(texture);
    }
}

int hook_cuStreamBeginCapture_v2(CUstream stream, int mode) {
    void* ra = __builtin_return_address(0);
    EnterCriticalSection(&g_lock);
    if (from_nvpresent(ra)) {
        ++g_capturing;
        g_mfg_ready = false;
        g_warp_function = nullptr;
        g_mfg_buffer = 0;
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
        resolve_output_texture();
        g_mfg_ready = g_warp_function && g_mfg_buffer && g_scale_function && g_scale_memory2 && g_out_texture;
        logf("3x: %s (warp=%p buffer=0x%llx output texture=%p)", g_mfg_ready ? "ready" : "NOT ready", g_warp_function,
             static_cast<unsigned long long>(g_mfg_buffer), g_out_texture);
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
    if (rc == 0 && mipmap && from_nvpresent(ra)) g_mip_to_extmem.add(*mipmap, memory);
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

// ------------------------------------------------------------ D3D11/DXGI
//
// For DX11 games NvPresent shares D3D11 textures and fences with CUDA, so its
// output reaches the screen through D3D11 copies and a DXGI Present. These
// hooks sit on the d3d11.dll / dxgi.dll implementations (addresses taken from
// a throwaway WARP device at start-up, before NvPresent detours anything) and
// log calls made by NvPresent. Present is logged for every caller.

bool nvp_fast(void* return_address) {
    const auto* ra = static_cast<const uint8_t*>(return_address);
    const uint8_t* base = g_nvp;
    if (base && ra >= base && ra < base + g_nvp_size) return true;
    return false;
}

const wchar_t* caller_name(void* return_address, wchar_t* buffer) {
    HMODULE m = nullptr;
    buffer[0] = L'?';
    buffer[1] = 0;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(return_address), &m) && m) {
        wchar_t path[MAX_PATH];
        const DWORD n = GetModuleFileNameW(m, path, MAX_PATH);
        DWORD start = n;
        while (start && path[start - 1] != L'\\') --start;
        lstrcpynW(buffer, path + start, 48);
    }
    return buffer;
}

void describe(void* resource, char* out, size_t out_size) {
    out[0] = 0;
    if (!resource) return;
    auto* r = static_cast<ID3D11Resource*>(resource);
    D3D11_RESOURCE_DIMENSION dimension = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    r->GetType(&dimension);
    if (dimension != D3D11_RESOURCE_DIMENSION_TEXTURE2D) {
        snprintf(out, out_size, "dim%d", int(dimension));
        return;
    }
    D3D11_TEXTURE2D_DESC d;
    static_cast<ID3D11Texture2D*>(r)->GetDesc(&d);
    snprintf(out, out_size, "%ux%u fmt=%d bind=0x%x misc=0x%x", d.Width, d.Height, int(d.Format), d.BindFlags, d.MiscFlags);
}

HRESULT (*real_Present)(void*, UINT, UINT);
HRESULT (*real_Present1)(void*, UINT, UINT, const void*);
void (*real_CopyResource)(void*, void*, void*);
void (*real_CopySubresourceRegion)(void*, void*, UINT, UINT, UINT, UINT, void*, UINT, const void*);
void (*real_Flush)(void*);
HRESULT (*real_CreateTexture2D)(void*, const D3D11_TEXTURE2D_DESC*, const void*, void**);

uint64_t g_present_count = 0;

void log_present(const char* which, void* swapchain, UINT sync, UINT flags, void* ra, double took, HRESULT hr) {
    EnterCriticalSection(&g_lock);
    from_nvpresent(ra);
    wchar_t name[48];
    const bool first = g_present_count++ < 30;
    if (g_timeline > 0) {
        // Which present is on screen, and since when (ms on this log's clock).
        DXGI_FRAME_STATISTICS stats = {};
        UINT last = 0;
        auto* sc = static_cast<IDXGISwapChain*>(swapchain);
        const HRESULT stats_hr = sc->GetFrameStatistics(&stats);
        sc->GetLastPresentCount(&last);
        tl("stats hr=0x%lx submitted=%u displayed=%u refresh=%u sync_refresh=%u at=%.3f", stats_hr, last, stats.PresentCount,
           stats.PresentRefreshCount, stats.SyncRefreshCount,
           1000.0 * double(stats.SyncQPCTime.QuadPart - g_qpc0.QuadPart) / double(g_qpf.QuadPart));
    }
    if (first || g_timeline > 0) {
        caller_name(ra, name);
        if (first)
            logf("%s #%llu sc=%p sync=%u flags=0x%x caller=%ls tid=%lu hr=0x%lx took %.3f ms", which,
                 static_cast<unsigned long long>(g_present_count), swapchain, sync, flags, name, GetCurrentThreadId(), hr, took);
        tl("%s sc=%p sync=%u flags=0x%x caller=%ls took %.3f ms", which, swapchain, sync, flags, name, took);
    }
    LeaveCriticalSection(&g_lock);
}

HRESULT (*real_GetBuffer)(void*, UINT, const IID*, void**);

void wait_until(double target_ms) {
    static HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002 /* high resolution */, TIMER_ALL_ACCESS);
    for (;;) {
        const double left = target_ms - now_ms();
        if (left <= 0) return;
        if (left > 1.5 && timer) {
            LARGE_INTEGER due;
            due.QuadPart = -static_cast<LONGLONG>((left - 1.0) * 10000.0);
            SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
            WaitForSingleObject(timer, 50);
        } else {
            YieldProcessor();
        }
    }
}

// 3x: called right after NvPresent presented its generated frame (built for t = 1/3).
// Builds the t = 2/3 frame with NVIDIA's own warp kernel into the same texture, copies it
// into the back buffer and presents it. NvPresent then presents the real frame as usual.
void present_second_generated_frame(void* self, UINT sync, UINT flags, double first_present_ms) {
    const char* skip = nullptr;
    auto* swapchain = static_cast<IDXGISwapChain*>(self);
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    ID3D11Texture2D* back = nullptr;
    int scale_rc = -1, warp_rc = -1, sync_rc = -1;
    HRESULT hr = E_FAIL;
    double t_start = 0, t_cuda = 0, t_done = 0;
    const IID iid_device = __uuidof(ID3D11Device), iid_texture = __uuidof(ID3D11Texture2D);
    if (FAILED(swapchain->GetDevice(iid_device, reinterpret_cast<void**>(&device))) || !device) skip = "no device";
    else if (device != g_out_device) skip = "swapchain is on a different device";
    else if (FAILED(real_GetBuffer(self, 0, &iid_texture, reinterpret_cast<void**>(&back))) || !back) skip = "no back buffer";
    else {
        D3D11_TEXTURE2D_DESC a, b;
        back->GetDesc(&a);
        static_cast<ID3D11Texture2D*>(g_out_texture)->GetDesc(&b);
        if (a.Width != b.Width || a.Height != b.Height || a.Format != b.Format || a.SampleDesc.Count != b.SampleDesc.Count)
            skip = "back buffer and generated texture differ";
    }
    double ready = 0;
    const double frame = g_frame_ms > 4.0 ? g_frame_ms : 33.0;
    if (!skip) {
        // The first generated frame can only be shown once the graph has finished on the GPU.
        real_cuStreamSynchronize(g_mfg_stream);
        ready = now_ms();
        g_ready_ms = ready;
        // No waiting here: any wait on the game's present thread lengthens the base frame (0.8 showed this).
        t_start = now_ms();
        uint64_t buffer = g_mfg_buffer;
        uint32_t count = g_mfg_count;
        void* scale_params[3] = {&buffer, &g_scale_memory2, &count};
        scale_rc = real_cuLaunchKernel(g_scale_function, (count + 255) / 256, 1, 1, 256, 1, 1, 0, g_mfg_stream, scale_params, nullptr);
        void* warp_params[1] = {g_warp_block};
        if (scale_rc == 0)
            warp_rc = real_cuLaunchKernel(g_warp_function, g_warp_dims[0], g_warp_dims[1], g_warp_dims[2], g_warp_dims[3],
                                          g_warp_dims[4], g_warp_dims[5], g_warp_dims[6], g_mfg_stream, warp_params, nullptr);
        sync_rc = real_cuStreamSynchronize(g_mfg_stream);
        t_cuda = now_ms();
        if (scale_rc == 0 && warp_rc == 0 && sync_rc == 0) {
            device->GetImmediateContext(&context);
            context->CopyResource(back, static_cast<ID3D11Texture2D*>(g_out_texture));
            hr = real_Present(self, sync, flags);
            t_done = now_ms();
            g_extra_cost_ms = 0.9 * g_extra_cost_ms + 0.1 * (t_cuda - t_start + 0.5);
            if (g_extra_cost_ms > frame / 3.0) g_extra_cost_ms = frame / 3.0;
        } else {
            skip = "CUDA launch failed";
        }
    }
    if (context) context->Release();
    if (back) back->Release();
    if (device) device->Release();
    EnterCriticalSection(&g_lock);
    if (skip) {
        if (g_extra_skipped++ < 10) logf("3x: extra frame skipped: %s (scale rc=%d warp rc=%d sync rc=%d)", skip, scale_rc, warp_rc, sync_rc);
        if (g_extra_skipped == 10) {
            g_mfg_on = false;
            upload_scales(0.5f);
            logf("3x: switched off after repeated skips");
        }
    } else {
        const uint64_t n = g_extra_frames++;
        if (n < 20 || n % 600 == 0)
            logf("3x: extra frame %llu: graph ready %.2f ms after launch; second frame at ready+%.2f (cuda %.2f, copy+present %.2f) hr=0x%lx; base frame %.2f ms",
                 static_cast<unsigned long long>(n), ready - g_last_launch_ms, t_done - ready, t_cuda - t_start, t_done - t_cuda, hr, g_frame_ms);
        tl("3x second generated frame presented: ready=launch+%.2f, presented at ready+%.2f (cuda %.2f ms)", ready - g_last_launch_ms,
           t_done - ready, t_cuda - t_start);
    }
    LeaveCriticalSection(&g_lock);
}

HRESULT hook_Present(void* self, UINT sync, UINT flags) {
    void* ra = __builtin_return_address(0);
    const bool same_thread = g_mfg_tid == GetCurrentThreadId();
    const bool first_generated = g_mfg_pending && same_thread;
    if (first_generated) g_mfg_pending = false;
    if (!first_generated && g_real_pending && same_thread) {
        // The real frame: hold it until two thirds of a frame after the first generated one.
        g_real_pending = false;
        wait_until(g_real_target_ms);
        EnterCriticalSection(&g_lock);
        tl("3x real frame released at ready+%.2f", now_ms() - g_ready_ms);
        LeaveCriticalSection(&g_lock);
    }
    const double t0 = now_ms();
    const HRESULT hr = real_Present(self, sync, flags);
    const double t1 = now_ms();
    log_present("Present", self, sync, flags, ra, t1 - t0, hr);
    if (first_generated && SUCCEEDED(hr)) present_second_generated_frame(self, sync, flags, t1);
    return hr;
}

HRESULT hook_Present1(void* self, UINT sync, UINT flags, const void* params) {
    void* ra = __builtin_return_address(0);
    const double t0 = now_ms();
    const HRESULT hr = real_Present1(self, sync, flags, params);
    log_present("Present1", self, sync, flags, ra, now_ms() - t0, hr);
    return hr;
}

void log_copy(const char* which, void* context, void* dst, void* src) {
    static uint64_t count = 0;
    EnterCriticalSection(&g_lock);
    const bool first = count++ < 40;
    if (first || g_timeline > 0) {
        char a[96], b[96];
        describe(dst, a, sizeof(a));
        describe(src, b, sizeof(b));
        if (first) logf("%s ctx=%p dst=%p [%s] src=%p [%s] tid=%lu", which, context, dst, a, src, b, GetCurrentThreadId());
        tl("%s ctx=%p dst=%p [%s] src=%p [%s]", which, context, dst, a, src, b);
    }
    LeaveCriticalSection(&g_lock);
}

void hook_CopyResource(void* self, void* dst, void* src) {
    if (nvp_fast(__builtin_return_address(0))) log_copy("CopyResource", self, dst, src);
    real_CopyResource(self, dst, src);
}

void hook_CopySubresourceRegion(void* self, void* dst, UINT dst_sub, UINT x, UINT y, UINT z, void* src, UINT src_sub,
                                const void* box) {
    if (nvp_fast(__builtin_return_address(0))) log_copy("CopySubresourceRegion", self, dst, src);
    real_CopySubresourceRegion(self, dst, dst_sub, x, y, z, src, src_sub, box);
}

void hook_Flush(void* self) {
    if (nvp_fast(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        tl("Flush ctx=%p", self);
        LeaveCriticalSection(&g_lock);
    }
    real_Flush(self);
}

HRESULT hook_CreateTexture2D(void* self, const D3D11_TEXTURE2D_DESC* desc, const void* initial, void** texture) {
    const HRESULT hr = real_CreateTexture2D(self, desc, initial, texture);
    if (nvp_fast(__builtin_return_address(0)) && desc) {
        EnterCriticalSection(&g_lock);
        logf("CreateTexture2D dev=%p %ux%u fmt=%d mips=%u usage=%d bind=0x%x misc=0x%x hr=0x%lx tex=%p", self, desc->Width,
             desc->Height, int(desc->Format), desc->MipLevels, int(desc->Usage), desc->BindFlags, desc->MiscFlags, hr,
             texture ? *texture : nullptr);
        LeaveCriticalSection(&g_lock);
    }
    return hr;
}

// More of NvPresent's D3D11 use: how an image gets into the back buffer.

void describe_view(ID3D11View* view, char* out, size_t out_size) {
    out[0] = 0;
    if (!view) return;
    ID3D11Resource* resource = nullptr;
    view->GetResource(&resource);
    char text[96];
    describe(resource, text, sizeof(text));
    snprintf(out, out_size, "%p [%s]", static_cast<void*>(resource), text);
    if (resource) resource->Release();
}

void (*real_Draw)(void*, UINT, UINT);
void (*real_DrawIndexed)(void*, UINT, UINT, INT);
void (*real_OMSetRenderTargets)(void*, UINT, ID3D11RenderTargetView* const*, void*);
void (*real_PSSetShaderResources)(void*, UINT, UINT, ID3D11ShaderResourceView* const*);
void (*real_CopySubresourceRegion1)(void*, void*, UINT, UINT, UINT, UINT, void*, UINT, const void*, UINT);
HRESULT (*real_Signal)(void*, void*, UINT64);
HRESULT (*real_Wait)(void*, void*, UINT64);
HRESULT (*real_OpenSharedResource1)(void*, HANDLE, const IID*, void**);
HRESULT (*real_OpenSharedResource)(void*, HANDLE, const IID*, void**);

void hook_Draw(void* self, UINT count, UINT start) {
    if (nvp_fast(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        tl("Draw ctx=%p vertices=%u", self, count);
        LeaveCriticalSection(&g_lock);
    }
    real_Draw(self, count, start);
}

void hook_DrawIndexed(void* self, UINT count, UINT start, INT base) {
    if (nvp_fast(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        tl("DrawIndexed ctx=%p indices=%u", self, count);
        LeaveCriticalSection(&g_lock);
    }
    real_DrawIndexed(self, count, start, base);
}

void hook_OMSetRenderTargets(void* self, UINT count, ID3D11RenderTargetView* const* views, void* depth) {
    if (nvp_fast(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        if (g_timeline > 0) {
            char text[160] = "";
            if (count && views) describe_view(views[0], text, sizeof(text));
            tl("OMSetRenderTargets ctx=%p n=%u target=%s", self, count, text);
        }
        LeaveCriticalSection(&g_lock);
    }
    real_OMSetRenderTargets(self, count, views, depth);
}

void hook_PSSetShaderResources(void* self, UINT start, UINT count, ID3D11ShaderResourceView* const* views) {
    if (nvp_fast(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        if (g_timeline > 0) {
            for (UINT i = 0; i < count && i < 4 && views; ++i) {
                char text[160];
                describe_view(views[i], text, sizeof(text));
                tl("PSSetShaderResources ctx=%p slot=%u source=%s", self, start + i, text);
            }
        }
        LeaveCriticalSection(&g_lock);
    }
    real_PSSetShaderResources(self, start, count, views);
}

void hook_CopySubresourceRegion1(void* self, void* dst, UINT dst_sub, UINT x, UINT y, UINT z, void* src, UINT src_sub,
                                 const void* box, UINT flags) {
    if (nvp_fast(__builtin_return_address(0))) log_copy("CopySubresourceRegion1", self, dst, src);
    real_CopySubresourceRegion1(self, dst, dst_sub, x, y, z, src, src_sub, box, flags);
}

HRESULT hook_Signal(void* self, void* fence, UINT64 value) {
    if (nvp_fast(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        tl("d3d Signal ctx=%p fence=%p value=%llu", self, fence, static_cast<unsigned long long>(value));
        LeaveCriticalSection(&g_lock);
    }
    return real_Signal(self, fence, value);
}

HRESULT hook_Wait(void* self, void* fence, UINT64 value) {
    if (nvp_fast(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        tl("d3d Wait   ctx=%p fence=%p value=%llu", self, fence, static_cast<unsigned long long>(value));
        LeaveCriticalSection(&g_lock);
    }
    return real_Wait(self, fence, value);
}

HRESULT hook_GetBuffer(void* self, UINT index, const IID* iid, void** out) {
    const HRESULT hr = real_GetBuffer(self, index, iid, out);
    if (nvp_fast(__builtin_return_address(0))) {
        static uint64_t count = 0;
        EnterCriticalSection(&g_lock);
        char text[96] = "";
        const bool first = count++ < 12;
        if ((first || g_timeline > 0) && SUCCEEDED(hr) && out && *out && iid && iid->Data1 == 0x6f15aaf2u)  // ID3D11Texture2D
            describe(*out, text, sizeof(text));
        if (first) logf("GetBuffer sc=%p index=%u iid=%08lx -> %p [%s] hr=0x%lx", self, index, iid ? iid->Data1 : 0, out ? *out : nullptr, text, hr);
        tl("GetBuffer sc=%p index=%u -> %p [%s]", self, index, out ? *out : nullptr, text);
        LeaveCriticalSection(&g_lock);
    }
    return hr;
}

void log_open(const char* which, void* device, HANDLE handle, const IID* iid, void** out, HRESULT hr, void* ra) {
    EnterCriticalSection(&g_lock);
    wchar_t name[48];
    char text[96] = "";
    if (SUCCEEDED(hr) && out && *out && iid && iid->Data1 == 0x6f15aaf2u) describe(*out, text, sizeof(text));
    logf("%s dev=%p handle=0x%llx iid=%08lx -> %p [%s] hr=0x%lx caller=%ls", which, device,
         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(handle)), iid ? iid->Data1 : 0, out ? *out : nullptr, text,
         hr, caller_name(ra, name));
    LeaveCriticalSection(&g_lock);
}

HRESULT hook_OpenSharedResource1(void* self, HANDLE handle, const IID* iid, void** out) {
    const HRESULT hr = real_OpenSharedResource1(self, handle, iid, out);
    if (SUCCEEDED(hr) && out && *out && iid && iid->Data1 == 0x6f15aaf2u && nvp_fast(__builtin_return_address(0))) {
        EnterCriticalSection(&g_lock);
        g_handle_to_texture.add(handle, *out);
        g_texture_to_device.add(*out, self);
        LeaveCriticalSection(&g_lock);
    }
    log_open("OpenSharedResource1", self, handle, iid, out, hr, __builtin_return_address(0));
    return hr;
}

HRESULT hook_OpenSharedResource(void* self, HANDLE handle, const IID* iid, void** out) {
    const HRESULT hr = real_OpenSharedResource(self, handle, iid, out);
    log_open("OpenSharedResource", self, handle, iid, out, hr, __builtin_return_address(0));
    return hr;
}

bool inside_module(void* address, const wchar_t* module_name) {
    const HMODULE m = GetModuleHandleW(module_name);
    if (!m) return false;
    auto* base = reinterpret_cast<uint8_t*>(m);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + reinterpret_cast<const IMAGE_DOS_HEADER*>(base)->e_lfanew);
    auto* a = static_cast<uint8_t*>(address);
    return a >= base && a < base + nt->OptionalHeader.SizeOfImage;
}

template <class T>
void hook_address(const char* name, void* target, const wchar_t* module_name, T detour, T* original) {
    if (!inside_module(target, module_name)) {
        logf("hook %-44s skipped: %p is not inside %ls (wrapped by another mod?)", name, target, module_name);
        return;
    }
    const MH_STATUS status = MH_CreateHook(target, reinterpret_cast<void*>(detour), reinterpret_cast<void**>(original));
    logf("hook %-44s %s", name, MH_StatusToString(status));
}

// Creates a software (WARP) device and swapchain on a hidden window only to read the
// d3d11/dxgi method addresses, then releases them.
void install_d3d_hooks() {
    const HMODULE d3d11 = LoadLibraryW(L"d3d11.dll");
    using CreateFn = HRESULT(WINAPI*)(IDXGIAdapter*, D3D_DRIVER_TYPE, HMODULE, UINT, const D3D_FEATURE_LEVEL*, UINT, UINT,
                                      const DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**, ID3D11Device**, D3D_FEATURE_LEVEL*,
                                      ID3D11DeviceContext**);
    const auto create = d3d11 ? reinterpret_cast<CreateFn>(reinterpret_cast<void*>(GetProcAddress(d3d11, "D3D11CreateDeviceAndSwapChain"))) : nullptr;
    if (!create) {
        logf("d3d hooks: D3D11CreateDeviceAndSwapChain not found");
        return;
    }
    const HWND window = CreateWindowExW(0, L"STATIC", L"smm", WS_POPUP, 0, 0, 16, 16, nullptr, nullptr, nullptr, nullptr);
    DXGI_SWAP_CHAIN_DESC desc = {};
    desc.BufferCount = 1;
    desc.BufferDesc.Width = 16;
    desc.BufferDesc.Height = 16;
    desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.OutputWindow = window;
    desc.SampleDesc.Count = 1;
    desc.Windowed = TRUE;
    IDXGISwapChain* swapchain = nullptr;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    const HRESULT hr = create(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &desc, &swapchain,
                              &device, nullptr, &context);
    logf("d3d hooks: probe device hr=0x%lx", hr);
    if (SUCCEEDED(hr) && swapchain && device && context) {
        void** sc = *reinterpret_cast<void***>(swapchain);
        void** ctx = *reinterpret_cast<void***>(context);
        void** dev = *reinterpret_cast<void***>(device);
        hook_address("IDXGISwapChain::Present", sc[8], L"dxgi.dll", &hook_Present, &real_Present);
        hook_address("IDXGISwapChain1::Present1", sc[22], L"dxgi.dll", &hook_Present1, &real_Present1);
        hook_address("ID3D11DeviceContext::CopyResource", ctx[47], L"d3d11.dll", &hook_CopyResource, &real_CopyResource);
        hook_address("ID3D11DeviceContext::CopySubresourceRegion", ctx[46], L"d3d11.dll", &hook_CopySubresourceRegion,
                     &real_CopySubresourceRegion);
        hook_address("ID3D11DeviceContext::Flush", ctx[111], L"d3d11.dll", &hook_Flush, &real_Flush);
        hook_address("ID3D11Device::CreateTexture2D", dev[5], L"d3d11.dll", &hook_CreateTexture2D, &real_CreateTexture2D);
        hook_address("ID3D11DeviceContext::Draw", ctx[13], L"d3d11.dll", &hook_Draw, &real_Draw);
        hook_address("ID3D11DeviceContext::DrawIndexed", ctx[12], L"d3d11.dll", &hook_DrawIndexed, &real_DrawIndexed);
        hook_address("ID3D11DeviceContext::OMSetRenderTargets", ctx[33], L"d3d11.dll", &hook_OMSetRenderTargets, &real_OMSetRenderTargets);
        hook_address("ID3D11DeviceContext::PSSetShaderResources", ctx[8], L"d3d11.dll", &hook_PSSetShaderResources, &real_PSSetShaderResources);
        hook_address("ID3D11DeviceContext1::CopySubresourceRegion1", ctx[115], L"d3d11.dll", &hook_CopySubresourceRegion1, &real_CopySubresourceRegion1);
        hook_address("ID3D11DeviceContext4::Signal", ctx[147], L"d3d11.dll", &hook_Signal, &real_Signal);
        hook_address("ID3D11DeviceContext4::Wait", ctx[148], L"d3d11.dll", &hook_Wait, &real_Wait);
        hook_address("IDXGISwapChain::GetBuffer", sc[9], L"dxgi.dll", &hook_GetBuffer, &real_GetBuffer);
        hook_address("ID3D11Device::OpenSharedResource", dev[28], L"d3d11.dll", &hook_OpenSharedResource, &real_OpenSharedResource);
        hook_address("ID3D11Device1::OpenSharedResource1", dev[48], L"d3d11.dll", &hook_OpenSharedResource1, &real_OpenSharedResource1);
    }
    if (context) context->Release();
    if (swapchain) swapchain->Release();
    if (device) device->Release();
    if (window) DestroyWindow(window);
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
    logf("smooth-motion-mfg " SMM_VERSION " in %ls (test key: F10 = 3x on/off)", exe);

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
    hook("cuMipmappedArrayGetLevel", &hook_cuMipmappedArrayGetLevel, &real_cuMipmappedArrayGetLevel);
    hook("cuSurfObjectCreate", &hook_cuSurfObjectCreate, &real_cuSurfObjectCreate);
    install_d3d_hooks();
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
