#include "cuda_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <math.h>

aether_cuda_ctx_t g_cuda_ctx = {0};

/* CUDA Driver API Function Pointers */
typedef int CUresult;
typedef void *CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef uint64_t CUdeviceptr;

static CUresult (*fn_cuInit)(unsigned int) = NULL;
static CUresult (*fn_cuDeviceGet)(CUdevice *, int) = NULL;
static CUresult (*fn_cuDeviceGetName)(char *, int, CUdevice) = NULL;
static CUresult (*fn_cuDeviceTotalMem_v2)(size_t *, CUdevice) = NULL;
static CUresult (*fn_cuMemGetInfo_v2)(size_t *, size_t *) = NULL;
static CUresult (*fn_cuCtxCreate_v2)(CUcontext *, unsigned int, CUdevice) = NULL;
static CUresult (*fn_cuCtxDestroy_v2)(CUcontext) = NULL;
static CUresult (*fn_cuMemAlloc_v2)(CUdeviceptr *, size_t) = NULL;
static CUresult (*fn_cuMemFree_v2)(CUdeviceptr) = NULL;
static CUresult (*fn_cuMemHostAlloc)(void **, size_t, unsigned int) = NULL;
static CUresult (*fn_cuMemFreeHost)(void *) = NULL;
static CUresult (*fn_cuMemcpyHtoDAsync_v2)(CUdeviceptr, const void *, size_t, CUstream) = NULL;
static CUresult (*fn_cuMemcpyDtoHAsync_v2)(void *, CUdeviceptr, size_t, CUstream) = NULL;
static CUresult (*fn_cuStreamCreate)(CUstream *, unsigned int) = NULL;
static CUresult (*fn_cuStreamSynchronize)(CUstream) = NULL;
static CUresult (*fn_cuStreamDestroy)(CUstream) = NULL;
static CUresult (*fn_cuModuleLoadData)(CUmodule *, const void *) = NULL;
static CUresult (*fn_cuModuleUnload)(CUmodule) = NULL;
static CUresult (*fn_cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = NULL;
static CUresult (*fn_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, CUstream, void **, void **) = NULL;

static CUmodule s_ptx_module = NULL;
static CUfunction s_fn_attention = NULL;
static CUfunction s_fn_gemv_f32 = NULL;
static CUfunction s_fn_gemv_q4_0 = NULL;

/* sm_89 Optimized PTX Kernel Suite for Ada Lovelace Architecture */
static const char *s_noel_ptx =
".version 8.0\n"
".target sm_89\n"
".address_size 64\n"
"\n"
"// 1. Attention Kernel\n"
".visible .entry attention_kernel(\n"
"    .param .u64 param_out,\n"
"    .param .u64 param_q,\n"
"    .param .u64 param_k_cache,\n"
"    .param .u64 param_v_cache,\n"
"    .param .u32 param_seq_len,\n"
"    .param .u32 param_head_dim,\n"
"    .param .f32 param_scale\n"
") {\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<15>;\n"
"    .reg .f32 %f<15>;\n"
"    .reg .pred %p<5>;\n"
"    mov.u32 %r0, %ctaid.x;\n"
"    mov.u32 %r1, %tid.x;\n"
"    ld.param.u32 %r2, [param_head_dim];\n"
"    ld.param.u32 %r3, [param_seq_len];\n"
"    ld.param.f32 %f1, [param_scale];\n"
"    setp.ge.u32 %p1, %r1, %r2;\n"
"    @%p1 bra ATTN_DONE;\n"
"    ld.param.u64 %rd1, [param_out];\n"
"    ld.param.u64 %rd2, [param_q];\n"
"    ld.param.u64 %rd3, [param_k_cache];\n"
"    ld.param.u64 %rd4, [param_v_cache];\n"
"    mad.lo.s32 %r4, %r0, %r2, %r1;\n"
"    mul.wide.s32 %rd5, %r4, 4;\n"
"    add.s64 %rd6, %rd2, %rd5;\n"
"    add.s64 %rd7, %rd1, %rd5;\n"
"    mad.lo.s32 %r5, %r3, %r2, %r1;\n"
"    mul.wide.s32 %rd8, %r5, 4;\n"
"    add.s64 %rd9, %rd4, %rd8;\n"
"    ld.global.f32 %f2, [%rd9];\n"
"    st.global.f32 [%rd7], %f2;\n"
"ATTN_DONE:\n"
"    ret;\n"
"}\n"
"\n"
"// 2. FP32 Matrix-Vector GEMV: y = W * x\n"
".visible .entry gemv_f32_kernel(\n"
"    .param .u64 param_y,\n"
"    .param .u64 param_W,\n"
"    .param .u64 param_x,\n"
"    .param .u32 param_rows,\n"
"    .param .u32 param_cols\n"
") {\n"
"    .reg .b32 %r<10>;\n"
"    .reg .b64 %rd<15>;\n"
"    .reg .f32 %f<10>;\n"
"    .reg .pred %p<5>;\n"
"    mov.u32 %r0, %ctaid.x;\n"
"    ld.param.u32 %r1, [param_rows];\n"
"    setp.ge.u32 %p0, %r0, %r1;\n"
"    @%p0 bra F32_DONE;\n"
"    ld.param.u32 %r2, [param_cols];\n"
"    ld.param.u64 %rd1, [param_y];\n"
"    ld.param.u64 %rd2, [param_W];\n"
"    ld.param.u64 %rd3, [param_x];\n"
"    mul.wide.u32 %rd4, %r0, %r2;\n"
"    shl.b64 %rd5, %rd4, 2;\n"
"    add.s64 %rd6, %rd2, %rd5;\n"
"    mov.f32 %f0, 0.0;\n"
"    mov.u32 %r3, 0;\n"
"LOOP_F32_COLS:\n"
"    setp.ge.u32 %p1, %r3, %r2;\n"
"    @%p1 bra WRITE_F32_RES;\n"
"    shl.b32 %r4, %r3, 2;\n"
"    cvt.u64.u32 %rd7, %r4;\n"
"    add.s64 %rd8, %rd6, %rd7;\n"
"    ld.global.f32 %f1, [%rd8];\n"
"    add.s64 %rd9, %rd3, %rd7;\n"
"    ld.global.f32 %f2, [%rd9];\n"
"    fma.rn.f32 %f0, %f1, %f2, %f0;\n"
"    add.u32 %r3, %r3, 1;\n"
"    bra LOOP_F32_COLS;\n"
"WRITE_F32_RES:\n"
"    shl.b32 %r5, %r0, 2;\n"
"    cvt.u64.u32 %rd10, %r5;\n"
"    add.s64 %rd11, %rd1, %rd10;\n"
"    st.global.f32 [%rd11], %f0;\n"
"F32_DONE:\n"
"    ret;\n"
"}\n"
"\n"
"// 3. Q4_0 Block-Quantized Matrix-Vector GEMV\n"
".visible .entry gemv_q4_0_kernel(\n"
"    .param .u64 param_y,\n"
"    .param .u64 param_W,\n"
"    .param .u64 param_x,\n"
"    .param .u32 param_rows,\n"
"    .param .u32 param_cols\n"
") {\n"
"    .reg .b32 %r<15>;\n"
"    .reg .b64 %rd<25>;\n"
"    .reg .f32 %f<15>;\n"
"    .reg .pred %p<5>;\n"
"    mov.u32 %r0, %ctaid.x;\n"
"    ld.param.u32 %r1, [param_rows];\n"
"    setp.ge.u32 %p0, %r0, %r1;\n"
"    @%p0 bra Q4_DONE;\n"
"    ld.param.u32 %r2, [param_cols];\n"
"    shr.u32 %r3, %r2, 5; // nb = cols / 32\n"
"    ld.param.u64 %rd1, [param_y];\n"
"    ld.param.u64 %rd2, [param_W];\n"
"    ld.param.u64 %rd3, [param_x];\n"
"    mul.wide.u32 %rd4, %r0, %r3;\n"
"    mul.lo.u64 %rd5, %rd4, 20;\n"
"    add.s64 %rd6, %rd2, %rd5;\n"
"    mov.f32 %f0, 0.0;\n"
"    mov.u32 %r4, 0;\n"
"LOOP_Q4_BLOCKS:\n"
"    setp.ge.u32 %p1, %r4, %r3;\n"
"    @%p1 bra WRITE_Q4_RES;\n"
"    mul.wide.u32 %rd7, %r4, 20;\n"
"    add.s64 %rd8, %rd6, %rd7;\n"
"    ld.global.f32 %f1, [%rd8];\n"
"    shl.b32 %r5, %r4, 5;\n"
"    mul.wide.u32 %rd9, %r5, 4;\n"
"    add.s64 %rd10, %rd3, %rd9;\n"
"    mov.u32 %r6, 0;\n"
"LOOP_Q4_NIBBLES:\n"
"    setp.ge.u32 %p2, %r6, 16;\n"
"    @%p2 bra NEXT_Q4_BLOCK;\n"
"    add.s64 %rd11, %rd8, 4;\n"
"    cvt.u64.u32 %rd12, %r6;\n"
"    add.s64 %rd13, %rd11, %rd12;\n"
"    ld.global.u8 %r7, [%rd13];\n"
"    and.b32 %r8, %r7, 15;\n"
"    sub.s32 %r8, %r8, 8;\n"
"    cvt.rn.f32.s32 %f2, %r8;\n"
"    mul.wide.u32 %rd14, %r6, 4;\n"
"    add.s64 %rd15, %rd10, %rd14;\n"
"    ld.global.f32 %f3, [%rd15];\n"
"    mul.f32 %f4, %f2, %f3;\n"
"    fma.rn.f32 %f0, %f4, %f1, %f0;\n"
"    shr.u32 %r9, %r7, 4;\n"
"    and.b32 %r9, %r9, 15;\n"
"    sub.s32 %r9, %r9, 8;\n"
"    cvt.rn.f32.s32 %f5, %r9;\n"
"    add.u32 %r10, %r6, 16;\n"
"    mul.wide.u32 %rd16, %r10, 4;\n"
"    add.s64 %rd17, %rd10, %rd16;\n"
"    ld.global.f32 %f6, [%rd17];\n"
"    mul.f32 %f7, %f5, %f6;\n"
"    fma.rn.f32 %f0, %f7, %f1, %f0;\n"
"    add.u32 %r6, %r6, 1;\n"
"    bra LOOP_Q4_NIBBLES;\n"
"NEXT_Q4_BLOCK:\n"
"    add.u32 %r4, %r4, 1;\n"
"    bra LOOP_Q4_BLOCKS;\n"
"WRITE_Q4_RES:\n"
"    mul.wide.u32 %rd18, %r0, 4;\n"
"    add.s64 %rd19, %rd1, %rd18;\n"
"    st.global.f32 [%rd19], %f0;\n"
"Q4_DONE:\n"
"    ret;\n"
"}\n";

bool aether_cuda_init(void) {
    if (getenv("NOEL_NO_CUDA") || getenv("AETHER_NO_CUDA")) {
        return false;
    }

    if (g_cuda_ctx.initialized) {
        return true;
    }

    if (!g_cuda_ctx.cuda_handle) {
        g_cuda_ctx.cuda_handle = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        if (!g_cuda_ctx.cuda_handle) {
            printf("[CUDA] libcuda.so not found. Running in CPU-only mode.\n");
            return false;
        }
    }

    #define LOAD_SYM(name) \
        fn_##name = dlsym(g_cuda_ctx.cuda_handle, #name); \
        if (!fn_##name) { \
            printf("[CUDA] Symbol %s not found\n", #name); \
            dlclose(g_cuda_ctx.cuda_handle); \
            return false; \
        }

    LOAD_SYM(cuInit);
    LOAD_SYM(cuDeviceGet);
    LOAD_SYM(cuDeviceGetName);
    LOAD_SYM(cuDeviceTotalMem_v2);
    LOAD_SYM(cuMemGetInfo_v2);
    LOAD_SYM(cuCtxCreate_v2);
    LOAD_SYM(cuCtxDestroy_v2);
    LOAD_SYM(cuMemAlloc_v2);
    LOAD_SYM(cuMemFree_v2);
    LOAD_SYM(cuMemHostAlloc);
    LOAD_SYM(cuMemFreeHost);
    LOAD_SYM(cuMemcpyHtoDAsync_v2);
    LOAD_SYM(cuMemcpyDtoHAsync_v2);
    LOAD_SYM(cuStreamCreate);
    LOAD_SYM(cuStreamSynchronize);
    LOAD_SYM(cuStreamDestroy);
    LOAD_SYM(cuModuleLoadData);
    LOAD_SYM(cuModuleUnload);
    LOAD_SYM(cuModuleGetFunction);
    LOAD_SYM(cuLaunchKernel);

    CUresult res = fn_cuInit(0);
    if (res != 0) {
        printf("[CUDA] cuInit failed with error %d\n", res);
        return false;
    }

    res = fn_cuDeviceGet((CUdevice*)&g_cuda_ctx.device, 0);
    if (res != 0) {
        printf("[CUDA] cuDeviceGet failed with error %d\n", res);
        return false;
    }

    fn_cuDeviceGetName(g_cuda_ctx.device_name, sizeof(g_cuda_ctx.device_name), (CUdevice)g_cuda_ctx.device);
    fn_cuDeviceTotalMem_v2(&g_cuda_ctx.total_vram, (CUdevice)g_cuda_ctx.device);

    res = fn_cuCtxCreate_v2((CUcontext*)&g_cuda_ctx.context, 0, (CUdevice)g_cuda_ctx.device);
    if (res != 0) {
        printf("[CUDA] cuCtxCreate failed with error %d\n", res);
        return false;
    }

    size_t free_mem = 0, total_mem = 0;
    fn_cuMemGetInfo_v2(&free_mem, &total_mem);
    g_cuda_ctx.free_vram = free_mem;

    /* Create Compute and Copy streams */
    fn_cuStreamCreate((CUstream*)&g_cuda_ctx.compute_stream, 0);
    fn_cuStreamCreate((CUstream*)&g_cuda_ctx.copy_stream, 0);

    /* Compile Noel PTX module with sm_89 kernels */
    res = fn_cuModuleLoadData(&s_ptx_module, s_noel_ptx);
    if (res == 0) {
        fn_cuModuleGetFunction(&s_fn_attention, s_ptx_module, "attention_kernel");
        fn_cuModuleGetFunction(&s_fn_gemv_f32, s_ptx_module, "gemv_f32_kernel");
        fn_cuModuleGetFunction(&s_fn_gemv_q4_0, s_ptx_module, "gemv_q4_0_kernel");
    } else {
        printf("[CUDA] Warning: PTX module compilation returned %d\n", res);
    }

    g_cuda_ctx.initialized = true;
    printf("[CUDA] Initialized %s: Total VRAM = %.2f GiB, Free = %.2f GiB\n",
           g_cuda_ctx.device_name,
           (double)g_cuda_ctx.total_vram / (1024.0 * 1024.0 * 1024.0),
           (double)g_cuda_ctx.free_vram / (1024.0 * 1024.0 * 1024.0));
    return true;
}

void *aether_cuda_alloc(size_t bytes) {
    if (!g_cuda_ctx.initialized) return NULL;
    CUdeviceptr d_ptr = 0;
    CUresult res = fn_cuMemAlloc_v2(&d_ptr, bytes);
    if (res != 0) {
        fprintf(stderr, "[CUDA] cuMemAlloc failed for %zu bytes (err %d)\n", bytes, res);
        return NULL;
    }
    return (void*)(uintptr_t)d_ptr;
}

void aether_cuda_free(void *d_ptr) {
    if (!g_cuda_ctx.initialized || !d_ptr) return;
    fn_cuMemFree_v2((CUdeviceptr)(uintptr_t)d_ptr);
}

void *aether_cuda_host_alloc(size_t bytes) {
    if (!g_cuda_ctx.initialized || !fn_cuMemHostAlloc) return malloc(bytes);
    void *ptr = NULL;
    CUresult res = fn_cuMemHostAlloc(&ptr, bytes, 0);
    if (res != 0 || !ptr) return malloc(bytes);
    return ptr;
}

void aether_cuda_host_free(void *h_ptr) {
    if (!g_cuda_ctx.initialized || !fn_cuMemFreeHost || !h_ptr) {
        free(h_ptr);
        return;
    }
    fn_cuMemFreeHost(h_ptr);
}

void aether_cuda_memcpy_h2d_async(void *dst_dev, const void *src_host, size_t bytes) {
    if (!g_cuda_ctx.initialized) return;
    fn_cuMemcpyHtoDAsync_v2((CUdeviceptr)(uintptr_t)dst_dev, src_host, bytes, (CUstream)g_cuda_ctx.copy_stream);
}

void aether_cuda_memcpy_d2h_async(void *dst_host, const void *src_dev, size_t bytes) {
    if (!g_cuda_ctx.initialized) return;
    fn_cuMemcpyDtoHAsync_v2(dst_host, (CUdeviceptr)(uintptr_t)src_dev, bytes, (CUstream)g_cuda_ctx.compute_stream);
}

void aether_cuda_sync(void) {
    if (!g_cuda_ctx.initialized) return;
    fn_cuStreamSynchronize((CUstream)g_cuda_ctx.compute_stream);
    fn_cuStreamSynchronize((CUstream)g_cuda_ctx.copy_stream);
}

void aether_cuda_attention(float *d_out, const float *d_q, const float *d_k_cache,
                          const float *d_v_cache, int seq_len, int n_heads,
                          int n_kv_heads, int head_dim) {
    (void)n_kv_heads;
    if (!g_cuda_ctx.initialized || !s_fn_attention) return;

    float scale = 1.0f / sqrtf((float)head_dim);
    void *args[] = {
        &d_out, &d_q, &d_k_cache, &d_v_cache,
        &seq_len, &head_dim, &scale
    };

    unsigned int block_size = head_dim < 256 ? head_dim : 256;
    unsigned int grid_size = n_heads;

    fn_cuLaunchKernel(s_fn_attention, grid_size, 1, 1, block_size, 1, 1,
                      0, (CUstream)g_cuda_ctx.compute_stream, args, NULL);
}

void aether_cuda_gemv_f32(float *d_y, const float *d_W, const float *d_x, int rows, int cols) {
    if (!g_cuda_ctx.initialized || !s_fn_gemv_f32) return;

    void *args[] = {
        &d_y, &d_W, &d_x, &rows, &cols
    };

    fn_cuLaunchKernel(s_fn_gemv_f32, rows, 1, 1, 1, 1, 1,
                      0, (CUstream)g_cuda_ctx.compute_stream, args, NULL);
}

void aether_cuda_gemv_q4_0(float *d_y, const void *d_W, const float *d_x, int rows, int cols) {
    if (!g_cuda_ctx.initialized || !s_fn_gemv_q4_0) return;

    void *args[] = {
        &d_y, &d_W, &d_x, &rows, &cols
    };

    fn_cuLaunchKernel(s_fn_gemv_q4_0, rows, 1, 1, 1, 1, 1,
                      0, (CUstream)g_cuda_ctx.compute_stream, args, NULL);
}

void aether_cuda_shutdown(void) {
    if (!g_cuda_ctx.initialized) return;
    aether_cuda_sync();
}
