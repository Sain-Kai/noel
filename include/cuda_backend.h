#ifndef AETHER_CUDA_BACKEND_H
#define AETHER_CUDA_BACKEND_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *cuda_handle;
    void *context;
    void *device;
    char device_name[256];
    size_t total_vram;
    size_t free_vram;
    void *compute_stream;
    void *copy_stream;
    bool initialized;
} aether_cuda_ctx_t;

extern aether_cuda_ctx_t g_cuda_ctx;

/* Initialize CUDA Driver API and create context on RTX 4050 */
bool aether_cuda_init(void);

/* Allocate device memory in VRAM */
void *aether_cuda_alloc(size_t bytes);

/* Free device memory */
void aether_cuda_free(void *d_ptr);

/* Asynchronous copy from Host RAM to GPU VRAM */
void aether_cuda_memcpy_h2d_async(void *dst_dev, const void *src_host, size_t bytes);

/* Asynchronous copy from GPU VRAM to Host RAM */
void aether_cuda_memcpy_d2h_async(void *dst_host, const void *src_dev, size_t bytes);

/* Synchronize compute stream */
void aether_cuda_sync(void);

/* Shutdown CUDA driver and release context */
void aether_cuda_shutdown(void);

/* Run GPU Self-Attention forward pass */
void aether_cuda_attention(float *d_out, const float *d_q, const float *d_k_cache,
                          const float *d_v_cache, int seq_len, int n_heads,
                          int n_kv_heads, int head_dim);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_CUDA_BACKEND_H */
