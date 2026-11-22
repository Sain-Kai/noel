#ifndef NOEL_KV_CACHE_H
#define NOEL_KV_CACHE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "noel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NOEL_KV_FP32      = 0,
    NOEL_KV_FP8_E4M3  = 1,
    NOEL_KV_FP8_E5M2  = 2,
    NOEL_KV_Q8_0      = 3,
    NOEL_KV_Q4_0      = 4
} noel_kv_dtype_t;

typedef struct {
    noel_kv_dtype_t dtype;
    int n_layers;
    int seq_len;
    int n_kv_heads;
    int head_dim;
    int kv_dim;               /* n_kv_heads * head_dim */
    
    void *k_data;             /* Allocated key storage */
    void *v_data;             /* Allocated value storage */
    float *k_scales;          /* Per-token/head scales for FP8/Quantized */
    float *v_scales;
    
    size_t row_bytes;         /* Bytes per layer per sequence position */
    size_t total_bytes;       /* Total memory allocated */
    size_t fp32_equivalent_bytes; /* Baseline FP32 memory size */
} noel_kv_cache_t;

/* Cache Lifecycle APIs */
int noel_kv_cache_init(noel_kv_cache_t *cache,
                       noel_kv_dtype_t dtype,
                       int n_layers,
                       int seq_len,
                       int n_kv_heads,
                       int head_dim);

void noel_kv_cache_free(noel_kv_cache_t *cache);
void noel_kv_cache_reset(noel_kv_cache_t *cache);

const char *noel_kv_dtype_name(noel_kv_dtype_t dtype);
size_t noel_kv_dtype_row_bytes(noel_kv_dtype_t dtype, int kv_dim);
double noel_kv_cache_compression_ratio(const noel_kv_cache_t *cache);

/* FP8 Codecs & Row Conversions */
uint8_t noel_f32_to_fp8_e4m3(float x);
float noel_fp8_e4m3_to_f32(uint8_t val);
uint8_t noel_f32_to_fp8_e5m2(float x);
float noel_fp8_e5m2_to_f32(uint8_t val);

void noel_quantize_row_fp8_e4m3(uint8_t *dst, const float *src, int n, float *out_scale);
void noel_dequantize_row_fp8_e4m3(float *dst, const uint8_t *src, int n, float scale);
void noel_quantize_row_fp8_e5m2(uint8_t *dst, const float *src, int n, float *out_scale);
void noel_dequantize_row_fp8_e5m2(float *dst, const uint8_t *src, int n, float scale);

#ifdef __cplusplus
}
#endif

#endif /* NOEL_KV_CACHE_H */
