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

typedef struct noel_kv_cache_s {
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

/**
 * @brief Apply StreamingLLM attention sinks and rolling sliding window
 *
 * Preserves the first `n_sink` tokens and shifts the latest `n_recent`
 * tokens, discarding intermediate obsolete context to prevent perplexity collapse.
 *
 * @param cache KV cache pointer
 * @param current_len Current sequence length in the cache
 * @param n_sink Number of initial sink tokens to preserve (typically 2-4)
 * @param n_recent Number of recent sliding-window tokens to retain
 * @return New sequence position (n_sink + n_recent) or current_len if no roll occurred
 */
int noel_kv_cache_apply_attention_sink(noel_kv_cache_t *cache, int current_len, int n_sink, int n_recent);

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

/* Block-Quantized Q8_0 and Q4_0 Codecs */
void noel_quantize_row_q8_0(block_q8_0 *dst, const float *src, int n);
void noel_dequantize_row_q8_0(float *dst, const block_q8_0 *src, int n);
void noel_quantize_row_q4_0(block_q4_0 *dst, const float *src, int n);
void noel_dequantize_row_q4_0(float *dst, const block_q4_0 *src, int n);

/* High-Level KV Cache Store & Load */
void noel_kv_cache_store(noel_kv_cache_t *cache, int layer, int pos, const float *k, const float *v);
void noel_kv_cache_load(const noel_kv_cache_t *cache, int layer, int pos, float *k_out, float *v_out);

#ifdef __cplusplus
}
#endif

#endif /* NOEL_KV_CACHE_H */
