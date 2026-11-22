#include "kv_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

const char *noel_kv_dtype_name(noel_kv_dtype_t dtype) {
    switch (dtype) {
        case NOEL_KV_FP32:     return "FP32";
        case NOEL_KV_FP8_E4M3: return "FP8_E4M3";
        case NOEL_KV_FP8_E5M2: return "FP8_E5M2";
        case NOEL_KV_Q8_0:     return "Q8_0";
        case NOEL_KV_Q4_0:     return "Q4_0";
        default:               return "Unknown";
    }
}

size_t noel_kv_dtype_row_bytes(noel_kv_dtype_t dtype, int kv_dim) {
    if (kv_dim <= 0) return 0;
    switch (dtype) {
        case NOEL_KV_FP32:
            return (size_t)kv_dim * sizeof(float);
        case NOEL_KV_FP8_E4M3:
        case NOEL_KV_FP8_E5M2:
            return (size_t)kv_dim * sizeof(uint8_t);
        case NOEL_KV_Q8_0: {
            int nb = (kv_dim + NOEL_BLOCK_SIZE - 1) / NOEL_BLOCK_SIZE;
            return (size_t)nb * sizeof(block_q8_0);
        }
        case NOEL_KV_Q4_0: {
            int nb = (kv_dim + NOEL_BLOCK_SIZE - 1) / NOEL_BLOCK_SIZE;
            return (size_t)nb * sizeof(block_q4_0);
        }
        default:
            return (size_t)kv_dim * sizeof(float);
    }
}

int noel_kv_cache_init(noel_kv_cache_t *cache,
                       noel_kv_dtype_t dtype,
                       int n_layers,
                       int seq_len,
                       int n_kv_heads,
                       int head_dim) {
    if (!cache) return -1;
    memset(cache, 0, sizeof(*cache));

    if (n_layers <= 0 || seq_len <= 0 || n_kv_heads <= 0 || head_dim <= 0) {
        return -1;
    }

    cache->dtype = dtype;
    cache->n_layers = n_layers;
    cache->seq_len = seq_len;
    cache->n_kv_heads = n_kv_heads;
    cache->head_dim = head_dim;
    cache->kv_dim = n_kv_heads * head_dim;

    cache->row_bytes = noel_kv_dtype_row_bytes(dtype, cache->kv_dim);
    size_t total_elements_per_cache = (size_t)n_layers * seq_len;
    cache->total_bytes = total_elements_per_cache * cache->row_bytes * 2; /* keys + values */
    cache->fp32_equivalent_bytes = total_elements_per_cache * (size_t)cache->kv_dim * sizeof(float) * 2;

    int ret_k = posix_memalign(&cache->k_data, NOEL_ALIGNMENT, total_elements_per_cache * cache->row_bytes);
    int ret_v = posix_memalign(&cache->v_data, NOEL_ALIGNMENT, total_elements_per_cache * cache->row_bytes);
    if (ret_k != 0 || ret_v != 0 || !cache->k_data || !cache->v_data) {
        noel_kv_cache_free(cache);
        return -1;
    }

    memset(cache->k_data, 0, total_elements_per_cache * cache->row_bytes);
    memset(cache->v_data, 0, total_elements_per_cache * cache->row_bytes);

    /* Allocate scale buffers for FP8 if needed (per token per head) */
    if (dtype == NOEL_KV_FP8_E4M3 || dtype == NOEL_KV_FP8_E5M2) {
        size_t n_scales = total_elements_per_cache * n_kv_heads;
        cache->k_scales = (float *)calloc(n_scales, sizeof(float));
        cache->v_scales = (float *)calloc(n_scales, sizeof(float));
        if (!cache->k_scales || !cache->v_scales) {
            noel_kv_cache_free(cache);
            return -1;
        }
    }

    return 0;
}

void noel_kv_cache_free(noel_kv_cache_t *cache) {
    if (!cache) return;
    if (cache->k_data) {
        free(cache->k_data);
        cache->k_data = NULL;
    }
    if (cache->v_data) {
        free(cache->v_data);
        cache->v_data = NULL;
    }
    if (cache->k_scales) {
        free(cache->k_scales);
        cache->k_scales = NULL;
    }
    if (cache->v_scales) {
        free(cache->v_scales);
        cache->v_scales = NULL;
    }
    memset(cache, 0, sizeof(*cache));
}

void noel_kv_cache_reset(noel_kv_cache_t *cache) {
    if (!cache) return;
    size_t total_elements_per_cache = (size_t)cache->n_layers * cache->seq_len;
    if (cache->k_data) {
        memset(cache->k_data, 0, total_elements_per_cache * cache->row_bytes);
    }
    if (cache->v_data) {
        memset(cache->v_data, 0, total_elements_per_cache * cache->row_bytes);
    }
    if (cache->k_scales) {
        memset(cache->k_scales, 0, total_elements_per_cache * cache->n_kv_heads * sizeof(float));
    }
    if (cache->v_scales) {
        memset(cache->v_scales, 0, total_elements_per_cache * cache->n_kv_heads * sizeof(float));
    }
}

double noel_kv_cache_compression_ratio(const noel_kv_cache_t *cache) {
    if (!cache || cache->total_bytes == 0) return 1.0;
    return (double)cache->fp32_equivalent_bytes / (double)cache->total_bytes;
}

/* ------------------------------------------------------------- */
/* FP8 (E4M3 / E5M2) Emulation Codecs                            */
/* ------------------------------------------------------------- */

float noel_fp8_e4m3_to_f32(uint8_t val) {
    uint32_t sign = (val >> 7) & 1;
    uint32_t exp  = (val >> 3) & 0x0F;
    uint32_t mant = val & 0x07;

    float result;
    if (exp == 0) {
        /* Subnormal */
        result = (float)mant / 8.0f * (1.0f / 64.0f);
    } else if (exp == 15 && mant == 7) {
        result = 448.0f; /* Max finite clamp */
    } else {
        /* Normal */
        int e = (int)exp - 7;
        float factor = ldexpf(1.0f, e);
        result = (1.0f + (float)mant / 8.0f) * factor;
    }
    return sign ? -result : result;
}

uint8_t noel_f32_to_fp8_e4m3(float x) {
    uint8_t sign = 0;
    if (x < 0.0f) {
        sign = 0x80;
        x = -x;
    }
    if (x >= 448.0f) return sign | 0x7E; /* Max finite: exp=15, mant=6 */
    if (x < (1.0f / 128.0f)) return sign; /* Underflow to 0 */

    int exp;
    float frac = frexpf(x, &exp); /* x = frac * 2^exp, frac in [0.5, 1.0) */
    /* Normalize to 1.mantissa: x = (1 + m/8) * 2^(e4 - 7) */
    int biased_exp = exp + 6; /* since frexp gives exp relative to [0.5, 1.0) */

    if (biased_exp <= 0) {
        /* Subnormal */
        int mant = (int)roundf(x * 64.0f * 8.0f);
        if (mant > 7) mant = 7;
        return sign | (uint8_t)mant;
    } else if (biased_exp >= 15) {
        return sign | 0x7E;
    }

    float normalized = frac * 2.0f; /* in [1.0, 2.0) */
    int mant = (int)roundf((normalized - 1.0f) * 8.0f);
    if (mant >= 8) {
        mant = 0;
        biased_exp++;
        if (biased_exp >= 15) return sign | 0x7E;
    }
    return sign | ((uint8_t)biased_exp << 3) | ((uint8_t)mant & 0x07);
}

float noel_fp8_e5m2_to_f32(uint8_t val) {
    uint32_t sign = (val >> 7) & 1;
    uint32_t exp  = (val >> 2) & 0x1F;
    uint32_t mant = val & 0x03;

    float result;
    if (exp == 0) {
        result = (float)mant / 4.0f * (1.0f / 16384.0f);
    } else if (exp == 31) {
        result = 57344.0f; /* Max finite */
    } else {
        int e = (int)exp - 15;
        float factor = ldexpf(1.0f, e);
        result = (1.0f + (float)mant / 4.0f) * factor;
    }
    return sign ? -result : result;
}

uint8_t noel_f32_to_fp8_e5m2(float x) {
    uint8_t sign = 0;
    if (x < 0.0f) {
        sign = 0x80;
        x = -x;
    }
    if (x >= 57344.0f) return sign | 0x7B;
    if (x < (1.0f / 32768.0f)) return sign;

    int exp;
    float frac = frexpf(x, &exp);
    int biased_exp = exp + 14;

    if (biased_exp <= 0) {
        int mant = (int)roundf(x * 16384.0f * 4.0f);
        if (mant > 3) mant = 3;
        return sign | (uint8_t)mant;
    } else if (biased_exp >= 31) {
        return sign | 0x7B;
    }

    float normalized = frac * 2.0f;
    int mant = (int)roundf((normalized - 1.0f) * 4.0f);
    if (mant >= 4) {
        mant = 0;
        biased_exp++;
        if (biased_exp >= 31) return sign | 0x7B;
    }
    return sign | ((uint8_t)biased_exp << 2) | ((uint8_t)mant & 0x03);
}

void noel_quantize_row_fp8_e4m3(uint8_t *dst, const float *src, int n, float *out_scale) {
    if (!dst || !src || n <= 0) return;
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > max_abs) max_abs = a;
    }

    float scale = (max_abs > 1e-12f) ? (max_abs / 440.0f) : 1.0f;
    if (out_scale) *out_scale = scale;
    float inv_scale = 1.0f / scale;

    for (int i = 0; i < n; i++) {
        dst[i] = noel_f32_to_fp8_e4m3(src[i] * inv_scale);
    }
}

void noel_dequantize_row_fp8_e4m3(float *dst, const uint8_t *src, int n, float scale) {
    if (!dst || !src || n <= 0) return;
    for (int i = 0; i < n; i++) {
        dst[i] = noel_fp8_e4m3_to_f32(src[i]) * scale;
    }
}

void noel_quantize_row_fp8_e5m2(uint8_t *dst, const float *src, int n, float *out_scale) {
    if (!dst || !src || n <= 0) return;
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > max_abs) max_abs = a;
    }

    float scale = (max_abs > 1e-12f) ? (max_abs / 57000.0f) : 1.0f;
    if (out_scale) *out_scale = scale;
    float inv_scale = 1.0f / scale;

    for (int i = 0; i < n; i++) {
        dst[i] = noel_f32_to_fp8_e5m2(src[i] * inv_scale);
    }
}

void noel_dequantize_row_fp8_e5m2(float *dst, const uint8_t *src, int n, float scale) {
    if (!dst || !src || n <= 0) return;
    for (int i = 0; i < n; i++) {
        dst[i] = noel_fp8_e5m2_to_f32(src[i]) * scale;
    }
}
