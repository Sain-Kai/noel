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

/* ------------------------------------------------------------- */
/* Block-Quantized Q8_0 and Q4_0 Codecs                          */
/* ------------------------------------------------------------- */

void noel_quantize_row_q8_0(block_q8_0 *dst, const float *src, int n) {
    if (!dst || !src || n <= 0) return;
    int nb = (n + NOEL_BLOCK_SIZE - 1) / NOEL_BLOCK_SIZE;

    for (int b = 0; b < nb; b++) {
        int offset = b * NOEL_BLOCK_SIZE;
        int rem = n - offset;
        int count = rem < NOEL_BLOCK_SIZE ? rem : NOEL_BLOCK_SIZE;

        float max_abs = 0.0f;
        for (int i = 0; i < count; i++) {
            float a = fabsf(src[offset + i]);
            if (a > max_abs) max_abs = a;
        }

        float scale = max_abs / 127.0f;
        float inv_scale = (scale > 1e-12f) ? (1.0f / scale) : 0.0f;
        dst[b].scale = scale;

        for (int i = 0; i < count; i++) {
            float v = src[offset + i] * inv_scale;
            int q = (int)roundf(v);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            dst[b].qs[i] = (int8_t)q;
        }
        for (int i = count; i < NOEL_BLOCK_SIZE; i++) {
            dst[b].qs[i] = 0;
        }
    }
}

void noel_dequantize_row_q8_0(float *dst, const block_q8_0 *src, int n) {
    if (!dst || !src || n <= 0) return;
    int nb = (n + NOEL_BLOCK_SIZE - 1) / NOEL_BLOCK_SIZE;

    for (int b = 0; b < nb; b++) {
        int offset = b * NOEL_BLOCK_SIZE;
        int rem = n - offset;
        int count = rem < NOEL_BLOCK_SIZE ? rem : NOEL_BLOCK_SIZE;
        float scale = src[b].scale;

        for (int i = 0; i < count; i++) {
            dst[offset + i] = (float)src[b].qs[i] * scale;
        }
    }
}

void noel_quantize_row_q4_0(block_q4_0 *dst, const float *src, int n) {
    if (!dst || !src || n <= 0) return;
    int nb = (n + NOEL_BLOCK_SIZE - 1) / NOEL_BLOCK_SIZE;

    for (int b = 0; b < nb; b++) {
        int offset = b * NOEL_BLOCK_SIZE;
        int rem = n - offset;
        int count = rem < NOEL_BLOCK_SIZE ? rem : NOEL_BLOCK_SIZE;

        float max_abs = 0.0f;
        for (int i = 0; i < count; i++) {
            float a = fabsf(src[offset + i]);
            if (a > max_abs) max_abs = a;
        }

        float scale = max_abs / -8.0f;
        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
        dst[b].scale = scale;

        for (int i = 0; i < 16; i++) {
            float x0 = (i < count) ? src[offset + i] : 0.0f;
            float x1 = (i + 16 < count) ? src[offset + i + 16] : 0.0f;

            int q0 = (int)roundf(x0 * inv_scale) + 8;
            int q1 = (int)roundf(x1 * inv_scale) + 8;
            if (q0 < 0) q0 = 0;
            if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0;
            if (q1 > 15) q1 = 15;

            dst[b].qs[i] = (uint8_t)((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        }
    }
}

void noel_dequantize_row_q4_0(float *dst, const block_q4_0 *src, int n) {
    if (!dst || !src || n <= 0) return;
    int nb = (n + NOEL_BLOCK_SIZE - 1) / NOEL_BLOCK_SIZE;

    for (int b = 0; b < nb; b++) {
        int offset = b * NOEL_BLOCK_SIZE;
        int rem = n - offset;
        int count = rem < NOEL_BLOCK_SIZE ? rem : NOEL_BLOCK_SIZE;
        float scale = src[b].scale;

        for (int i = 0; i < 16; i++) {
            uint8_t v = src[b].qs[i];
            int q0 = (v & 0x0F) - 8;
            int q1 = ((v >> 4) & 0x0F) - 8;

            if (i < count) dst[offset + i] = (float)q0 * scale;
            if (i + 16 < count) dst[offset + i + 16] = (float)q1 * scale;
        }
    }
}

void noel_kv_cache_store(noel_kv_cache_t *cache, int layer, int pos, const float *k, const float *v) {
    if (!cache || !k || !v || layer < 0 || layer >= cache->n_layers || pos < 0 || pos >= cache->seq_len) return;

    size_t row_idx = (size_t)layer * cache->seq_len + pos;
    size_t byte_offset = row_idx * cache->row_bytes;
    int kv_dim = cache->kv_dim;

    switch (cache->dtype) {
        case NOEL_KV_FP32: {
            float *k_dst = (float *)((uint8_t *)cache->k_data + byte_offset);
            float *v_dst = (float *)((uint8_t *)cache->v_data + byte_offset);
            memcpy(k_dst, k, kv_dim * sizeof(float));
            memcpy(v_dst, v, kv_dim * sizeof(float));
            break;
        }
        case NOEL_KV_FP8_E4M3: {
            uint8_t *k_dst = (uint8_t *)cache->k_data + byte_offset;
            uint8_t *v_dst = (uint8_t *)cache->v_data + byte_offset;
            float *k_sc = cache->k_scales + row_idx * cache->n_kv_heads;
            float *v_sc = cache->v_scales + row_idx * cache->n_kv_heads;
            noel_quantize_row_fp8_e4m3(k_dst, k, kv_dim, k_sc);
            noel_quantize_row_fp8_e4m3(v_dst, v, kv_dim, v_sc);
            break;
        }
        case NOEL_KV_FP8_E5M2: {
            uint8_t *k_dst = (uint8_t *)cache->k_data + byte_offset;
            uint8_t *v_dst = (uint8_t *)cache->v_data + byte_offset;
            float *k_sc = cache->k_scales + row_idx * cache->n_kv_heads;
            float *v_sc = cache->v_scales + row_idx * cache->n_kv_heads;
            noel_quantize_row_fp8_e5m2(k_dst, k, kv_dim, k_sc);
            noel_quantize_row_fp8_e5m2(v_dst, v, kv_dim, v_sc);
            break;
        }
        case NOEL_KV_Q8_0: {
            block_q8_0 *k_dst = (block_q8_0 *)((uint8_t *)cache->k_data + byte_offset);
            block_q8_0 *v_dst = (block_q8_0 *)((uint8_t *)cache->v_data + byte_offset);
            noel_quantize_row_q8_0(k_dst, k, kv_dim);
            noel_quantize_row_q8_0(v_dst, v, kv_dim);
            break;
        }
        case NOEL_KV_Q4_0: {
            block_q4_0 *k_dst = (block_q4_0 *)((uint8_t *)cache->k_data + byte_offset);
            block_q4_0 *v_dst = (block_q4_0 *)((uint8_t *)cache->v_data + byte_offset);
            noel_quantize_row_q4_0(k_dst, k, kv_dim);
            noel_quantize_row_q4_0(v_dst, v, kv_dim);
            break;
        }
    }
}

void noel_kv_cache_load(const noel_kv_cache_t *cache, int layer, int pos, float *k_out, float *v_out) {
    if (!cache || layer < 0 || layer >= cache->n_layers || pos < 0 || pos >= cache->seq_len) return;

    size_t row_idx = (size_t)layer * cache->seq_len + pos;
    size_t byte_offset = row_idx * cache->row_bytes;
    int kv_dim = cache->kv_dim;

    switch (cache->dtype) {
        case NOEL_KV_FP32: {
            const float *k_src = (const float *)((const uint8_t *)cache->k_data + byte_offset);
            const float *v_src = (const float *)((const uint8_t *)cache->v_data + byte_offset);
            if (k_out) memcpy(k_out, k_src, kv_dim * sizeof(float));
            if (v_out) memcpy(v_out, v_src, kv_dim * sizeof(float));
            break;
        }
        case NOEL_KV_FP8_E4M3: {
            const uint8_t *k_src = (const uint8_t *)cache->k_data + byte_offset;
            const uint8_t *v_src = (const uint8_t *)cache->v_data + byte_offset;
            float k_sc = cache->k_scales ? cache->k_scales[row_idx * cache->n_kv_heads] : 1.0f;
            float v_sc = cache->v_scales ? cache->v_scales[row_idx * cache->n_kv_heads] : 1.0f;
            if (k_out) noel_dequantize_row_fp8_e4m3(k_out, k_src, kv_dim, k_sc);
            if (v_out) noel_dequantize_row_fp8_e4m3(v_out, v_src, kv_dim, v_sc);
            break;
        }
        case NOEL_KV_FP8_E5M2: {
            const uint8_t *k_src = (const uint8_t *)cache->k_data + byte_offset;
            const uint8_t *v_src = (const uint8_t *)cache->v_data + byte_offset;
            float k_sc = cache->k_scales ? cache->k_scales[row_idx * cache->n_kv_heads] : 1.0f;
            float v_sc = cache->v_scales ? cache->v_scales[row_idx * cache->n_kv_heads] : 1.0f;
            if (k_out) noel_dequantize_row_fp8_e5m2(k_out, k_src, kv_dim, k_sc);
            if (v_out) noel_dequantize_row_fp8_e5m2(v_out, v_src, kv_dim, v_sc);
            break;
        }
        case NOEL_KV_Q8_0: {
            const block_q8_0 *k_src = (const block_q8_0 *)((const uint8_t *)cache->k_data + byte_offset);
            const block_q8_0 *v_src = (const block_q8_0 *)((const uint8_t *)cache->v_data + byte_offset);
            if (k_out) noel_dequantize_row_q8_0(k_out, k_src, kv_dim);
            if (v_out) noel_dequantize_row_q8_0(v_out, v_src, kv_dim);
            break;
        }
        case NOEL_KV_Q4_0: {
            const block_q4_0 *k_src = (const block_q4_0 *)((const uint8_t *)cache->k_data + byte_offset);
            const block_q4_0 *v_src = (const block_q4_0 *)((const uint8_t *)cache->v_data + byte_offset);
            if (k_out) noel_dequantize_row_q4_0(k_out, k_src, kv_dim);
            if (v_out) noel_dequantize_row_q4_0(v_out, v_src, kv_dim);
            break;
        }
    }
}

int noel_kv_cache_apply_attention_sink(noel_kv_cache_t *cache, int current_len, int n_sink, int n_recent) {
    if (!cache || !cache->k_data || !cache->v_data) return current_len;
    if (n_sink < 0) n_sink = 0;
    if (n_recent < 0) n_recent = 0;
    if (current_len <= n_sink + n_recent) return current_len;
    if (n_sink + n_recent >= cache->seq_len) return current_len;

    int new_len = n_sink + n_recent;
    int discard_count = current_len - new_len;
    if (discard_count <= 0) return current_len;

    size_t row_bytes = cache->row_bytes;
    int n_heads = cache->n_kv_heads;

    for (int l = 0; l < cache->n_layers; l++) {
        size_t layer_base = (size_t)l * cache->seq_len;

        /* Move the recent tokens right after the sink tokens */
        for (int j = 0; j < n_recent; j++) {
            int src_pos = (current_len - n_recent) + j;
            int dst_pos = n_sink + j;

            size_t src_idx = layer_base + src_pos;
            size_t dst_idx = layer_base + dst_pos;

            uint8_t *k_src = (uint8_t *)cache->k_data + src_idx * row_bytes;
            uint8_t *k_dst = (uint8_t *)cache->k_data + dst_idx * row_bytes;
            uint8_t *v_src = (uint8_t *)cache->v_data + src_idx * row_bytes;
            uint8_t *v_dst = (uint8_t *)cache->v_data + dst_idx * row_bytes;

            memmove(k_dst, k_src, row_bytes);
            memmove(v_dst, v_src, row_bytes);

            if (cache->k_scales && cache->v_scales) {
                float *k_sc_src = cache->k_scales + src_idx * n_heads;
                float *k_sc_dst = cache->k_scales + dst_idx * n_heads;
                float *v_sc_src = cache->v_scales + src_idx * n_heads;
                float *v_sc_dst = cache->v_scales + dst_idx * n_heads;

                memmove(k_sc_dst, k_sc_src, n_heads * sizeof(float));
                memmove(v_sc_dst, v_sc_src, n_heads * sizeof(float));
            }
        }

        /* Clear out vacated trailing slots */
        for (int p = new_len; p < current_len; p++) {
            size_t p_idx = layer_base + p;
            uint8_t *k_p = (uint8_t *)cache->k_data + p_idx * row_bytes;
            uint8_t *v_p = (uint8_t *)cache->v_data + p_idx * row_bytes;
            memset(k_p, 0, row_bytes);
            memset(v_p, 0, row_bytes);

            if (cache->k_scales && cache->v_scales) {
                float *k_sc_p = cache->k_scales + p_idx * n_heads;
                float *v_sc_p = cache->v_scales + p_idx * n_heads;
                memset(k_sc_p, 0, n_heads * sizeof(float));
                memset(v_sc_p, 0, n_heads * sizeof(float));
            }
        }
    }

    return new_len;
}

void noel_kv_cache_rollback(noel_kv_cache_t *cache, int target_pos, int current_pos) {
    if (!cache || !cache->k_data || !cache->v_data) return;
    if (target_pos < 0) target_pos = 0;
    if (target_pos >= current_pos) return;
    if (current_pos > cache->seq_len) current_pos = cache->seq_len;

    size_t row_bytes = cache->row_bytes;
    int n_heads = cache->n_kv_heads;

    for (int l = 0; l < cache->n_layers; l++) {
        size_t layer_base = (size_t)l * cache->seq_len;

        for (int p = target_pos; p < current_pos; p++) {
            size_t p_idx = layer_base + p;
            uint8_t *k_p = (uint8_t *)cache->k_data + p_idx * row_bytes;
            uint8_t *v_p = (uint8_t *)cache->v_data + p_idx * row_bytes;
            memset(k_p, 0, row_bytes);
            memset(v_p, 0, row_bytes);

            if (cache->k_scales && cache->v_scales) {
                float *k_sc_p = cache->k_scales + p_idx * n_heads;
                float *v_sc_p = cache->v_scales + p_idx * n_heads;
                memset(k_sc_p, 0, n_heads * sizeof(float));
                memset(v_sc_p, 0, n_heads * sizeof(float));
            }
        }
    }
}
