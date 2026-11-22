#include "kv_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
