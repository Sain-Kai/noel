#include "dequant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

int noel_ggml_block_size(noel_ggml_type_t type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return 1;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
            return 32;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_XS:
            return 256;
        default:
            return 32;
    }
}

size_t noel_ggml_type_size(noel_ggml_type_t type) {
    switch (type) {
        case GGML_TYPE_F32:     return 4;
        case GGML_TYPE_F16:     return 2;
        case GGML_TYPE_BF16:    return 2;
        case GGML_TYPE_Q4_0:    return 18;
        case GGML_TYPE_Q4_1:    return 20;
        case GGML_TYPE_Q5_0:    return 22;
        case GGML_TYPE_Q5_1:    return 24;
        case GGML_TYPE_Q8_0:    return 34;
        case GGML_TYPE_IQ4_NL:  return 18;
        case GGML_TYPE_Q2_K:    return 84;
        case GGML_TYPE_Q3_K:    return 110;
        case GGML_TYPE_Q4_K:    return 144;
        case GGML_TYPE_Q5_K:    return 176;
        case GGML_TYPE_Q6_K:    return 210;
        case GGML_TYPE_Q8_K:    return 292;
        case GGML_TYPE_IQ2_XXS: return 66;
        case GGML_TYPE_IQ2_XS:  return 74;
        case GGML_TYPE_IQ3_XXS: return 98;
        case GGML_TYPE_IQ4_XS:  return 136;
        default:                return 18;
    }
}

size_t noel_ggml_row_bytes(noel_ggml_type_t type, int64_t n_elems) {
    int bs = noel_ggml_block_size(type);
    size_t ts = noel_ggml_type_size(type);
    int64_t n_blocks = (n_elems + bs - 1) / bs;
    return (size_t)n_blocks * ts;
}

bool noel_dequant_init(void) {
    return true;
}

/* Native AVX2 Vectorized Dequantizer: BF16 */
static void dequant_row_bf16_native(const void *src, float *dst, int64_t n) {
    const uint16_t *s = (const uint16_t*)src;
    int64_t i = 0;
    for (; i <= n - 8; i += 8) {
        __m128i raw = _mm_loadu_si128((const __m128i*)(s + i));
        __m256i raw32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(raw32));
    }
    for (; i < n; i++) {
        dst[i] = noel_bf16_to_fp32(s[i]);
    }
}

/* Native AVX2 Vectorized Dequantizer: F16 */
static void dequant_row_f16_native(const void *src, float *dst, int64_t n) {
    const uint16_t *s = (const uint16_t*)src;
    int64_t i = 0;
    for (; i <= n - 8; i += 8) {
        __m128i raw = _mm_loadu_si128((const __m128i*)(s + i));
        __m256 f = _mm256_cvtph_ps(raw);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < n; i++) {
        dst[i] = noel_fp16_to_fp32(s[i]);
    }
}

void noel_dequantize_row(noel_ggml_type_t type, const void *src, float *dst, int64_t n) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst, src, n * sizeof(float));
            break;
        case GGML_TYPE_BF16:
            dequant_row_bf16_native(src, dst, n);
            break;
        case GGML_TYPE_F16:
            dequant_row_f16_native(src, dst, n);
            break;
        default:
            memset(dst, 0, n * sizeof(float));
            break;
    }
}

void noel_gemv_quantized(noel_ggml_type_t type,
                        float *y,
                        const void *W,
                        const float *x,
                        int rows,
                        int cols,
                        int n_threads) {
    (void)type;
    (void)y;
    (void)W;
    (void)x;
    (void)rows;
    (void)cols;
    (void)n_threads;
}
