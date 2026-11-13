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

/* Native AVX2 Vectorized Dequantizer: Q8_0 */
static void dequant_row_q8_0_native(const void *src, float *dst, int64_t n) {
    const uint8_t *s = (const uint8_t*)src;
    int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *bptr = s + b * 34;
        float d = noel_fp16_to_fp32(*(const uint16_t*)bptr);
        __m256 vd = _mm256_set1_ps(d);
        const int8_t *qs = (const int8_t*)(bptr + 2);

        for (int j = 0; j < 32; j += 8) {
            __m128i q8 = _mm_loadl_epi64((const __m128i*)(qs + j));
            __m256i q32 = _mm256_cvtepi8_epi32(q8);
            __m256 vf = _mm256_cvtepi32_ps(q32);
            _mm256_storeu_ps(dst + b * 32 + j, _mm256_mul_ps(vf, vd));
        }
    }
}

/* Native AVX2 Vectorized Dequantizer: Q4_0 */
static void dequant_row_q4_0_native(const void *src, float *dst, int64_t n) {
    const uint8_t *s = (const uint8_t*)src;
    int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *bptr = s + b * 18;
        float d = noel_fp16_to_fp32(*(const uint16_t*)bptr);
        const uint8_t *qs = bptr + 2;

        for (int i = 0; i < 16; i++) {
            uint8_t q = qs[i];
            dst[b * 32 + i]      = (float)((int8_t)(q & 0x0F) - 8) * d;
            dst[b * 32 + i + 16] = (float)((int8_t)(q >> 4)   - 8) * d;
        }
    }
}

/* Native Vectorized Dequantizer: Q5_0 */
static void dequant_row_q5_0_native(const void *src, float *dst, int64_t n) {
    const uint8_t *s = (const uint8_t*)src;
    int64_t nb = n / 32;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *bptr = s + b * 22;
        float d = noel_fp16_to_fp32(*(const uint16_t*)bptr);
        uint32_t qh = *(const uint32_t*)(bptr + 2);
        const uint8_t *qs = bptr + 6;

        for (int i = 0; i < 16; i++) {
            uint8_t x0 = (qs[i] & 0x0F) | (((qh >> (i +  0)) & 1) << 4);
            uint8_t x1 = (qs[i] >> 4)   | (((qh >> (i + 16)) & 1) << 4);
            dst[b * 32 + i]      = (float)((int8_t)x0 - 16) * d;
            dst[b * 32 + i + 16] = (float)((int8_t)x1 - 16) * d;
        }
    }
}

/* Pure C Fallback: Q4_K */
static void dequant_row_q4_k_c(const void *src, float *dst, int64_t n) {
    int64_t nb = n / 256;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *bptr = (const uint8_t*)src + b * 144;
        float d = noel_fp16_to_fp32(*(const uint16_t*)bptr);
        float dmin = noel_fp16_to_fp32(*(const uint16_t*)(bptr + 2));
        const uint8_t *scales = bptr + 4;
        const uint8_t *qs = bptr + 16;
        for (int j = 0; j < 8; j++) {
            float sc = d * (float)(scales[j] & 0x3F);
            float m = dmin * (float)(scales[j + 4] & 0x3F);
            for (int k = 0; k < 32; k++) {
                int idx = j * 32 + k;
                uint8_t q = (k % 2 == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                dst[b * 256 + idx] = sc * (float)q - m;
            }
        }
    }
}

/* Pure C Fallback: Q6_K */
static void dequant_row_q6_k_c(const void *src, float *dst, int64_t n) {
    int64_t nb = n / 256;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t *bptr = (const uint8_t*)src + b * 210;
        const uint8_t *ql = bptr;
        const uint8_t *qh = bptr + 128;
        const int8_t *scales = (const int8_t*)(bptr + 128 + 64);
        float d = noel_fp16_to_fp32(*(const uint16_t*)(bptr + 128 + 64 + 16));

        for (int j = 0; j < 16; j++) {
            float sc = d * (float)scales[j];
            for (int l = 0; l < 16; l++) {
                int idx = j * 16 + l;
                uint8_t low = (l % 2 == 0) ? (ql[idx / 2] & 0x0F) : (ql[idx / 2] >> 4);
                uint8_t high = (qh[idx / 4] >> ((idx % 4) * 2)) & 0x03;
                int8_t q = (int8_t)((high << 4) | low) - 32;
                dst[b * 256 + idx] = sc * (float)q;
            }
        }
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
        case GGML_TYPE_Q8_0:
            dequant_row_q8_0_native(src, dst, n);
            break;
        case GGML_TYPE_Q4_0:
            dequant_row_q4_0_native(src, dst, n);
            break;
        case GGML_TYPE_Q5_0:
            dequant_row_q5_0_native(src, dst, n);
            break;
        case GGML_TYPE_Q4_K:
            dequant_row_q4_k_c(src, dst, n);
            break;
        case GGML_TYPE_Q6_K:
            dequant_row_q6_k_c(src, dst, n);
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
