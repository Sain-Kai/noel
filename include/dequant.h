#ifndef NOEL_DEQUANT_H
#define NOEL_DEQUANT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "noel.h"
#include "gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize dequantization subsystem (discovers hardware SIMD and dynamic acceleration) */
bool noel_dequant_init(void);

/* Get number of elements per block for a given GGML quantization type */
int noel_ggml_block_size(noel_ggml_type_t type);

/* Get byte size of a block for a given GGML quantization type */
size_t noel_ggml_type_size(noel_ggml_type_t type);

/* Calculate total byte size for a row of n elements */
size_t noel_ggml_row_bytes(noel_ggml_type_t type, int64_t n_elems);

/* Core generic dequantization function: dequantizes n elements from src into float array dst */
void noel_dequantize_row(noel_ggml_type_t type, const void *src, float *dst, int64_t n);

/* High-performance multi-threaded GEMV over any quantized matrix format */
void noel_gemv_quantized(noel_ggml_type_t type,
                        float *y,
                        const void *W,
                        const float *x,
                        int rows,
                        int cols,
                        int n_threads);

/* Fast float-16 (IEEE 754 half) conversion utility */
static inline float noel_fp16_to_fp32(uint16_t h) {
    uint32_t w = (uint32_t)h << 16;
    uint32_t sign = w & 0x80000000;
    uint32_t two_w = w + w;
    uint32_t exp = two_w >> 24;
    uint32_t mant = two_w >> 13;
    uint32_t magic = 126 << 23;
    float magic_f;
    __builtin_memcpy(&magic_f, &magic, sizeof(magic_f));

    if (exp == 0) {
        if (mant == 0) {
            float f = 0.0f;
            if (sign) f = -0.0f;
            return f;
        }
        magic = (127 - 15) << 23;
        __builtin_memcpy(&magic_f, &magic, sizeof(magic_f));
        uint32_t u = (127 - 15) << 23 | (mant << 13);
        float f;
        __builtin_memcpy(&f, &u, sizeof(f));
        f -= magic_f;
        if (sign) f = -f;
        return f;
    } else if (exp == 0x3F) {
        uint32_t u = sign | 0x7F800000 | (mant << 13);
        float f;
        __builtin_memcpy(&f, &u, sizeof(f));
        return f;
    }

    uint32_t u = sign | (((exp + (127 - 15)) << 23) + (mant << 13));
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

/* Fast bfloat16 conversion utility */
static inline float noel_bf16_to_fp32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

#ifdef __cplusplus
}
#endif

#endif /* NOEL_DEQUANT_H */
