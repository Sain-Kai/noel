#ifndef AETHER_SIMD_VNNI_H
#define AETHER_SIMD_VNNI_H

#include "aether.h"
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Vectorized RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight */
void aether_rmsnorm(float *out, const float *x, const float *weight, int size, float eps);

/* Vectorized Rotary Positional Embedding (RoPE) */
void aether_rope(float *vec, int n_heads, int head_dim, int pos, float rope_theta);

/* Vectorized SwiGLU activation: out = (gate / (1.0 + exp(-gate))) * up */
void aether_swiglu(float *out, const float *gate, const float *up, int size);

/* Numerically stable vectorized Softmax */
void aether_softmax(float *x, int size);

/* AVX-VNNI accelerated Q8_0 Matrix-Vector multiplication (y = W * x) */
void aether_gemv_q8_0(float *y, const block_q8_0 *W, const float *x, int rows, int cols, int num_threads);

/* AVX2/AVX-VNNI accelerated Q4_0 Matrix-Vector multiplication (y = W * x) */
void aether_gemv_q4_0(float *y, const block_q4_0 *W, const float *x, int rows, int cols, int num_threads);

/* Sparse GEMV: Only multiplies active column/neuron indices (PowerInfer methodology) */
void aether_sparse_gemv_q4_0(float *y, const block_q4_0 *W, const float *x,
                             const int32_t *active_indices, int num_active,
                             int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_SIMD_VNNI_H */
