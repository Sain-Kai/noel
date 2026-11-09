#include "simd_vnni.h"
#include "hardware.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Fast horizontal sum of __m256 */
static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 v128  = _mm_add_ps(vlow, vhigh);
    __m128 v64   = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
    __m128 v32   = _mm_add_ss(v64, _mm_shuffle_ps(v64, v64, 0x55));
    return _mm_cvtss_f32(v32);
}

void aether_rmsnorm(float *out, const float *x, const float *weight, int size, float eps) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        sum_vec = _mm256_fmadd_ps(vx, vx, sum_vec);
    }
    float sum_sq = hsum256_ps(sum_vec);
    for (; i < size; i++) {
        sum_sq += x[i] * x[i];
    }

    float scale = 1.0f / sqrtf((sum_sq / (float)size) + eps);
    __m256 vscale = _mm256_set1_ps(scale);

    i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 vout = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vw);
        _mm256_storeu_ps(out + i, vout);
    }
    for (; i < size; i++) {
        out[i] = x[i] * scale * weight[i];
    }
}

void aether_rope(float *vec, int n_heads, int head_dim, int pos, float rope_theta) {
    for (int h = 0; h < n_heads; h++) {
        float *head_vec = vec + h * head_dim;
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
            float angle = (float)pos * freq;
            float cos_th = cosf(angle);
            float sin_th = sinf(angle);

            float v0 = head_vec[2 * i];
            float v1 = head_vec[2 * i + 1];

            head_vec[2 * i]     = v0 * cos_th - v1 * sin_th;
            head_vec[2 * i + 1] = v0 * sin_th + v1 * cos_th;
        }
    }
}

static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void aether_swiglu(float *out, const float *gate, const float *up, int size) {
    int i = 0;
    for (; i <= size - 8; i += 8) {
        for (int k = 0; k < 8; k++) {
            float g = gate[i + k];
            float silu_g = g * fast_sigmoid(g);
            out[i + k] = silu_g * up[i + k];
        }
    }
    for (; i < size; i++) {
        float g = gate[i];
        out[i] = (g * fast_sigmoid(g)) * up[i];
    }
}

void aether_softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

/* AVX2 / AVX-VNNI accelerated Q4_0 Matrix-Vector GEMV: y = W * x */
void aether_gemv_q4_0(float *y, const block_q4_0 *W, const float *x, int rows, int cols, int num_threads) {
    const int nb = cols / AETHER_BLOCK_SIZE;

    #pragma omp parallel for schedule(static) if(num_threads > 1) num_threads(num_threads)
    for (int r = 0; r < rows; r++) {
        const block_q4_0 *row_blocks = W + r * nb;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        const __m128i mask_low = _mm_set1_epi8(0x0F);
        const __m256i offset_8 = _mm256_set1_epi32(8);

        for (int b = 0; b < nb; b++) {
            const block_q4_0 *blk = &row_blocks[b];
            const float *xb = x + b * AETHER_BLOCK_SIZE;
            const __m256 vscale = _mm256_set1_ps(blk->scale);

            /* Load 16 bytes (32 4-bit nibbles) */
            __m128i raw = _mm_loadu_si128((const __m128i*)blk->qs);

            /* Low nibbles: elements 0..15 */
            __m128i low_nibbles = _mm_and_si128(raw, mask_low);
            /* High nibbles: elements 16..31 */
            __m128i high_nibbles = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_low);

            /* Unpack first 8 values (low 8 of low_nibbles) to 32-bit int, subtract 8, multiply */
            __m256i i32_0 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(low_nibbles), offset_8);
            __m256 f32_0 = _mm256_cvtepi32_ps(i32_0);
            __m256 vx0 = _mm256_loadu_ps(xb);
            acc0 = _mm256_fmadd_ps(_mm256_mul_ps(f32_0, vscale), vx0, acc0);

            /* Next 8 values (high 8 of low_nibbles) */
            __m128i low_hi8 = _mm_srli_si128(low_nibbles, 8);
            __m256i i32_1 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(low_hi8), offset_8);
            __m256 f32_1 = _mm256_cvtepi32_ps(i32_1);
            __m256 vx1 = _mm256_loadu_ps(xb + 8);
            acc1 = _mm256_fmadd_ps(_mm256_mul_ps(f32_1, vscale), vx1, acc1);

            /* Next 8 values (low 8 of high_nibbles) */
            __m256i i32_2 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(high_nibbles), offset_8);
            __m256 f32_2 = _mm256_cvtepi32_ps(i32_2);
            __m256 vx2 = _mm256_loadu_ps(xb + 16);
            acc0 = _mm256_fmadd_ps(_mm256_mul_ps(f32_2, vscale), vx2, acc0);

            /* Next 8 values (high 8 of high_nibbles) */
            __m128i high_hi8 = _mm_srli_si128(high_nibbles, 8);
            __m256i i32_3 = _mm256_sub_epi32(_mm256_cvtepu8_epi32(high_hi8), offset_8);
            __m256 f32_3 = _mm256_cvtepi32_ps(i32_3);
            __m256 vx3 = _mm256_loadu_ps(xb + 24);
            acc1 = _mm256_fmadd_ps(_mm256_mul_ps(f32_3, vscale), vx3, acc1);
        }

        y[r] = hsum256_ps(_mm256_add_ps(acc0, acc1));
    }
}

/* AVX-VNNI accelerated Q8_0 Matrix-Vector GEMV using vpdpbusd */
void aether_gemv_q8_0(float *y, const block_q8_0 *W, const float *x, int rows, int cols, int num_threads) {
    const int nb = cols / AETHER_BLOCK_SIZE;

    /* Pre-quantize input vector x to uint8 with block scaling for AVX-VNNI */
    /* Dynamically quantize x per 32-element block */
    float *x_scales = (float*)malloc(nb * sizeof(float));
    uint8_t *x_quant = (uint8_t*)malloc(cols * sizeof(uint8_t));

    for (int b = 0; b < nb; b++) {
        const float *xb = x + b * AETHER_BLOCK_SIZE;
        float amax = 0.0f;
        for (int i = 0; i < AETHER_BLOCK_SIZE; i++) {
            float v = fabsf(xb[i]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        x_scales[b] = d;
        float id = d > 0.0f ? 1.0f / d : 0.0f;
        for (int i = 0; i < AETHER_BLOCK_SIZE; i++) {
            int q = (int)roundf(xb[i] * id) + 128; /* Offset by 128 to make unsigned for dpbusd */
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            x_quant[b * AETHER_BLOCK_SIZE + i] = (uint8_t)q;
        }
    }

    #pragma omp parallel for schedule(static) if(num_threads > 1) num_threads(num_threads)
    for (int r = 0; r < rows; r++) {
        const block_q8_0 *row_blocks = W + r * nb;
        float row_sum = 0.0f;

        for (int b = 0; b < nb; b++) {
            const block_q8_0 *blk = &row_blocks[b];
            const uint8_t *xq = x_quant + b * AETHER_BLOCK_SIZE;

            /* Load 32 unsigned bytes from activation */
            __m256i u_act = _mm256_loadu_si256((const __m256i*)xq);
            /* Load 32 signed bytes from quantized weight */
            __m256i s_wt  = _mm256_loadu_si256((const __m256i*)blk->qs);

            /* AVX-VNNI: computes sum of 4-byte dot products in 32-bit accumulators */
            __m256i dot_acc = _mm256_setzero_si256();
            dot_acc = _mm256_dpbusd_epi32(dot_acc, u_act, s_wt);

            /* Sum the 8 int32 lanes */
            int32_t temp[8];
            _mm256_storeu_si256((__m256i*)temp, dot_acc);
            int32_t raw_dot = 0;
            for (int k = 0; k < 8; k++) raw_dot += temp[k];

            /* Compensate for +128 offset: dot = sum((u - 128) * s) = raw_dot - 128 * sum(s) */
            int32_t sum_s = 0;
            for (int k = 0; k < 32; k++) sum_s += blk->qs[k];

            int32_t true_int_dot = raw_dot - 128 * sum_s;
            row_sum += (float)true_int_dot * (blk->scale * x_scales[b]);
        }
        y[r] = row_sum;
    }

    free(x_scales);
    free(x_quant);
}

/* PowerInfer Sparse GEMV: Only multiplies columns for active neurons */
void aether_sparse_gemv_q4_0(float *y, const block_q4_0 *W, const float *x,
                             const int32_t *active_indices, int num_active,
                             int rows, int cols) {
    memset(y, 0, rows * sizeof(float));
    const int nb = cols / AETHER_BLOCK_SIZE;

    for (int k = 0; k < num_active; k++) {
        int idx = active_indices[k];
        int block_idx = idx / AETHER_BLOCK_SIZE;
        int in_block  = idx % AETHER_BLOCK_SIZE;
        float act_val = x[idx];

        if (fabsf(act_val) < 1e-6f) continue;

        for (int r = 0; r < rows; r++) {
            const block_q4_0 *blk = &W[r * nb + block_idx];
            uint8_t byte_val = blk->qs[in_block / 2];
            int8_t nibble = (in_block % 2 == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
            int8_t weight_int = nibble - 8;
            float weight = (float)weight_int * blk->scale;
            y[r] += weight * act_val;
        }
    }
}
