#include "attention.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

float noel_vec_dot_f32_avx2(const float *a, const float *b, int dim) {
    if (!a || !b || dim <= 0) return 0.0f;

    __m256 sum256 = _mm256_setzero_ps();
    int d = 0;
    for (; d + 7 < dim; d += 8) {
        __m256 va = _mm256_loadu_ps(a + d);
        __m256 vb = _mm256_loadu_ps(b + d);
        sum256 = _mm256_fmadd_ps(va, vb, sum256);
    }

    __m128 lo = _mm256_castps256_ps128(sum256);
    __m128 hi = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float dot = _mm_cvtss_f32(sum128);

    for (; d < dim; d++) {
        dot += a[d] * b[d];
    }
    return dot;
}

static inline void vec_scale_add_f32_avx2(float *dst, float factor, const float *src, int dim) {
    __m256 vfactor = _mm256_set1_ps(factor);
    int d = 0;
    for (; d + 7 < dim; d += 8) {
        __m256 vd = _mm256_loadu_ps(dst + d);
        __m256 vs = _mm256_loadu_ps(src + d);
        vd = _mm256_fmadd_ps(vd, vfactor, vs);
        _mm256_storeu_ps(dst + d, vd);
    }
    for (; d < dim; d++) {
        dst[d] = dst[d] * factor + src[d];
    }
}

static inline void vec_add_weighted_f32_avx2(float *dst, float weight, const float *src, int dim) {
    __m256 vweight = _mm256_set1_ps(weight);
    int d = 0;
    for (; d + 7 < dim; d += 8) {
        __m256 vd = _mm256_loadu_ps(dst + d);
        __m256 vs = _mm256_loadu_ps(src + d);
        vd = _mm256_fmadd_ps(vs, vweight, vd);
        _mm256_storeu_ps(dst + d, vd);
    }
    for (; d < dim; d++) {
        dst[d] += weight * src[d];
    }
}

void noel_gqa_attention(float *out,
                        const float *q,
                        const float *key_cache,
                        const float *val_cache,
                        int n_heads,
                        int n_kv_heads,
                        int head_dim,
                        int seq_len,
                        int pos,
                        int n_threads) {
    if (!out || !q || !key_cache || !val_cache) return;
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || seq_len <= 0 || pos < 0) return;

    int max_t = (pos < seq_len) ? pos : (seq_len - 1);
    int group_size = n_heads / n_kv_heads;
    if (group_size < 1) group_size = 1;
    int kv_dim = n_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    if (n_threads < 1) n_threads = 1;

    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int h = 0; h < n_heads; h++) {
        int kv_head = h / group_size;
        if (kv_head >= n_kv_heads) kv_head = n_kv_heads - 1;

        const float *q_h = q + h * head_dim;
        float *out_h = out + h * head_dim;

        float acc_stack[512];
        float *acc = acc_stack;
        bool heap_allocated = false;
        if (head_dim > 512) {
            acc = (float *)malloc(head_dim * sizeof(float));
            heap_allocated = true;
        }
        memset(acc, 0, head_dim * sizeof(float));

        float max_score = -1e30f;
        float sum_exp = 0.0f;

        for (int t = 0; t <= max_t; t++) {
            const float *k_t = key_cache + (size_t)t * kv_dim + kv_head * head_dim;
            const float *v_t = val_cache + (size_t)t * kv_dim + kv_head * head_dim;

            float score = noel_vec_dot_f32_avx2(q_h, k_t, head_dim) * scale;

            if (score > max_score) {
                float factor = expf(max_score - score);
                max_score = score;
                sum_exp = sum_exp * factor + 1.0f;
                vec_scale_add_f32_avx2(acc, factor, v_t, head_dim);
            } else {
                float weight = expf(score - max_score);
                sum_exp += weight;
                vec_add_weighted_f32_avx2(acc, weight, v_t, head_dim);
            }
        }

        float inv_sum = (sum_exp > 1e-9f) ? (1.0f / sum_exp) : 0.0f;
        int d = 0;
        __m256 vinv = _mm256_set1_ps(inv_sum);
        for (; d + 7 < head_dim; d += 8) {
            __m256 vacc = _mm256_loadu_ps(acc + d);
            _mm256_storeu_ps(out_h + d, _mm256_mul_ps(vacc, vinv));
        }
        for (; d < head_dim; d++) {
            out_h[d] = acc[d] * inv_sum;
        }

        if (heap_allocated) {
            free(acc);
        }
    }
}
