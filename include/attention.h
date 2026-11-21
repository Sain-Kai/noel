#ifndef NOEL_ATTENTION_H
#define NOEL_ATTENTION_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Grouped-Query Attention (GQA) with vectorized online softmax.
 *
 * @param out         Output vector [dim = n_heads * head_dim]
 * @param q           Query vector for current token [dim = n_heads * head_dim]
 * @param key_cache   Key cache for this layer [seq_len * (n_kv_heads * head_dim)]
 * @param val_cache   Value cache for this layer [seq_len * (n_kv_heads * head_dim)]
 * @param n_heads     Number of query attention heads
 * @param n_kv_heads  Number of key/value heads (GQA when n_kv_heads < n_heads)
 * @param head_dim    Dimension of each head (e.g., 64 or 128)
 * @param seq_len     Total sequence buffer capacity
 * @param pos         Current token position (0-indexed, attends to tokens 0..pos)
 * @param n_threads   Number of CPU threads for parallelizing across heads
 */
void noel_gqa_attention(float *out,
                        const float *q,
                        const float *key_cache,
                        const float *val_cache,
                        int n_heads,
                        int n_kv_heads,
                        int head_dim,
                        int seq_len,
                        int pos,
                        int n_threads);

/**
 * Optimized AVX2 dot-product between two float vectors of length `dim`.
 */
float noel_vec_dot_f32_avx2(const float *a, const float *b, int dim);

#ifdef __cplusplus
}
#endif

#endif /* NOEL_ATTENTION_H */
