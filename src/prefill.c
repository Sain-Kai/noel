/**
 * @file prefill.c
 * @brief Chunked Prompt Prefill Engine Implementation
 *
 * Implements chunk partitioning, activation state buffer management,
 * and prefill execution primitives.
 */

#include "prefill.h"
#include "engine.h"
#include "dequant.h"
#include "simd_vnni.h"
#include "pipeline_stages.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void noel_prefill_config_init_default(noel_prefill_config_t *cfg) {
    if (!cfg) return;
    cfg->chunk_size = NOEL_DEFAULT_PREFILL_CHUNK_SIZE;
    cfg->causal = true;
    cfg->num_threads = 0; /* Auto-detect */
    cfg->use_quantized_kv = false;
}

bool noel_prefill_state_init(noel_prefill_state_t *state, int chunk_size, int dim, int kv_dim, int seq_len) {
    if (!state || chunk_size <= 0 || dim <= 0 || kv_dim <= 0 || seq_len <= 0) {
        return false;
    }

    memset(state, 0, sizeof(noel_prefill_state_t));
    state->allocated_chunk_size = chunk_size;
    state->dim = dim;
    state->kv_dim = kv_dim;
    state->seq_len = seq_len;

    size_t q_sz = (size_t)chunk_size * dim * sizeof(float);
    size_t kv_sz = (size_t)chunk_size * kv_dim * sizeof(float);
    size_t dim_sz = (size_t)chunk_size * dim * sizeof(float);
    size_t scores_sz = (size_t)chunk_size * seq_len * sizeof(float);

    state->q_chunk = (float *)calloc(1, q_sz);
    state->k_chunk = (float *)calloc(1, kv_sz);
    state->v_chunk = (float *)calloc(1, kv_sz);
    state->xb_chunk = (float *)calloc(1, dim_sz);
    state->xb2_chunk = (float *)calloc(1, dim_sz);
    state->attn_scores = (float *)calloc(1, scores_sz);

    if (!state->q_chunk || !state->k_chunk || !state->v_chunk ||
        !state->xb_chunk || !state->xb2_chunk || !state->attn_scores) {
        noel_prefill_state_free(state);
        return false;
    }

    return true;
}

void noel_prefill_state_free(noel_prefill_state_t *state) {
    if (!state) return;

    if (state->q_chunk) { free(state->q_chunk); state->q_chunk = NULL; }
    if (state->k_chunk) { free(state->k_chunk); state->k_chunk = NULL; }
    if (state->v_chunk) { free(state->v_chunk); state->v_chunk = NULL; }
    if (state->xb_chunk) { free(state->xb_chunk); state->xb_chunk = NULL; }
    if (state->xb2_chunk) { free(state->xb2_chunk); state->xb2_chunk = NULL; }
    if (state->attn_scores) { free(state->attn_scores); state->attn_scores = NULL; }

    state->allocated_chunk_size = 0;
    state->dim = 0;
    state->kv_dim = 0;
    state->seq_len = 0;
}

int noel_prefill_calc_num_chunks(int prompt_len, int chunk_size) {
    if (prompt_len <= 0 || chunk_size <= 0) return 0;
    return (prompt_len + chunk_size - 1) / chunk_size;
}

noel_prefill_chunk_t noel_prefill_get_chunk(const int32_t *tokens, int prompt_len, int chunk_size, int chunk_idx) {
    noel_prefill_chunk_t chunk;
    memset(&chunk, 0, sizeof(chunk));

    if (!tokens || prompt_len <= 0 || chunk_size <= 0 || chunk_idx < 0) {
        return chunk;
    }

    int start_pos = chunk_idx * chunk_size;
    if (start_pos >= prompt_len) {
        return chunk;
    }

    int remaining = prompt_len - start_pos;
    int cur_tokens = (remaining < chunk_size) ? remaining : chunk_size;

    chunk.chunk_idx = chunk_idx;
    chunk.start_pos = start_pos;
    chunk.num_tokens = cur_tokens;
    chunk.tokens = tokens + start_pos;
    return chunk;
}

void noel_prefill_project_qkv(noel_prefill_state_t *state,
                              const aether_weights_t *weights,
                              int layer,
                              const noel_prefill_chunk_t *chunk,
                              int dim,
                              int n_heads,
                              int n_kv_heads,
                              int head_dim,
                              float rope_theta,
                              int num_threads) {
    if (!state || !weights || !chunk || chunk->num_tokens <= 0) return;

    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    size_t q4_nb = sizeof(block_q4_0) / 32;

    int threads = (num_threads > 0) ? num_threads : g_cpu_topo.num_p_cores;
    if (threads < 1) threads = 1;

    if (chunk->num_tokens == 1) {
        /* Single token in chunk: direct multi-threaded GEMV */
        const float *x = state->xb_chunk;
        float *q = state->q_chunk;
        float *k = state->k_chunk;
        float *v = state->v_chunk;

        if (weights->is_bound_gguf && weights->layer_tensors[layer].attn_q) {
            noel_gemv_quantized(weights->layer_tensors[layer].attn_q_type, q,
                                weights->layer_tensors[layer].attn_q, x, q_dim, dim, threads);
            noel_gemv_quantized(weights->layer_tensors[layer].attn_k_type, k,
                                weights->layer_tensors[layer].attn_k, x, kv_dim, dim, threads);
            noel_gemv_quantized(weights->layer_tensors[layer].attn_v_type, v,
                                weights->layer_tensors[layer].attn_v, x, kv_dim, dim, threads);
        } else if (weights->wq) {
            aether_gemv_q4_0(q, weights->wq + layer * q_dim * q4_nb, x, q_dim, dim, threads);
            aether_gemv_q4_0(k, weights->wk + layer * kv_dim * q4_nb, x, kv_dim, dim, threads);
            aether_gemv_q4_0(v, weights->wv + layer * kv_dim * q4_nb, x, kv_dim, dim, threads);
        }

        int pos = chunk->start_pos;
        aether_rope(q, n_heads, head_dim, pos, rope_theta);
        aether_rope(k, n_kv_heads, head_dim, pos, rope_theta);
    } else {
        /* Multi-token chunk: parallelize across tokens */
        #pragma omp parallel for schedule(dynamic) num_threads(threads)
        for (int t = 0; t < chunk->num_tokens; t++) {
            const float *x = state->xb_chunk + (size_t)t * dim;
            float *q = state->q_chunk + (size_t)t * q_dim;
            float *k = state->k_chunk + (size_t)t * kv_dim;
            float *v = state->v_chunk + (size_t)t * kv_dim;

            if (weights->is_bound_gguf && weights->layer_tensors[layer].attn_q) {
                noel_gemv_quantized(weights->layer_tensors[layer].attn_q_type, q,
                                    weights->layer_tensors[layer].attn_q, x, q_dim, dim, 1);
                noel_gemv_quantized(weights->layer_tensors[layer].attn_k_type, k,
                                    weights->layer_tensors[layer].attn_k, x, kv_dim, dim, 1);
                noel_gemv_quantized(weights->layer_tensors[layer].attn_v_type, v,
                                    weights->layer_tensors[layer].attn_v, x, kv_dim, dim, 1);
            } else if (weights->wq) {
                aether_gemv_q4_0(q, weights->wq + layer * q_dim * q4_nb, x, q_dim, dim, 1);
                aether_gemv_q4_0(k, weights->wk + layer * kv_dim * q4_nb, x, kv_dim, dim, 1);
                aether_gemv_q4_0(v, weights->wv + layer * kv_dim * q4_nb, x, kv_dim, dim, 1);
            }

            int pos = chunk->start_pos + t;
            aether_rope(q, n_heads, head_dim, pos, rope_theta);
            aether_rope(k, n_kv_heads, head_dim, pos, rope_theta);
        }
    }
}
