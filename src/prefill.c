/**
 * @file prefill.c
 * @brief Chunked Prompt Prefill Engine Implementation
 *
 * Implements chunk partitioning, activation state buffer management,
 * and prefill execution primitives.
 */

#include "prefill.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
