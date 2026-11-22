/**
 * @file prefill.h
 * @brief Chunked Prompt Prefill Engine for Batched Context Ingestion
 *
 * Provides chunked prompt partitioning, scratch activation buffers,
 * and batched execution structures to optimize Time-To-First-Token (TTFT)
 * and memory bandwidth utilization during prompt evaluation.
 *
 * Zero OEM/hardware-specific identifiers; pure cross-platform C99/POSIX.
 */

#ifndef NOEL_PREFILL_H
#define NOEL_PREFILL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NOEL_DEFAULT_PREFILL_CHUNK_SIZE 128
#define NOEL_MAX_PREFILL_CHUNK_SIZE     2048

/**
 * @brief Prefill configuration parameters
 */
typedef struct {
    int chunk_size;         /* Tokens per chunk (default: 128) */
    bool causal;            /* Enforce causal masking (default: true) */
    int num_threads;        /* OpenMP execution threads */
    bool use_quantized_kv;  /* Direct quantization into KV cache */
} noel_prefill_config_t;

/**
 * @brief Partitioned chunk slice descriptor
 */
typedef struct {
    int chunk_idx;          /* 0-indexed chunk sequence */
    int start_pos;          /* Starting absolute sequence position */
    int num_tokens;         /* Number of tokens in this chunk */
    const int32_t *tokens;  /* Pointer to token ID array slice */
} noel_prefill_chunk_t;

/**
 * @brief Scratch activation and attention buffers for chunked prefill
 */
typedef struct {
    int allocated_chunk_size;
    int dim;
    int kv_dim;
    int seq_len;
    float *q_chunk;         /* Chunk query activations [chunk_size * dim] */
    float *k_chunk;         /* Chunk key activations [chunk_size * kv_dim] */
    float *v_chunk;         /* Chunk value activations [chunk_size * kv_dim] */
    float *xb_chunk;        /* Chunk intermediate buffer [chunk_size * dim] */
    float *xb2_chunk;       /* Chunk attention output buffer [chunk_size * dim] */
    float *attn_scores;     /* Chunk attention scores scratchpad [chunk_size * seq_len] */
} noel_prefill_state_t;

/**
 * @brief Initialize default prefill configuration
 */
void noel_prefill_config_init_default(noel_prefill_config_t *cfg);

/**
 * @brief Allocate chunked prefill scratch activation state
 */
bool noel_prefill_state_init(noel_prefill_state_t *state, int chunk_size, int dim, int kv_dim, int seq_len);

/**
 * @brief Free chunked prefill scratch activation state
 */
void noel_prefill_state_free(noel_prefill_state_t *state);

/**
 * @brief Calculate total number of chunks for a prompt of given length
 */
int noel_prefill_calc_num_chunks(int prompt_len, int chunk_size);

/**
 * @brief Extract chunk metadata and token slice for a given chunk index
 */
noel_prefill_chunk_t noel_prefill_get_chunk(const int32_t *tokens, int prompt_len, int chunk_size, int chunk_idx);

#ifdef __cplusplus
}
#endif

#endif /* NOEL_PREFILL_H */
