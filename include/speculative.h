/**
 * @file speculative.h
 * @brief Speculative Decoding Architecture & Speculative Verification Structures
 *
 * Implements speculative decoding definitions, candidate proposal buffers,
 * draft generation structures, and stochastic rejection sampling for accelerating
 * autoregressive inference on heterogeneous tiered memory systems.
 *
 * Zero OEM/hardware-specific identifiers; pure cross-platform C99/POSIX.
 */

#ifndef NOEL_SPECULATIVE_H
#define NOEL_SPECULATIVE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NOEL_DEFAULT_SPEC_TOKENS 4
#define NOEL_MAX_SPEC_TOKENS     16

/**
 * @brief Speculative decoding configuration
 */
typedef struct {
    int num_speculative_tokens;  /* Number of candidate draft tokens gamma (default: 4) */
    float draft_temperature;     /* Draft model temperature */
    float acceptance_threshold;  /* Minimum acceptance threshold */
    char draft_model_path[512];  /* Filepath to draft GGUF model */
} noel_speculative_config_t;

/**
 * @brief Candidate token proposal batch
 */
typedef struct {
    int num_proposed;                           /* Number of draft tokens generated */
    int32_t tokens[NOEL_MAX_SPEC_TOKENS];       /* Proposed token IDs */
    float draft_probs[NOEL_MAX_SPEC_TOKENS];    /* Draft model probability per token */
    float target_probs[NOEL_MAX_SPEC_TOKENS];   /* Target model probability per token */
    int accepted_count;                         /* Number of accepted tokens in batch */
    int32_t final_sampled_token;                /* Bonus/correction token sampled */
} noel_speculative_proposal_t;

/**
 * @brief Speculative runtime execution and telemetry state
 */
typedef struct {
    int max_spec_tokens;
    int vocab_size;

    /* Telemetry & Performance Counters */
    uint64_t total_draft_tokens;
    uint64_t accepted_draft_tokens;
    uint64_t speculative_rounds;

    /* Scratch distributions for verification and residual sampling */
    float *target_dist;     /* vocab_size */
    float *draft_dist;      /* vocab_size */
    float *residual_dist;   /* vocab_size */
} noel_speculative_state_t;

/**
 * @brief Initialize default speculative decoding configuration
 */
void noel_speculative_config_init_default(noel_speculative_config_t *cfg);

/**
 * @brief Allocate speculative execution and scratch distributions state
 */
bool noel_speculative_state_init(noel_speculative_state_t *state, int max_spec_tokens, int vocab_size);

/**
 * @brief Free speculative execution state
 */
void noel_speculative_state_free(noel_speculative_state_t *state);

/**
 * @brief Reset speculative telemetry counters
 */
void noel_speculative_reset_stats(noel_speculative_state_t *state);

/**
 * @brief Get empirical acceptance rate (accepted_tokens / total_draft_tokens)
 */
double noel_speculative_get_acceptance_rate(const noel_speculative_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* NOEL_SPECULATIVE_H */
