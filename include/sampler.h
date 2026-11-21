#ifndef NOEL_SAMPLER_H
#define NOEL_SAMPLER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float temperature;         /* <= 0.0f = greedy argmax, > 0.0f = stochastic */
    float top_p;               /* Nucleus sampling threshold (0.0 < top_p <= 1.0) */
    int top_k;                 /* Top-K candidate limit (<= 0 = disabled) */
    float repetition_penalty;  /* 1.0 = disabled, > 1.0 = penalize seen tokens */
    int repetition_window;     /* Number of recent tokens to check (0 = entire history) */
    uint64_t seed;             /* PRNG seed (0 = auto seed from time/cpu) */
} noel_sampler_config_t;

typedef struct {
    int32_t id;
    float val;
} noel_token_prob_t;

typedef struct {
    noel_sampler_config_t config;
    uint64_t rng_state[2];
    noel_token_prob_t *candidates;
    int candidate_capacity;
} noel_sampler_t;

/**
 * Initialize sampler with given configuration.
 */
void noel_sampler_init(noel_sampler_t *sampler, const noel_sampler_config_t *cfg, int max_vocab);

/**
 * Free sampler resources.
 */
void noel_sampler_free(noel_sampler_t *sampler);

/**
 * Update sampler configuration at runtime.
 */
void noel_sampler_set_config(noel_sampler_t *sampler, const noel_sampler_config_t *cfg);

/**
 * Sample next token from logits with repetition penalty, temperature, top-k, and top-p.
 * Note: logits array may be modified in-place for performance.
 */
int32_t noel_sampler_sample(noel_sampler_t *sampler,
                            float *logits,
                            int vocab_size,
                            const int32_t *history,
                            int history_len);

/**
 * Uniform random float in [0.0, 1.0).
 */
float noel_sampler_random_f32(noel_sampler_t *sampler);

#ifdef __cplusplus
}
#endif

#endif /* NOEL_SAMPLER_H */
