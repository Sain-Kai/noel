#include "sampler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static inline uint64_t xorshift128plus(uint64_t s[2]) {
    uint64_t s1 = s[0];
    const uint64_t s0 = s[1];
    s[0] = s0;
    s1 ^= s1 << 23;
    s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return s[1] + s0;
}

float noel_sampler_random_f32(noel_sampler_t *sampler) {
    if (!sampler) return 0.0f;
    uint64_t v = xorshift128plus(sampler->rng_state);
    return (float)(v >> 40) * (1.0f / (float)(1ULL << 24));
}

void noel_sampler_init(noel_sampler_t *sampler, const noel_sampler_config_t *cfg, int max_vocab) {
    if (!sampler) return;
    memset(sampler, 0, sizeof(*sampler));

    if (cfg) {
        sampler->config = *cfg;
    } else {
        sampler->config.temperature = 0.7f;
        sampler->config.top_p = 0.9f;
        sampler->config.top_k = 40;
        sampler->config.repetition_penalty = 1.1f;
        sampler->config.repetition_window = 64;
        sampler->config.seed = 0;
    }

    uint64_t seed = sampler->config.seed;
    if (seed == 0) {
        seed = (uint64_t)time(NULL) ^ ((uint64_t)(uintptr_t)sampler << 16);
    }
    sampler->rng_state[0] = seed;
    sampler->rng_state[1] = seed ^ 0x9E3779B97F4A7C15ULL;
    if (sampler->rng_state[0] == 0 && sampler->rng_state[1] == 0) {
        sampler->rng_state[0] = 123456789ULL;
        sampler->rng_state[1] = 987654321ULL;
    }

    int cap = (max_vocab > 1024) ? max_vocab : 1024;
    sampler->candidates = (noel_token_prob_t *)malloc(cap * sizeof(noel_token_prob_t));
    sampler->candidate_capacity = cap;
}

void noel_sampler_free(noel_sampler_t *sampler) {
    if (!sampler) return;
    if (sampler->candidates) {
        free(sampler->candidates);
        sampler->candidates = NULL;
    }
    sampler->candidate_capacity = 0;
}

void noel_sampler_set_config(noel_sampler_t *sampler, const noel_sampler_config_t *cfg) {
    if (!sampler || !cfg) return;
    sampler->config = *cfg;
    if (cfg->seed != 0) {
        sampler->rng_state[0] = cfg->seed;
        sampler->rng_state[1] = cfg->seed ^ 0x9E3779B97F4A7C15ULL;
    }
}

static int cmp_candidates_desc(const void *a, const void *b) {
    const noel_token_prob_t *ca = (const noel_token_prob_t *)a;
    const noel_token_prob_t *cb = (const noel_token_prob_t *)b;
    if (cb->val > ca->val) return 1;
    if (cb->val < ca->val) return -1;
    return 0;
}

int32_t noel_sampler_sample(noel_sampler_t *sampler,
                            float *logits,
                            int vocab_size,
                            const int32_t *history,
                            int history_len) {
    if (!sampler || !logits || vocab_size <= 0) return 0;

    /* 1. Apply Repetition Penalty */
    if (sampler->config.repetition_penalty > 1.0f && history && history_len > 0) {
        int window = sampler->config.repetition_window;
        int start = (window > 0 && window < history_len) ? (history_len - window) : 0;
        float rep_pen = sampler->config.repetition_penalty;

        for (int i = start; i < history_len; i++) {
            int32_t tok = history[i];
            if (tok >= 0 && tok < vocab_size) {
                if (logits[tok] > 0.0f) {
                    logits[tok] /= rep_pen;
                } else {
                    logits[tok] *= rep_pen;
                }
            }
        }
    }

    /* 2. Greedy ArgMax Mode */
    if (sampler->config.temperature <= 1e-4f) {
        int32_t best_tok = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_tok = i;
            }
        }
        return best_tok;
    }

    /* 3. Softmax with Temperature Scaling */
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    /* Resize candidates buffer if needed */
    if (sampler->candidate_capacity < vocab_size) {
        sampler->candidates = (noel_token_prob_t *)realloc(sampler->candidates, vocab_size * sizeof(noel_token_prob_t));
        sampler->candidate_capacity = vocab_size;
    }

    float inv_temp = 1.0f / sampler->config.temperature;
    noel_token_prob_t *cands = sampler->candidates;
    int num_cands = 0;

    for (int i = 0; i < vocab_size; i++) {
        float diff = (logits[i] - max_val) * inv_temp;
        if (diff > -15.0f) {
            cands[num_cands].id = i;
            cands[num_cands].val = expf(diff);
            num_cands++;
        }
    }

    if (num_cands == 0) {
        cands[0].id = 0;
        cands[0].val = 1.0f;
        num_cands = 1;
    }

    /* Sort descending */
    qsort(cands, num_cands, sizeof(noel_token_prob_t), cmp_candidates_desc);

    /* 4. Top-K truncation */
    if (sampler->config.top_k > 0 && sampler->config.top_k < num_cands) {
        num_cands = sampler->config.top_k;
    }

    /* 5. Top-P (Nucleus) truncation */
    float total_mass = 0.0f;
    for (int i = 0; i < num_cands; i++) {
        total_mass += cands[i].val;
    }

    float top_p = sampler->config.top_p;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cutoff = top_p * total_mass;
        float cum_sum = 0.0f;
        int p_cands = 0;
        for (int i = 0; i < num_cands; i++) {
            cum_sum += cands[i].val;
            p_cands++;
            if (cum_sum >= cutoff) {
                break;
            }
        }
        num_cands = p_cands;
        total_mass = cum_sum;
    }

    /* 6. Sample token proportionally */
    float r = noel_sampler_random_f32(sampler) * total_mass;
    float acc = 0.0f;
    for (int i = 0; i < num_cands; i++) {
        acc += cands[i].val;
        if (r <= acc || i == num_cands - 1) {
            return cands[i].id;
        }
    }

    return cands[0].id;
}
