/**
 * @file speculative.c
 * @brief Speculative Decoding Runtime & Proposal Implementation
 *
 * Implements draft candidate generation, stochastic Leviathan rejection sampling,
 * and speculative state coordination for accelerated autoregressive decoding.
 */

#include "speculative.h"
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void noel_speculative_config_init_default(noel_speculative_config_t *cfg) {
    if (!cfg) return;
    cfg->num_speculative_tokens = NOEL_DEFAULT_SPEC_TOKENS;
    cfg->draft_temperature = 0.7f;
    cfg->acceptance_threshold = 0.0f;
    memset(cfg->draft_model_path, 0, sizeof(cfg->draft_model_path));
}

bool noel_speculative_state_init(noel_speculative_state_t *state, int max_spec_tokens, int vocab_size) {
    if (!state || max_spec_tokens <= 0 || vocab_size <= 0) return false;

    memset(state, 0, sizeof(noel_speculative_state_t));
    state->max_spec_tokens = max_spec_tokens;
    state->vocab_size = vocab_size;

    size_t dist_bytes = (size_t)vocab_size * sizeof(float);
    state->target_dist = (float *)calloc(1, dist_bytes);
    state->draft_dist = (float *)calloc(1, dist_bytes);
    state->residual_dist = (float *)calloc(1, dist_bytes);

    if (!state->target_dist || !state->draft_dist || !state->residual_dist) {
        noel_speculative_state_free(state);
        return false;
    }

    return true;
}

void noel_speculative_state_free(noel_speculative_state_t *state) {
    if (!state) return;
    if (state->target_dist) { free(state->target_dist); state->target_dist = NULL; }
    if (state->draft_dist) { free(state->draft_dist); state->draft_dist = NULL; }
    if (state->residual_dist) { free(state->residual_dist); state->residual_dist = NULL; }
    state->vocab_size = 0;
    state->max_spec_tokens = 0;
}

void noel_speculative_reset_stats(noel_speculative_state_t *state) {
    if (!state) return;
    state->total_draft_tokens = 0;
    state->accepted_draft_tokens = 0;
    state->speculative_rounds = 0;
}

double noel_speculative_get_acceptance_rate(const noel_speculative_state_t *state) {
    if (!state || state->total_draft_tokens == 0) return 0.0;
    return (double)state->accepted_draft_tokens / (double)state->total_draft_tokens;
}

static float noel_calc_token_prob(const float *logits, int vocab_size, int32_t token, float temperature) {
    if (!logits || token < 0 || token >= vocab_size) return 0.0f;

    if (temperature <= 0.01f) {
        /* Greedy argmax check */
        int32_t best_tok = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_tok = i;
            }
        }
        return (token == best_tok) ? 1.0f : 0.0f;
    }

    float inv_temp = 1.0f / temperature;
    float max_logit = logits[0] * inv_temp;
    for (int i = 1; i < vocab_size; i++) {
        float val = logits[i] * inv_temp;
        if (val > max_logit) max_logit = val;
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        sum_exp += expf(logits[i] * inv_temp - max_logit);
    }

    float token_exp = expf(logits[token] * inv_temp - max_logit);
    return (sum_exp > 0.0f) ? (token_exp / sum_exp) : 0.0f;
}

int noel_speculative_propose(aether_engine_t *draft_engine,
                             noel_speculative_state_t *spec_state,
                             int32_t current_token,
                             int current_pos,
                             int gamma,
                             noel_speculative_proposal_t *proposal) {
    if (!draft_engine || !proposal || gamma <= 0) return 0;
    if (gamma > NOEL_MAX_SPEC_TOKENS) gamma = NOEL_MAX_SPEC_TOKENS;

    memset(proposal, 0, sizeof(noel_speculative_proposal_t));
    int32_t in_token = current_token;
    int vocab_size = draft_engine->config.vocab_size;
    float temp = draft_engine->sampler.config.temperature;

    for (int i = 0; i < gamma; i++) {
        int pos = current_pos + i;
        int32_t draft_token = aether_engine_forward(draft_engine, in_token, pos);
        float prob = noel_calc_token_prob(draft_engine->state.logits, vocab_size, draft_token, temp);

        proposal->tokens[i] = draft_token;
        proposal->draft_probs[i] = prob;
        in_token = draft_token;
    }

    proposal->num_proposed = gamma;
    if (spec_state) {
        spec_state->total_draft_tokens += gamma;
        spec_state->speculative_rounds++;
    }

    return gamma;
}
