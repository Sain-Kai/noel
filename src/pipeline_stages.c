#include "pipeline_stages.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------- */
/* PREREQUISITE IMPLEMENTATIONS                                  */
/* ------------------------------------------------------------- */

void aether_prereq_embedding_lookup(float *dst_x, const float *host_embeddings,
                                   int32_t token, int dim) {
    /* Embedding table resides in Host RAM, keeping GPU VRAM completely free */
    const float *src = host_embeddings + (size_t)token * dim;
    memcpy(dst_x, src, dim * sizeof(float));
}

void aether_prereq_precompute_rope(float *cos_buf, float *sin_buf, int head_dim,
                                   int pos, float rope_theta) {
    int half_dim = head_dim / 2;
    for (int i = 0; i < half_dim; i++) {
        float freq = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
        float angle = (float)pos * freq;
        cos_buf[i] = cosf(angle);
        sin_buf[i] = sinf(angle);
    }
}

void noel_rope_extended(float *vec, int n_heads, int head_dim, int pos,
                        float rope_theta, float freq_scale, noel_rope_style_t style) {
    if (!vec || n_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) return;
    if (rope_theta <= 0.0f) rope_theta = 10000.0f;
    if (freq_scale <= 0.0f) freq_scale = 1.0f;

    int half_dim = head_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        float *head_vec = vec + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float freq = (1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim)) * freq_scale;
            float angle = (float)pos * freq;
            float cos_th = cosf(angle);
            float sin_th = sinf(angle);

            if (style == NOEL_ROPE_STYLE_NEOX) {
                float v0 = head_vec[i];
                float v1 = head_vec[i + half_dim];
                head_vec[i]            = v0 * cos_th - v1 * sin_th;
                head_vec[i + half_dim] = v0 * sin_th + v1 * cos_th;
            } else {
                float v0 = head_vec[2 * i];
                float v1 = head_vec[2 * i + 1];
                head_vec[2 * i]     = v0 * cos_th - v1 * sin_th;
                head_vec[2 * i + 1] = v0 * sin_th + v1 * cos_th;
            }
        }
    }
}

/* ------------------------------------------------------------- */
/* POST-REQUISITE IMPLEMENTATIONS (E-Core Optimized)             */
/* ------------------------------------------------------------- */

void aether_postreq_apply_penalties(float *logits, int vocab_size,
                                   const int32_t *history, int history_len,
                                   float penalty, int window) {
    if (penalty <= 1.0f || history_len <= 0) return;

    int start = (window > 0 && history_len > window) ? (history_len - window) : 0;
    for (int i = start; i < history_len; i++) {
        int32_t tok = history[i];
        if (tok >= 0 && tok < vocab_size) {
            if (logits[tok] > 0.0f) {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

void aether_postreq_grammar_mask(float *logits, int vocab_size, int fsm_state) {
    /* Example JSON schema enforcement:
       State 0: Must start with '{'
       State 1: Expect string key '"' or closing '}'
       Mask out all tokens incompatible with current syntax state */
    if (fsm_state == 0) {
        /* In state 0, only '{' (ASCII 123) is valid */
        for (int v = 0; v < vocab_size; v++) {
            if (v != '{') {
                logits[v] = -1e9f;
            }
        }
    }
}

/* Helper struct for Top-K sorting */
typedef struct {
    int32_t id;
    float prob;
} token_prob_t;

int32_t aether_postreq_sample(float *logits, int vocab_size,
                              const aether_sampling_params_t *params) {
    /* 1. Greedy decoding */
    if (params->temperature <= 1e-4f) {
        int32_t best_id = 0;
        float best_val = logits[0];
        for (int v = 1; v < vocab_size; v++) {
            if (logits[v] > best_val) {
                best_val = logits[v];
                best_id = v;
            }
        }
        return best_id;
    }

    /* 2. Temperature scaling */
    float inv_temp = 1.0f / params->temperature;
    for (int v = 0; v < vocab_size; v++) {
        logits[v] *= inv_temp;
    }

    /* 3. Softmax over entire vocabulary */
    float max_logit = logits[0];
    for (int v = 1; v < vocab_size; v++) {
        if (logits[v] > max_logit) max_logit = logits[v];
    }
    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        logits[v] = expf(logits[v] - max_logit);
        sum_exp += logits[v];
    }
    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    for (int v = 0; v < vocab_size; v++) {
        logits[v] *= inv_sum;
    }

    /* 4. Top-K Selection */
    int top_k = params->top_k;
    if (top_k <= 0 || top_k > vocab_size) top_k = vocab_size;
    if (top_k > 256) top_k = 256; /* Keep working set cache-resident on E-core L2 */

    token_prob_t *candidates = (token_prob_t*)alloca(top_k * sizeof(token_prob_t));
    for (int i = 0; i < top_k; i++) {
        candidates[i].id = -1;
        candidates[i].prob = -1.0f;
    }

    for (int v = 0; v < vocab_size; v++) {
        float p = logits[v];
        if (p > candidates[top_k - 1].prob) {
            int ins = top_k - 1;
            while (ins > 0 && p > candidates[ins - 1].prob) {
                candidates[ins] = candidates[ins - 1];
                ins--;
            }
            candidates[ins].id = v;
            candidates[ins].prob = p;
        }
    }

    /* 5. Top-P (Nucleus) filtering */
    float cum_prob = 0.0f;
    int num_candidates = 0;
    for (int i = 0; i < top_k; i++) {
        if (candidates[i].id < 0) break;
        cum_prob += candidates[i].prob;
        num_candidates++;
        if (cum_prob >= params->top_p) break;
    }

    /* Renormalize top-p candidates */
    float inv_cum = 1.0f / cum_prob;
    for (int i = 0; i < num_candidates; i++) {
        candidates[i].prob *= inv_cum;
    }

    /* 6. Random sample from filtered nucleus */
    float r = ((float)rand() / (float)RAND_MAX);
    float running_sum = 0.0f;
    for (int i = 0; i < num_candidates; i++) {
        running_sum += candidates[i].prob;
        if (r <= running_sum || i == num_candidates - 1) {
            return candidates[i].id;
        }
    }

    return candidates[0].id;
}

int aether_postreq_detokenize_stream(aether_detokenizer_state_t *state,
                                     const char *piece, char *out_text, int max_len) {
    if (!piece || !out_text || max_len <= 0) return 0;

    int written = snprintf(out_text, max_len, "%s", piece);
    (void)state;
    return written > 0 ? written : 0;
}
