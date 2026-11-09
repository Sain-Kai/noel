#ifndef AETHER_PIPELINE_STAGES_H
#define AETHER_PIPELINE_STAGES_H

#include "aether.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sampling parameters */
typedef struct {
    float temperature;
    float top_p;
    int top_k;
    float repetition_penalty;
    int repetition_window;    /* Number of past tokens to check */
    bool grammar_enforce_json;/* JSON schema constraint enforcement */
} aether_sampling_params_t;

/* State for asynchronous post-requisite sampling running on E-cores */
typedef struct {
    float *logits;            /* vocab_size */
    int vocab_size;
    const int32_t *history;   /* past generated tokens */
    int history_len;
    aether_sampling_params_t params;
} aether_sampler_task_t;

/* ------------------------------------------------------------- */
/* PREREQUISITE STAGES (Offloaded from VRAM to Host RAM & CPU)   */
/* ------------------------------------------------------------- */

/* Embedding Lookup from Host RAM (saves 0.5 - 2.0 GiB of VRAM) */
void aether_prereq_embedding_lookup(float *dst_x, const float *host_embeddings,
                                   int32_t token, int dim);

/* Precompute RoPE cos/sin frequencies for current sequence position */
void aether_prereq_precompute_rope(float *cos_buf, float *sin_buf, int head_dim,
                                   int pos, float rope_theta);

/* ------------------------------------------------------------- */
/* POST-REQUISITE STAGES (Offloaded to E-cores)                  */
/* ------------------------------------------------------------- */

/* Apply repetition and frequency penalties across context history */
void aether_postreq_apply_penalties(float *logits, int vocab_size,
                                   const int32_t *history, int history_len,
                                   float penalty, int window);

/* Enforce JSON/Grammar FSM constraints by masking invalid token logits */
void aether_postreq_grammar_mask(float *logits, int vocab_size, int fsm_state);

/* Advanced CPU Sampling: Temperature + Top-K (Quickselect) + Top-P (Nucleus) */
int32_t aether_postreq_sample(float *logits, int vocab_size,
                              const aether_sampling_params_t *params);

/* Streaming Detokenizer with UTF-8 byte boundary reconstruction */
typedef struct {
    char incomplete_utf8[4];
    int incomplete_len;
} aether_detokenizer_state_t;

int aether_postreq_detokenize_stream(aether_detokenizer_state_t *state,
                                     const char *piece, char *out_text, int max_len);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_PIPELINE_STAGES_H */
