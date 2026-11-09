#ifndef AETHER_ENGINE_H
#define AETHER_ENGINE_H

#include "aether.h"
#include "hardware.h"
#include "cuda_backend.h"
#include "nvme_streamer.h"
#include "simd_vnni.h"
#include "sparsity.h"
#include "tokenizer.h"
#include "priority.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *token_embedding;    /* vocab_size x dim */
    float *attn_norm;          /* n_layers x dim */
    block_q4_0 *wq;            /* n_layers x (dim * dim) */
    block_q4_0 *wk;            /* n_layers x (dim * kv_dim) */
    block_q4_0 *wv;            /* n_layers x (dim * kv_dim) */
    block_q4_0 *wo;            /* n_layers x (dim * dim) */
    float *ffn_norm;           /* n_layers x dim */
    /* Hot neuron weights pinned in GPU VRAM */
    block_q4_0 *w_hot_gate;
    block_q4_0 *w_hot_up;
    block_q4_0 *w_hot_down;
    /* MoE / Cold weights staged via NVMe Streamer */
    float *moe_gate;           /* n_layers x (n_experts * dim) */
    float *final_norm;         /* dim */
    block_q4_0 *lm_head;       /* vocab_size x dim */
} aether_weights_t;

typedef struct {
    float *x;                  /* dim */
    float *xb;                 /* dim */
    float *xb2;                /* dim */
    float *hb;                 /* hidden_dim */
    float *hb2;                /* hidden_dim */
    float *q;                  /* dim */
    float *k;                  /* kv_dim */
    float *v;                  /* kv_dim */
    float *att;                /* n_heads x seq_len */
    /* Paged KV Cache */
    float *key_cache;          /* n_layers x seq_len x kv_dim */
    float *val_cache;          /* n_layers x seq_len x kv_dim */
    float *logits;             /* vocab_size */
} aether_state_t;

typedef struct {
    aether_config_t config;
    aether_weights_t weights;
    aether_state_t state;
    aether_tokenizer_t tokenizer;
    aether_priority_mgr_t priority_mgr;
    aether_stats_t stats;
    bool has_gpu;
} aether_engine_t;

/* Initialize the full 4-tier engine */
bool aether_engine_init(aether_engine_t *engine, const aether_config_t *cfg, const char *model_path);

/* Generate next token given previous token ID and position */
int32_t aether_engine_forward(aether_engine_t *engine, int32_t token, int pos);

/* Generate a sequence of tokens from a prompt */
void aether_engine_generate(aether_engine_t *engine, const char *prompt, int max_gen_tokens);

/* Free engine resources */
void aether_engine_shutdown(aether_engine_t *engine);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_ENGINE_H */
