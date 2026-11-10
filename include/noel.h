#ifndef NOEL_H
#define NOEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NOEL_ALIGNMENT 4096
#define NOEL_BLOCK_SIZE 32
#define NOEL_MAGIC 0x4E4F454C /* "NOEL" */
#define NOEL_VERSION 2
#define NOEL_RELEASE "v1.1.0"

/* Compatibility aliases */
#define AETHER_ALIGNMENT NOEL_ALIGNMENT
#define AETHER_BLOCK_SIZE NOEL_BLOCK_SIZE
#define AETHER_MAGIC NOEL_MAGIC
#define AETHER_VERSION NOEL_VERSION
#define AETHER_RELEASE NOEL_RELEASE

typedef enum {
    NOEL_MODE_AUTO = 0,               /* Auto-detect based on VRAM capacity */
    NOEL_MODE_MONOLITHIC_GPU = 1,      /* 100% VRAM execution for sub-VRAM models (0.5B-3B) */
    NOEL_MODE_HETEROGENEOUS_4TIER = 2  /* 4-Tier HPRTA execution for out-of-core models (14B-70B) */
} noel_execution_mode_t;

typedef noel_execution_mode_t aether_execution_mode_t;

typedef enum {
    NOEL_DTYPE_FP32 = 0,
    NOEL_DTYPE_FP16 = 1,
    NOEL_DTYPE_Q8_0 = 2,
    NOEL_DTYPE_Q4_0 = 3
} noel_dtype_t;

typedef noel_dtype_t aether_dtype_t;

/* Quantized block definitions */
typedef struct {
    float scale;
    int8_t qs[NOEL_BLOCK_SIZE]; /* 32 bytes */
} block_q8_0;

typedef struct {
    float scale;
    uint8_t qs[NOEL_BLOCK_SIZE / 2]; /* 16 bytes for 32 4-bit nibbles */
} block_q4_0;

typedef struct {
    int32_t dim;             /* Hidden size (e.g., 4096) */
    int32_t hidden_dim;      /* Intermediate MLP size (e.g., 11008 or 14336) */
    int32_t n_layers;        /* Number of transformer layers */
    int32_t n_heads;         /* Attention query heads */
    int32_t n_kv_heads;      /* Attention KV heads (for GQA) */
    int32_t vocab_size;      /* Vocabulary size */
    int32_t seq_len;         /* Max context length */
    int32_t n_experts;       /* Number of experts (0 or 1 = dense model) */
    int32_t n_active_experts;/* Top-k active experts per token */
    float hot_ratio;         /* Ratio of hot neurons pinned in VRAM (PowerInfer) */
    float norm_eps;          /* RMSNorm epsilon */
    float rope_theta;        /* RoPE base frequency */
    noel_execution_mode_t execution_mode; /* Dynamic auto-tiering mode */
} noel_config_t;

typedef noel_config_t aether_config_t;

typedef struct {
    double total_ms;
    double attention_gpu_ms;
    double cold_gemv_cpu_ms;
    double ssd_io_ms;
    double prefetch_saved_ms;
    int64_t total_tokens;
    double tokens_per_sec;
} noel_stats_t;

typedef noel_stats_t aether_stats_t;

/* Calculate aligned byte size of a Q4_0 expert matrix */
static inline size_t noel_calc_expert_bytes(int dim, int hidden_dim) {
    size_t nb = hidden_dim / NOEL_BLOCK_SIZE;
    size_t raw_bytes = (size_t)dim * nb * sizeof(block_q4_0);
    return ((raw_bytes + NOEL_ALIGNMENT - 1) / NOEL_ALIGNMENT) * NOEL_ALIGNMENT;
}

static inline uint64_t noel_calc_expert_offset(const noel_config_t *cfg, int layer, int expert) {
    size_t exp_bytes = noel_calc_expert_bytes(cfg->dim, cfg->hidden_dim);
    return (uint64_t)NOEL_ALIGNMENT + (uint64_t)(layer * cfg->n_experts + expert) * exp_bytes;
}

#define aether_calc_expert_bytes noel_calc_expert_bytes
#define aether_calc_expert_offset noel_calc_expert_offset

#ifdef __cplusplus
}
#endif

#endif /* NOEL_H */
