#ifndef AETHER_PRIORITY_H
#define AETHER_PRIORITY_H

#include "aether.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TIER_VRAM = 0,    /* Highest bandwidth: Dedicated GPU VRAM @ 192+ GB/s */
    TIER_RAM  = 1,    /* Mid bandwidth: Host System RAM @ ~50 GB/s */
    TIER_NVME = 2     /* High capacity: PCIe NVMe SSD @ 5+ GB/s Direct I/O */
} aether_tier_t;

typedef enum {
    COMP_EMBEDDING,
    COMP_ATTENTION_LAYER,
    COMP_MOE_EXPERT,
    COMP_HOT_NEURONS,
    COMP_LM_HEAD
} aether_comp_type_t;

typedef struct {
    int id;
    aether_comp_type_t type;
    int layer_idx;
    int expert_idx;            /* -1 if not an MoE expert */
    size_t size_bytes;
    uint64_t file_offset;

    /* Frequency & Priority tracking */
    uint64_t total_hits;
    float ema_frequency;       /* Exponential Moving Average of activation rate */
    float priority_score;      /* Calculated dynamic priority score */
    int priority_rank;         /* 1 = highest priority, N = lowest */

    /* Physical Tier Placement */
    aether_tier_t assigned_tier;
    void *ram_ptr;             /* Pointer if resident in Host RAM */
    void *vram_ptr;            /* CUDA device pointer if resident in VRAM */
    bool is_pinned;            /* Pinned permanently in VRAM (e.g. embeddings/head) */
} aether_component_t;

typedef struct {
    aether_component_t *components;
    int total_components;
    size_t vram_budget_bytes;
    size_t ram_budget_bytes;
    size_t vram_used_bytes;
    size_t ram_used_bytes;
    float ema_decay;           /* EMA decay factor (e.g. 0.95) */
    uint64_t total_tokens_observed;
    noel_execution_mode_t active_mode; /* Dynamic auto-tiering state */
} aether_priority_mgr_t;

/* Initialize priority manager for given model configuration and tier budgets */
bool aether_priority_init(aether_priority_mgr_t *mgr, const aether_config_t *cfg,
                          size_t vram_budget, size_t ram_budget);

/* Record an activation of a layer / expert during inference */
void aether_priority_record_access(aether_priority_mgr_t *mgr, aether_comp_type_t type,
                                   int layer, int expert);

/* Recompute priority scores and sort components into priority rankings */
void aether_priority_recompute_ranks(aether_priority_mgr_t *mgr);

/* Rebalance physical tier allocations (promotes high-rank to VRAM/RAM, demotes low-rank to SSD) */
void aether_priority_rebalance_tiers(aether_priority_mgr_t *mgr);

/* Print priority report showing ranked placement of components across VRAM, RAM, and NVMe */
void aether_priority_print_report(const aether_priority_mgr_t *mgr);

/* Save / load priority profiles for specific prompt distributions */
bool aether_priority_save_profile(const aether_priority_mgr_t *mgr, const char *filepath);
bool aether_priority_load_profile(aether_priority_mgr_t *mgr, const char *filepath);

/* Free priority manager */
void aether_priority_free(aether_priority_mgr_t *mgr);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_PRIORITY_H */
