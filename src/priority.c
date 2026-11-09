#include "priority.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int compare_components_desc(const void *a, const void *b) {
    const aether_component_t *ca = (const aether_component_t *)a;
    const aether_component_t *cb = (const aether_component_t *)b;
    /* Pinned components always take top precedence */
    if (ca->is_pinned != cb->is_pinned) {
        return ca->is_pinned ? -1 : 1;
    }
    if (ca->priority_score > cb->priority_score) return -1;
    if (ca->priority_score < cb->priority_score) return 1;
    return 0;
}

bool aether_priority_init(aether_priority_mgr_t *mgr, const aether_config_t *cfg,
                          size_t vram_budget, size_t ram_budget) {
    memset(mgr, 0, sizeof(*mgr));
    mgr->vram_budget_bytes = vram_budget;
    mgr->ram_budget_bytes  = ram_budget;
    mgr->ema_decay         = 0.05f; /* Alpha = 0.05 */

    int num_comps = 2; /* Embedding + LM Head */
    if (cfg->n_experts > 1) {
        num_comps += cfg->n_layers * (1 + cfg->n_experts); /* 1 Attn + E experts per layer */
    } else {
        num_comps += cfg->n_layers * 2; /* Attn + MLP per layer */
    }

    mgr->total_components = num_comps;
    mgr->components = (aether_component_t*)calloc(num_comps, sizeof(aether_component_t));
    if (!mgr->components) return false;

    int idx = 0;
    size_t dim = cfg->dim;
    size_t q4_nb = dim / AETHER_BLOCK_SIZE;

    /* 1. Embedding component (Pinned to VRAM) */
    mgr->components[idx].id = idx;
    mgr->components[idx].type = COMP_EMBEDDING;
    mgr->components[idx].layer_idx = -1;
    mgr->components[idx].expert_idx = -1;
    mgr->components[idx].size_bytes = (size_t)cfg->vocab_size * dim * sizeof(float);
    mgr->components[idx].priority_score = 1000.0f;
    mgr->components[idx].is_pinned = true;
    mgr->components[idx].assigned_tier = TIER_VRAM;
    idx++;

    /* 2. LM Head component (Pinned to VRAM) */
    mgr->components[idx].id = idx;
    mgr->components[idx].type = COMP_LM_HEAD;
    mgr->components[idx].layer_idx = -1;
    mgr->components[idx].expert_idx = -1;
    mgr->components[idx].size_bytes = (size_t)cfg->vocab_size * q4_nb * sizeof(block_q4_0);
    mgr->components[idx].priority_score = 999.0f;
    mgr->components[idx].is_pinned = true;
    mgr->components[idx].assigned_tier = TIER_VRAM;
    idx++;

    /* 3. Layer components */
    size_t attn_bytes = dim * q4_nb * sizeof(block_q4_0) * 4; /* Wq, Wk, Wv, Wo */
    size_t expert_bytes = aether_calc_expert_bytes(cfg->dim, cfg->hidden_dim);

    for (int l = 0; l < cfg->n_layers; l++) {
        /* Attention component for layer l */
        mgr->components[idx].id = idx;
        mgr->components[idx].type = COMP_ATTENTION_LAYER;
        mgr->components[idx].layer_idx = l;
        mgr->components[idx].expert_idx = -1;
        mgr->components[idx].size_bytes = attn_bytes;
        /* Initial layers and final layers get structural priority boost */
        float layer_bonus = 0.0f;
        if (l < 2 || l >= cfg->n_layers - 2) layer_bonus = 50.0f;
        mgr->components[idx].priority_score = 100.0f + layer_bonus;
        mgr->components[idx].assigned_tier = TIER_VRAM;
        idx++;

        if (cfg->n_experts > 1) {
            for (int e = 0; e < cfg->n_experts; e++) {
                mgr->components[idx].id = idx;
                mgr->components[idx].type = COMP_MOE_EXPERT;
                mgr->components[idx].layer_idx = l;
                mgr->components[idx].expert_idx = e;
                mgr->components[idx].size_bytes = expert_bytes;
                mgr->components[idx].file_offset = aether_calc_expert_offset(cfg, l, e);
                /* Experts start with base priority */
                mgr->components[idx].priority_score = 10.0f;
                mgr->components[idx].assigned_tier = TIER_NVME;
                idx++;
            }
        } else {
            mgr->components[idx].id = idx;
            mgr->components[idx].type = COMP_HOT_NEURONS;
            mgr->components[idx].layer_idx = l;
            mgr->components[idx].expert_idx = -1;
            mgr->components[idx].size_bytes = expert_bytes;
            mgr->components[idx].priority_score = 50.0f;
            mgr->components[idx].assigned_tier = TIER_RAM;
            idx++;
        }
    }

    aether_priority_recompute_ranks(mgr);
    aether_priority_rebalance_tiers(mgr);
    return true;
}

void aether_priority_record_access(aether_priority_mgr_t *mgr, aether_comp_type_t type,
                                   int layer, int expert) {
    if (!mgr || !mgr->components) return;

    for (int i = 0; i < mgr->total_components; i++) {
        aether_component_t *c = &mgr->components[i];
        if (c->type == type && c->layer_idx == layer && c->expert_idx == expert) {
            c->total_hits++;
            c->ema_frequency = (1.0f - mgr->ema_decay) * c->ema_frequency + mgr->ema_decay * 1.0f;
            return;
        }
    }
}

void aether_priority_recompute_ranks(aether_priority_mgr_t *mgr) {
    if (!mgr || !mgr->components) return;

    /* Compute updated dynamic priority scores */
    for (int i = 0; i < mgr->total_components; i++) {
        aether_component_t *c = &mgr->components[i];
        if (c->is_pinned) continue;

        /* Score = base + EMA frequency * 500.0 + total_hits * 2.0 */
        float base = (c->type == COMP_ATTENTION_LAYER) ? 50.0f : 10.0f;
        if (c->layer_idx < 2 || c->layer_idx >= 14) base += 20.0f;
        c->priority_score = base + (c->ema_frequency * 500.0f) + (float)c->total_hits * 2.0f;
    }

    /* Sort components */
    qsort(mgr->components, mgr->total_components, sizeof(aether_component_t), compare_components_desc);

    /* Assign priority ranks: 1, 2, ..., N */
    for (int i = 0; i < mgr->total_components; i++) {
        mgr->components[i].priority_rank = i + 1;
    }
}

void aether_priority_rebalance_tiers(aether_priority_mgr_t *mgr) {
    if (!mgr || !mgr->components) return;

    size_t vram_used = 0;
    size_t ram_used = 0;

    for (int i = 0; i < mgr->total_components; i++) {
        aether_component_t *c = &mgr->components[i];

        if (c->is_pinned) {
            c->assigned_tier = TIER_VRAM;
            vram_used += c->size_bytes;
            continue;
        }

        /* Prefer VRAM if budget permits */
        if (vram_used + c->size_bytes <= mgr->vram_budget_bytes) {
            c->assigned_tier = TIER_VRAM;
            vram_used += c->size_bytes;
        }
        /* Else prefer Host System RAM */
        else if (ram_used + c->size_bytes <= mgr->ram_budget_bytes) {
            c->assigned_tier = TIER_RAM;
            ram_used += c->size_bytes;
        }
        /* Else reside on fast NVMe SSD */
        else {
            c->assigned_tier = TIER_NVME;
        }
    }

    mgr->vram_used_bytes = vram_used;
    mgr->ram_used_bytes  = ram_used;
}

void aether_priority_print_report(const aether_priority_mgr_t *mgr) {
    if (!mgr) return;

    int n_vram = 0, n_ram = 0, n_nvme = 0;
    for (int i = 0; i < mgr->total_components; i++) {
        if (mgr->components[i].assigned_tier == TIER_VRAM) n_vram++;
        else if (mgr->components[i].assigned_tier == TIER_RAM) n_ram++;
        else n_nvme++;
    }

    printf("\n=================================================================\n");
    printf("         LAYER & COMPONENT PRIORITY PLACEMENT REPORT             \n");
    printf("=================================================================\n");
    printf("• Tier 0 (RTX 4050 VRAM): %2d components (Allocated: %.2f / %.2f GiB)\n",
           n_vram, (double)mgr->vram_used_bytes / (1024*1024*1024),
           (double)mgr->vram_budget_bytes / (1024*1024*1024));
    printf("• Tier 1 (Host DDR5 RAM): %2d components (Allocated: %.2f / %.2f GiB)\n",
           n_ram, (double)mgr->ram_used_bytes / (1024*1024*1024),
           (double)mgr->ram_budget_bytes / (1024*1024*1024));
    printf("• Tier 2 (WD SN740 NVMe): %2d components (On-Demand Direct Streaming)\n", n_nvme);
    printf("-----------------------------------------------------------------\n");
    printf(" Rank | Component Type       | Layer | Expert | Hits | Score  | Tier \n");
    printf("-----------------------------------------------------------------\n");

    int display_count = mgr->total_components < 16 ? mgr->total_components : 16;
    for (int i = 0; i < display_count; i++) {
        const aether_component_t *c = &mgr->components[i];
        const char *type_name = "UNKNOWN";
        switch (c->type) {
            case COMP_EMBEDDING:        type_name = "Embedding Head "; break;
            case COMP_LM_HEAD:          type_name = "LM Head Logits "; break;
            case COMP_ATTENTION_LAYER:  type_name = "Self-Attention "; break;
            case COMP_MOE_EXPERT:       type_name = "MoE Expert     "; break;
            case COMP_HOT_NEURONS:      type_name = "Hot MLP Neurons"; break;
        }

        const char *tier_name = (c->assigned_tier == TIER_VRAM) ? "VRAM" :
                                (c->assigned_tier == TIER_RAM)  ? "RAM " : "NVMe";

        printf(" #%2d  | %s |  %2d   |   %2d   | %4lu | %6.1f | %s\n",
               c->priority_rank, type_name, c->layer_idx, c->expert_idx,
               c->total_hits, c->priority_score, tier_name);
    }
    if (mgr->total_components > display_count) {
        printf(" ...  | [%d lower-priority experts mapped to NVMe SSD on-demand] ...\n",
               mgr->total_components - display_count);
    }
    printf("=================================================================\n\n");
}

bool aether_priority_save_profile(const aether_priority_mgr_t *mgr, const char *filepath) {
    FILE *f = fopen(filepath, "wb");
    if (!f) return false;
    fwrite(&mgr->total_components, sizeof(int), 1, f);
    for (int i = 0; i < mgr->total_components; i++) {
        fwrite(&mgr->components[i].id, sizeof(int), 1, f);
        fwrite(&mgr->components[i].priority_rank, sizeof(int), 1, f);
        fwrite(&mgr->components[i].priority_score, sizeof(float), 1, f);
        fwrite(&mgr->components[i].total_hits, sizeof(uint64_t), 1, f);
    }
    fclose(f);
    return true;
}

bool aether_priority_load_profile(aether_priority_mgr_t *mgr, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return false;
    int count = 0;
    if (fread(&count, sizeof(int), 1, f) != 1 || count != mgr->total_components) {
        fclose(f);
        return false;
    }
    for (int i = 0; i < count; i++) {
        int id = 0, rank = 0;
        float score = 0.0f;
        uint64_t hits = 0;
        fread(&id, sizeof(int), 1, f);
        fread(&rank, sizeof(int), 1, f);
        fread(&score, sizeof(float), 1, f);
        fread(&hits, sizeof(uint64_t), 1, f);
        for (int k = 0; k < mgr->total_components; k++) {
            if (mgr->components[k].id == id) {
                mgr->components[k].priority_rank = rank;
                mgr->components[k].priority_score = score;
                mgr->components[k].total_hits = hits;
                break;
            }
        }
    }
    fclose(f);
    aether_priority_rebalance_tiers(mgr);
    return true;
}

void aether_priority_free(aether_priority_mgr_t *mgr) {
    if (mgr->components) {
        free(mgr->components);
        mgr->components = NULL;
    }
    mgr->total_components = 0;
}
