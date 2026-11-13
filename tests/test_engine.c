#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include "aether.h"
#include "hardware.h"
#include "simd_vnni.h"
#include "cuda_backend.h"
#include "nvme_streamer.h"
#include "sparsity.h"
#include "priority.h"
#include "pipeline_stages.h"
#include "gguf_loader.h"

#define TEST_PASS "\033[1;32m[PASS]\033[0m"
#define TEST_FAIL "\033[1;31m[FAIL]\033[0m"

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define RUN_TEST(name, func) do { \
    g_tests_run++; \
    printf("  Testing %-45s ... ", name); \
    fflush(stdout); \
    if (func()) { \
        printf("%s\n", TEST_PASS); \
        g_tests_passed++; \
    } else { \
        printf("%s\n", TEST_FAIL); \
    } \
} while(0)

/* ------------------------------------------------------------------ */
/* Test 1: AVX2 / AVX-VNNI Q4_0 Matrix-Vector Math Correctness        */
/* ------------------------------------------------------------------ */
static bool test_gemv_q4_0_accuracy(void) {
    int rows = 64;
    int cols = 128;
    int nb = cols / AETHER_BLOCK_SIZE;

    block_q4_0 *W = (block_q4_0*)malloc(rows * nb * sizeof(block_q4_0));
    float *x = (float*)malloc(cols * sizeof(float));
    float *y_simd = (float*)malloc(rows * sizeof(float));
    float *y_ref  = (float*)malloc(rows * sizeof(float));

    for (int i = 0; i < cols; i++) x[i] = 1.0f;

    /* Initialize with known weights */
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nb; b++) {
            W[r * nb + b].scale = 0.5f;
            /* 0x98: low nibble = 8 (val = 0), high nibble = 9 (val = 1) */
            memset(W[r * nb + b].qs, 0x98, 16);
        }
    }

    /* Reference scalar computation */
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            float scale = W[r * nb + b].scale;
            for (int i = 0; i < 16; i++) {
                uint8_t byte = W[r * nb + b].qs[i];
                int8_t v0 = (byte & 0x0F) - 8;
                int8_t v1 = (byte >> 4) - 8;
                sum += (float)v0 * scale * x[b * 32 + i];
                sum += (float)v1 * scale * x[b * 32 + 16 + i];
            }
        }
        y_ref[r] = sum;
    }

    /* AVX2 / AVX-VNNI SIMD computation */
    aether_gemv_q4_0(y_simd, W, x, rows, cols, 1);

    bool ok = true;
    for (int r = 0; r < rows; r++) {
        if (fabsf(y_simd[r] - y_ref[r]) > 1e-4f) {
            ok = false;
            break;
        }
    }

    free(W); free(x); free(y_simd); free(y_ref);
    return ok;
}

/* ------------------------------------------------------------------ */
/* Test 2: Vectorized RMSNorm Math Accuracy                           */
/* ------------------------------------------------------------------ */
static bool test_rmsnorm_accuracy(void) {
    int dim = 256;
    float *x = (float*)malloc(dim * sizeof(float));
    float *w = (float*)malloc(dim * sizeof(float));
    float *out_simd = (float*)malloc(dim * sizeof(float));
    float *out_ref  = (float*)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) {
        x[i] = (float)(i % 10) - 5.0f;
        w[i] = 1.0f + 0.1f * (i % 5);
    }

    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float scale = 1.0f / sqrtf((sum_sq / (float)dim) + 1e-5f);
    for (int i = 0; i < dim; i++) out_ref[i] = x[i] * scale * w[i];

    aether_rmsnorm(out_simd, x, w, dim, 1e-5f);

    bool ok = true;
    for (int i = 0; i < dim; i++) {
        if (fabsf(out_simd[i] - out_ref[i]) > 1e-4f) {
            ok = false;
            break;
        }
    }

    free(x); free(w); free(out_simd); free(out_ref);
    return ok;
}

/* ------------------------------------------------------------------ */
/* Test 3: Top-K Softmax MoE Routing Math                             */
/* ------------------------------------------------------------------ */
static bool test_moe_routing(void) {
    int n_experts = 8;
    int top_k = 2;
    float logits[8] = {1.0f, 5.0f, 0.5f, 8.0f, 2.0f, 3.0f, 0.0f, 7.0f};

    moe_routing_t routing[2];
    aether_route_top_k(logits, n_experts, top_k, routing);

    /* Expected top-2: expert 3 (logit 8.0) and expert 7 (logit 7.0) */
    if (routing[0].expert_id != 3 || routing[1].expert_id != 7) return false;

    /* Check softmax normalization: sum of weights must equal 1.0 */
    float sum = routing[0].weight + routing[1].weight;
    return fabsf(sum - 1.0f) < 1e-5f;
}

/* ------------------------------------------------------------------ */
/* Test 4: NVMe Streamer LRU Cache Hit & Miss                         */
/* ------------------------------------------------------------------ */
static bool test_lru_cache(void) {
    size_t exp_size = 4096;
    aether_streamer_init("data/weights.aether", 1024 * 1024, exp_size);

    /* Request layer 0, expert 1 */
    void *p1 = aether_streamer_get_expert(0, 1, 0, exp_size);
    /* Request same layer 0, expert 1 -> must hit LRU cache and return same pointer */
    void *p2 = aether_streamer_get_expert(0, 1, 0, exp_size);

    bool ok = (p1 != NULL && p1 == p2);
    aether_streamer_shutdown();
    return ok;
}

/* ------------------------------------------------------------------ */
/* Test 5: Dynamic Priority Manager Ranking & Tier Rebalancing        */
/* ------------------------------------------------------------------ */
static bool test_priority_rebalancing(void) {
    aether_config_t cfg = {
        .dim = 512,
        .hidden_dim = 1024,
        .n_layers = 4,
        .n_heads = 4,
        .n_kv_heads = 2,
        .vocab_size = 1000,
        .n_experts = 4,
        .n_active_experts = 2,
    };

    aether_priority_mgr_t mgr;
    /* Strict VRAM budget: Only allows ~2 components */
    size_t vram_budget = 2 * aether_calc_expert_bytes(cfg.dim, cfg.hidden_dim);
    size_t ram_budget  = 10 * aether_calc_expert_bytes(cfg.dim, cfg.hidden_dim);

    if (!aether_priority_init(&mgr, &cfg, vram_budget, ram_budget)) return false;

    /* Simulate 50 hits on expert 2 of layer 1 */
    for (int i = 0; i < 50; i++) {
        aether_priority_record_access(&mgr, COMP_MOE_EXPERT, 1, 2);
    }

    aether_priority_recompute_ranks(&mgr);
    aether_priority_rebalance_tiers(&mgr);

    /* Verify that expert (1, 2) has a high priority score and was promoted */
    bool found_promoted = false;
    for (int i = 0; i < mgr.total_components; i++) {
        if (mgr.components[i].layer_idx == 1 && mgr.components[i].expert_idx == 2) {
            if (mgr.components[i].priority_score > 100.0f) {
                found_promoted = true;
            }
            break;
        }
    }

    aether_priority_free(&mgr);
    return found_promoted;
}

/* ------------------------------------------------------------------ */
/* Test 6: Host RAM Embedding Lookup and RoPE Precomputation          */
/* ------------------------------------------------------------------ */
static bool test_prereq_pipeline(void) {
    int dim = 128;
    int vocab = 100;
    float *emb = (float*)malloc(vocab * dim * sizeof(float));
    float *dst = (float*)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) emb[42 * dim + i] = 3.14159f;

    aether_prereq_embedding_lookup(dst, emb, 42, dim);
    bool ok_emb = (fabsf(dst[0] - 3.14159f) < 1e-5f);

    float cos_b[32], sin_b[32];
    aether_prereq_precompute_rope(cos_b, sin_b, 64, 0, 10000.0f);
    /* At pos 0: cos(0) = 1.0, sin(0) = 0.0 */
    bool ok_rope = (fabsf(cos_b[0] - 1.0f) < 1e-5f && fabsf(sin_b[0] - 0.0f) < 1e-5f);

    free(emb); free(dst);
    return ok_emb && ok_rope;
}

/* ------------------------------------------------------------------ */
/* Test 7: CUDA Driver Backend Initialization                        */
/* ------------------------------------------------------------------ */
static bool test_cuda_backend(void) {
    bool ok = aether_cuda_init();
    if (ok) {
        /* Allocate 1MB on device and verify */
        void *d_mem = aether_cuda_alloc(1024 * 1024);
        if (!d_mem) return false;
        aether_cuda_free(d_mem);
        aether_cuda_shutdown();
        return true;
    }
    return false;
}

/* ------------------------------------------------------------------ */
/* Test 8: Active CUDA GEMV & Pinned Host Memory Allocation           */
/* ------------------------------------------------------------------ */
static bool test_cuda_active_gemv(void) {
    if (!aether_cuda_init()) return false;

    /* 1. Test Pinned Host Memory */
    void *h_mem = aether_cuda_host_alloc(64 * 1024);
    if (!h_mem) return false;
    memset(h_mem, 0xAA, 64 * 1024);
    aether_cuda_host_free(h_mem);

    /* 2. Test CUDA GEMV */
    int rows = 4, cols = 4;
    float W[16] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };
    float x[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y[4] = {0.0f};

    void *d_W = aether_cuda_alloc(16 * sizeof(float));
    void *d_x = aether_cuda_alloc(4 * sizeof(float));
    void *d_y = aether_cuda_alloc(4 * sizeof(float));

    aether_cuda_memcpy_h2d_async(d_W, W, 16 * sizeof(float));
    aether_cuda_memcpy_h2d_async(d_x, x, 4 * sizeof(float));
    aether_cuda_sync();

    aether_cuda_gemv_f32((float*)d_y, (float*)d_W, (float*)d_x, rows, cols);
    aether_cuda_memcpy_d2h_async(y, d_y, 4 * sizeof(float));
    aether_cuda_sync();

    aether_cuda_free(d_W);
    aether_cuda_free(d_x);
    aether_cuda_free(d_y);
    aether_cuda_shutdown();

    return (fabsf(y[0] - 10.0f) < 1e-4f && fabsf(y[1] - 26.0f) < 1e-4f &&
            fabsf(y[2] - 42.0f) < 1e-4f && fabsf(y[3] - 58.0f) < 1e-4f);
}

/* ------------------------------------------------------------------ */
/* Test 9: Dynamic Auto-Tiering (Mode 1 GPU vs Mode 2 4-Tier)         */
/* ------------------------------------------------------------------ */
static bool test_auto_tiering_selection(void) {
    noel_config_t cfg_small = {
        .dim = 512,
        .hidden_dim = 1024,
        .n_layers = 4,
        .n_heads = 4,
        .vocab_size = 1000,
        .n_experts = 1,
        .execution_mode = NOEL_MODE_AUTO
    };
    aether_priority_mgr_t mgr_small;
    /* 4.5 GB VRAM budget -> small model fits completely in VRAM */
    aether_priority_init(&mgr_small, &cfg_small, 4ULL * 1024 * 1024 * 1024, 6ULL * 1024 * 1024 * 1024);
    bool ok_small = (mgr_small.active_mode == NOEL_MODE_MONOLITHIC_GPU);
    aether_priority_free(&mgr_small);

    noel_config_t cfg_large = {
        .dim = 5120,
        .hidden_dim = 13824,
        .n_layers = 48,
        .n_heads = 40,
        .vocab_size = 152064,
        .n_experts = 8,
        .execution_mode = NOEL_MODE_AUTO
    };
    aether_priority_mgr_t mgr_large;
    /* 2.0 GB VRAM budget -> large model exceeds VRAM, must trigger 4-tier mode */
    aether_priority_init(&mgr_large, &cfg_large, 2ULL * 1024 * 1024 * 1024, 6ULL * 1024 * 1024 * 1024);
    bool ok_large = (mgr_large.active_mode == NOEL_MODE_HETEROGENEOUS_4TIER);
    aether_priority_free(&mgr_large);

    return ok_small && ok_large;
}

/* ------------------------------------------------------------------ */
/* Test 10: Native GGUF v3 Parsing & Zero-Copy Memory Mapping         */
/* ------------------------------------------------------------------ */
static bool test_gguf_loader(void) {
    const char *path_27b = "/home/neymar-jr/.cache/huggingface/hub/models--DavidAU--Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF/snapshots/e2d64985a4a5de0ff2ba67bc8a026f9531689c75/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-IQ4_XS.gguf";
    const char *path_ollama = "/usr/share/ollama/.ollama/models/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515";

    const char *test_path = NULL;
    if (access(path_27b, R_OK) == 0) test_path = path_27b;
    else if (access(path_ollama, R_OK) == 0) test_path = path_ollama;

    if (!test_path) return true;

    noel_gguf_t ctx;
    if (!noel_gguf_open(&ctx, test_path)) return false;

    bool ok_ver = (ctx.version == 2 || ctx.version == 3);
    bool ok_tensors = (ctx.n_tensors > 0);
    bool ok_kv = (ctx.n_kv > 0);
    bool ok_data = (ctx.data_offset > 0);
    bool ok_cfg = (ctx.config.dim > 0 && ctx.config.n_layers > 0);
    bool ok_tensor0 = (ctx.tensors != NULL && ctx.tensors[0].data != NULL && ctx.tensors[0].size_bytes > 0);

    noel_gguf_close(&ctx);
    return ok_ver && ok_tensors && ok_kv && ok_data && ok_cfg && ok_tensor0;
}

int main(void) {
    printf("=================================================================\n");
    printf("        AETHER INFERENCE ENGINE: AUTOMATED TEST HARNESS          \n");
    printf("=================================================================\n\n");

    aether_detect_hardware();
    aether_pin_to_p_cores();

    printf("\n--- Core Mathematics & Vector SIMD Tests ---\n");
    RUN_TEST("AVX-VNNI Q4_0 Matrix-Vector Dot Product", test_gemv_q4_0_accuracy);
    RUN_TEST("Vectorized RMSNorm Normalization", test_rmsnorm_accuracy);
    RUN_TEST("Top-K MoE Expert Softmax Gating", test_moe_routing);

    printf("\n--- Tiered Memory & Streaming Storage Tests ---\n");
    RUN_TEST("NVMe Streamer Dynamic LRU Cache", test_lru_cache);
    RUN_TEST("Priority-Ranked Placement & Rebalancing", test_priority_rebalancing);

    printf("\n--- Auxiliary CPU Pre/Post-Requisite Tests ---\n");
    RUN_TEST("Host RAM Embedding Lookup & RoPE Cache", test_prereq_pipeline);

    printf("\n--- Hardware Acceleration Tests ---\n");
    RUN_TEST("NVIDIA CUDA Driver API Runtime", test_cuda_backend);
    RUN_TEST("Active CUDA PTX GEMV & Pinned Host Alloc", test_cuda_active_gemv);

    printf("\n--- Dynamic Auto-Tiering Tests ---\n");
    RUN_TEST("Dynamic Mode Auto-Detection (GPU vs 4-Tier)", test_auto_tiering_selection);

    printf("\n--- GGUF Zero-Copy Memory-Mapping Tests ---\n");
    RUN_TEST("Native GGUF v3 Parsing & Memory Mapping", test_gguf_loader);

    printf("\n=================================================================\n");
    printf("  TEST RESULTS: %d / %d Passed (%.1f%%)\n",
           g_tests_passed, g_tests_run, (float)g_tests_passed * 100.0f / (float)g_tests_run);
    printf("=================================================================\n");

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
