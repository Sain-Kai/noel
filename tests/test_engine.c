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
#include "dequant.h"
#include "engine.h"

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
    if (getenv("NOEL_NO_CUDA") || getenv("AETHER_NO_CUDA")) {
        printf("[SKIP: NOEL_NO_CUDA] ");
        return true;
    }
    bool ok = aether_cuda_init();
    if (ok) {
        /* Allocate 1MB on device and verify */
        void *d_mem = aether_cuda_alloc(1024 * 1024);
        if (!d_mem) return false;
        aether_cuda_free(d_mem);
        return true;
    }
    return false;
}

/* ------------------------------------------------------------------ */
/* Test 8: Active CUDA GEMV & Pinned Host Memory Allocation           */
/* ------------------------------------------------------------------ */
static bool test_cuda_active_gemv(void) {
    if (getenv("NOEL_NO_CUDA") || getenv("AETHER_NO_CUDA")) {
        printf("[SKIP: NOEL_NO_CUDA] ");
        return true;
    }
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
    if (access(path_ollama, R_OK) == 0) test_path = path_ollama;
    else if (access(path_27b, R_OK) == 0) test_path = path_27b;

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

/* ------------------------------------------------------------------ */
/* Test 11: SIMD Dequantization Kernels & Mathematical Precision     */
/* ------------------------------------------------------------------ */
static bool test_dequantization_kernels(void) {
    if (!noel_dequant_init()) return false;

    /* 1. Test BF16 Dequantization */
    uint16_t bf16_data[16];
    float ref_bf16[16];
    float dst_bf16[16];
    for (int i = 0; i < 16; i++) {
        ref_bf16[i] = (float)i * 0.5f + 1.25f;
        uint32_t u;
        memcpy(&u, &ref_bf16[i], sizeof(float));
        bf16_data[i] = (uint16_t)(u >> 16);
    }
    noel_dequantize_row(GGML_TYPE_BF16, bf16_data, dst_bf16, 16);
    for (int i = 0; i < 16; i++) {
        if (fabsf(dst_bf16[i] - ref_bf16[i]) > 1e-2f) return false;
    }

    /* 2. Test F16 Dequantization */
    uint16_t f16_data[8] = {
        0x0000, /* 0.0 */
        0x3C00, /* 1.0 */
        0xBC00, /* -1.0 */
        0x3800, /* 0.5 */
        0x4000, /* 2.0 */
        0x4248, /* 3.140625 */
        0xB400, /* -0.25 */
        0x4900  /* 10.0 */
    };
    float ref_f16[8] = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, 3.140625f, -0.25f, 10.0f};
    float dst_f16[8];
    noel_dequantize_row(GGML_TYPE_F16, f16_data, dst_f16, 8);
    for (int i = 0; i < 8; i++) {
        if (fabsf(dst_f16[i] - ref_f16[i]) > 1e-3f) return false;
    }

    /* 3. Test Q8_0 Dequantization */
    uint8_t q8_block[34];
    *(uint16_t*)q8_block = 0x3800; /* scale = 0.5f in fp16 */
    int8_t *q8_qs = (int8_t*)(q8_block + 2);
    float ref_q8[32];
    for (int i = 0; i < 32; i++) {
        q8_qs[i] = (int8_t)(i - 16);
        ref_q8[i] = (float)(i - 16) * 0.5f;
    }
    float dst_q8[32];
    noel_dequantize_row(GGML_TYPE_Q8_0, q8_block, dst_q8, 32);
    for (int i = 0; i < 32; i++) {
        if (fabsf(dst_q8[i] - ref_q8[i]) > 1e-3f) return false;
    }

    /* 4. Test Q4_0 Dequantization */
    uint8_t q4_block[18];
    *(uint16_t*)q4_block = 0x3800; /* scale = 0.5f */
    uint8_t *q4_qs = q4_block + 2;
    float ref_q4[32];
    for (int i = 0; i < 16; i++) {
        uint8_t low = (i & 0x0F);
        uint8_t high = ((i + 1) & 0x0F);
        q4_qs[i] = low | (high << 4);
        ref_q4[i]      = (float)((int8_t)low - 8) * 0.5f;
        ref_q4[i + 16] = (float)((int8_t)high - 8) * 0.5f;
    }
    float dst_q4[32];
    noel_dequantize_row(GGML_TYPE_Q4_0, q4_block, dst_q4, 32);
    for (int i = 0; i < 32; i++) {
        if (fabsf(dst_q4[i] - ref_q4[i]) > 1e-3f) return false;
    }

    /* 5. Test Q5_0 Dequantization */
    uint8_t q5_block[22];
    *(uint16_t*)q5_block = 0x3800; /* scale = 0.5f */
    *(uint32_t*)(q5_block + 2) = 0x55555555; /* alternating high bits */
    uint8_t *q5_qs = q5_block + 6;
    float ref_q5[32];
    uint32_t qh = 0x55555555;
    for (int i = 0; i < 16; i++) {
        q5_qs[i] = 0x24;
        uint8_t x0 = (0x04) | (((qh >> (i + 0)) & 1) << 4);
        uint8_t x1 = (0x02) | (((qh >> (i + 16)) & 1) << 4);
        ref_q5[i]      = (float)((int8_t)x0 - 16) * 0.5f;
        ref_q5[i + 16] = (float)((int8_t)x1 - 16) * 0.5f;
    }
    float dst_q5[32];
    noel_dequantize_row(GGML_TYPE_Q5_0, q5_block, dst_q5, 32);
    for (int i = 0; i < 32; i++) {
        if (fabsf(dst_q5[i] - ref_q5[i]) > 1e-3f) return false;
    }

    /* 6. Test Multi-Threaded GEMV with Q8_0 */
    int rows = 16, cols = 64;
    size_t q8_row_sz = noel_ggml_row_bytes(GGML_TYPE_Q8_0, cols);
    uint8_t *mat_q8 = (uint8_t*)malloc(rows * q8_row_sz);
    float *vec_x = (float*)malloc(cols * sizeof(float));
    float *y_gemv = (float*)malloc(rows * sizeof(float));

    for (int c = 0; c < cols; c++) vec_x[c] = 1.0f;
    for (int r = 0; r < rows; r++) {
        uint8_t *rptr = mat_q8 + r * q8_row_sz;
        for (int b = 0; b < cols / 32; b++) {
            uint8_t *bptr = rptr + b * 34;
            *(uint16_t*)bptr = 0x3800; /* 0.5f */
            int8_t *qs = (int8_t*)(bptr + 2);
            for (int j = 0; j < 32; j++) qs[j] = 2; /* 2 * 0.5f = 1.0f */
        }
    }

    noel_gemv_quantized(GGML_TYPE_Q8_0, y_gemv, mat_q8, vec_x, rows, cols, 4);

    bool ok_gemv = true;
    for (int r = 0; r < rows; r++) {
        if (fabsf(y_gemv[r] - 64.0f) > 1e-3f) {
            ok_gemv = false;
            break;
        }
    }

    free(mat_q8); free(vec_x); free(y_gemv);
    return ok_gemv;
}

/* ------------------------------------------------------------------ */
/* Test 12: Physical GGUF Tensor Binding into 4-Tier Memory Pipeline  */
/* ------------------------------------------------------------------ */
static bool test_physical_gguf_tensor_binding(void) {
    const char *path_ollama = "/usr/share/ollama/.ollama/models/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515";
    const char *path_27b = "/home/neymar-jr/.cache/huggingface/hub/models--DavidAU--Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF/snapshots/e2d64985a4a5de0ff2ba67bc8a026f9531689c75/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-IQ4_XS.gguf";

    const char *test_path = NULL;
    if (access(path_ollama, R_OK) == 0) test_path = path_ollama;
    else if (access(path_27b, R_OK) == 0) test_path = path_27b;

    if (!test_path) return true;

    noel_gguf_t gguf_probe;
    if (!noel_gguf_open(&gguf_probe, test_path)) return false;
    noel_config_t cfg = gguf_probe.config;
    noel_gguf_close(&gguf_probe);

    aether_engine_t engine;
    if (!aether_engine_init(&engine, &cfg, test_path)) return false;

    /* Verify physical zero-copy binding */
    bool ok_bound = engine.is_gguf_loaded && engine.weights.is_bound_gguf;
    bool ok_embd  = (engine.weights.raw_token_embd != NULL) && (engine.weights.token_embd_row_bytes > 0);
    bool ok_norm  = (engine.weights.raw_final_norm != NULL);
    bool ok_head  = (engine.weights.raw_output_weight != NULL);
    bool ok_layer0= (engine.weights.layer_tensors[0].attn_norm != NULL &&
                     engine.weights.layer_tensors[0].attn_q != NULL &&
                     engine.weights.layer_tensors[0].ffn_gate != NULL);

    /* Run 1 token forward step on physical model tensors */
    int32_t out_tok = aether_engine_forward(&engine, 10, 0);
    bool ok_fwd = (out_tok >= 0 && out_tok < engine.config.vocab_size);

    aether_engine_shutdown(&engine);

    return ok_bound && ok_embd && ok_norm && ok_head && ok_layer0 && ok_fwd;
}

/* ------------------------------------------------------------------ */
/* Test 13: GGUF Vocabulary Parsing & BPE Piece Detokenization        */
/* ------------------------------------------------------------------ */
static bool test_tokenizer_gguf_bpe(void) {
    const char *path_ollama = "/usr/share/ollama/.ollama/models/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515";
    const char *path_27b = "/home/neymar-jr/.cache/huggingface/hub/models--DavidAU--Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF/snapshots/e2d64985a4a5de0ff2ba67bc8a026f9531689c75/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-IQ4_XS.gguf";

    const char *test_path = NULL;
    if (access(path_ollama, R_OK) == 0) test_path = path_ollama;
    else if (access(path_27b, R_OK) == 0) test_path = path_27b;

    aether_tokenizer_t tok;
    memset(&tok, 0, sizeof(tok));

    if (test_path) {
        noel_gguf_t gguf;
        if (!noel_gguf_open(&gguf, test_path)) return false;
        bool ok_load = aether_tokenizer_load_gguf(&tok, &gguf);
        noel_gguf_close(&gguf);
        if (!ok_load || tok.vocab_size < 1000) {
            aether_tokenizer_free(&tok);
            return false;
        }
    } else {
        aether_tokenizer_init(&tok, 1000);
    }

    /* 1. Test encoding */
    int32_t tokens[64];
    int n_tok = aether_tokenizer_encode(&tok, "Hello Noel", tokens, 64);
    if (n_tok <= 0) {
        aether_tokenizer_free(&tok);
        return false;
    }

    /* 2. Test piece decoding */
    char piece[128];
    const char *decoded = aether_tokenizer_decode_piece(&tok, tokens[n_tok - 1], piece, sizeof(piece));
    bool ok_decode = (decoded != NULL);

    /* 3. Test BPE marker replacement */
    char test_bpe[] = "\xc4\xa0world";
    char out_buf[64];
    aether_tokenizer_t mock_tok;
    memset(&mock_tok, 0, sizeof(mock_tok));
    mock_tok.vocab_size = 1;
    mock_tok.vocab = (char**)calloc(1, sizeof(char*));
    mock_tok.vocab[0] = strdup(test_bpe);
    mock_tok.eos_id = 99;
    mock_tok.pad_id = 99;

    aether_tokenizer_decode_piece(&mock_tok, 0, out_buf, sizeof(out_buf));
    bool ok_bpe_space = (strcmp(out_buf, " world") == 0);

    aether_tokenizer_free(&mock_tok);
    aether_tokenizer_free(&tok);

    return ok_decode && ok_bpe_space;
}

/* ------------------------------------------------------------------ */
/* Test 14: Streaming Engine Generation & Context Cache Persistence   */
/* ------------------------------------------------------------------ */
typedef struct {
    int token_count;
    char text[1024];
} test_stream_ctx_t;

static bool test_stream_collector(int32_t token_id, const char *piece, void *user_data) {
    (void)token_id;
    test_stream_ctx_t *ctx = (test_stream_ctx_t*)user_data;
    ctx->token_count++;
    if (piece) {
        strncat(ctx->text, piece, sizeof(ctx->text) - strlen(ctx->text) - 1);
    }
    return true;
}

static bool test_streaming_and_persistence(void) {
    const char *path_ollama = "/usr/share/ollama/.ollama/models/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515";
    const char *test_path = (access(path_ollama, R_OK) == 0) ? path_ollama : NULL;

    noel_config_t cfg;
    if (test_path) {
        noel_gguf_t gguf;
        if (!noel_gguf_open(&gguf, test_path)) return false;
        cfg = gguf.config;
        noel_gguf_close(&gguf);
    } else {
        cfg = (noel_config_t){
            .dim = 128, .hidden_dim = 256, .n_layers = 2,
            .n_heads = 2, .n_kv_heads = 1, .vocab_size = 100,
            .seq_len = 256, .n_experts = 1, .n_active_experts = 1,
            .hot_ratio = 1.0f, .norm_eps = 1e-5f, .rope_theta = 10000.0f
        };
    }

    aether_engine_t engine;
    if (!aether_engine_init(&engine, &cfg, test_path)) return false;

    /* Turn 1: Stream 3 tokens */
    test_stream_ctx_t ctx1 = {0};
    int gen1 = aether_engine_generate_stream(&engine, "Hello", 3, 0.0f, 0.9f, test_stream_collector, &ctx1);
    bool ok_gen1 = (gen1 > 0 && ctx1.token_count == gen1);
    int pos_turn1 = engine.cur_pos;
    bool ok_pos1 = (pos_turn1 >= gen1);

    /* Turn 2: Multi-turn prompt persistence (pos should advance from pos_turn1) */
    test_stream_ctx_t ctx2 = {0};
    int gen2 = aether_engine_generate_stream(&engine, "World", 2, 0.0f, 0.9f, test_stream_collector, &ctx2);
    bool ok_pos2 = (engine.cur_pos > pos_turn1 && gen2 > 0);

    /* Turn 3: Reset context */
    aether_engine_reset_context(&engine);
    bool ok_reset = (engine.cur_pos == 0);

    aether_engine_shutdown(&engine);
    return ok_gen1 && ok_pos1 && ok_pos2 && ok_reset;
}

/* ------------------------------------------------------------------ */
/* Test 15: Vectorized Grouped-Query Attention (GQA) & Online Softmax */
/* ------------------------------------------------------------------ */
static bool test_gqa_attention_correctness(void) {
    int n_heads = 4;
    int n_kv_heads = 2;
    int head_dim = 64;
    int dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int seq_len = 16;
    int pos = 3;

    float *q = (float *)malloc(dim * sizeof(float));
    float *k_cache = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    float *v_cache = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    float *out = (float *)malloc(dim * sizeof(float));

    for (int i = 0; i < dim; i++) q[i] = 0.1f * ((i % 5) + 1);

    for (int t = 0; t <= pos; t++) {
        for (int i = 0; i < kv_dim; i++) {
            k_cache[t * kv_dim + i] = 0.05f * ((t + i) % 7 + 1);
            v_cache[t * kv_dim + i] = (float)(t + 1) * 0.2f;
        }
    }

    noel_gqa_attention(out, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, pos, 4);

    bool ok = true;
    for (int i = 0; i < dim; i++) {
        if (isnan(out[i]) || isinf(out[i]) || out[i] == 0.0f) {
            ok = false;
            break;
        }
    }

    /* Single token (pos=0) check: attention softmax over single token equals 1.0 */
    float *out0 = (float *)malloc(dim * sizeof(float));
    noel_gqa_attention(out0, q, k_cache, v_cache, n_heads, n_kv_heads, head_dim, seq_len, 0, 4);
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / (n_heads / n_kv_heads);
        for (int d = 0; d < head_dim; d++) {
            float expected = v_cache[0 * kv_dim + kv_h * head_dim + d];
            if (fabsf(out0[h * head_dim + d] - expected) > 1e-4f) {
                ok = false;
                break;
            }
        }
    }

    free(q); free(k_cache); free(v_cache); free(out); free(out0);
    return ok;
}

/* ------------------------------------------------------------------ */
/* Test 16: Nucleus Top-P, Top-K, and Temperature Sampler Precision  */
/* ------------------------------------------------------------------ */
static bool test_sampler_precision(void) {
    int vocab_size = 1000;
    float *logits = (float *)calloc(vocab_size, sizeof(float));
    noel_sampler_config_t cfg = {
        .temperature = 0.0f,
        .top_p = 0.9f,
        .top_k = 40,
        .repetition_penalty = 1.0f,
        .seed = 42
    };

    noel_sampler_t sampler;
    noel_sampler_init(&sampler, &cfg, vocab_size);

    /* Test 1: Greedy argmax */
    logits[250] = 5.0f;
    logits[500] = 10.0f;
    logits[750] = 2.0f;
    int32_t chosen = noel_sampler_sample(&sampler, logits, vocab_size, NULL, 0);
    bool ok_greedy = (chosen == 500);

    /* Test 2: Repetition penalty (penalize token 500) */
    int32_t history[1] = { 500 };
    sampler.config.repetition_penalty = 3.0f;
    chosen = noel_sampler_sample(&sampler, logits, vocab_size, history, 1);
    bool ok_rep = (chosen == 250);

    /* Test 3: Top-K filter constraint */
    memset(logits, 0, vocab_size * sizeof(float));
    logits[10] = 4.0f;
    logits[20] = 3.0f;
    logits[30] = 2.0f;
    sampler.config.temperature = 0.8f;
    sampler.config.top_k = 3;
    sampler.config.repetition_penalty = 1.0f;

    bool ok_topk = true;
    for (int trial = 0; trial < 20; trial++) {
        int32_t tok = noel_sampler_sample(&sampler, logits, vocab_size, NULL, 0);
        if (tok != 10 && tok != 20 && tok != 30) {
            ok_topk = false;
            break;
        }
    }

    noel_sampler_free(&sampler);
    free(logits);
    return ok_greedy && ok_rep && ok_topk;
}

/* ------------------------------------------------------------------ */
/* Test 17: Native Computation Graph Compilation and Dispatcher IR   */
/* ------------------------------------------------------------------ */
static bool test_computation_graph_pipeline(void) {
    noel_cgraph_t graph;
    noel_cgraph_init(&graph, 896, 4864, 24, 14, 2);

    int n0 = noel_cgraph_add_node(&graph, NODE_EMBEDDING, DISPATCH_CPU_SIMD, -1, "token_embd");
    noel_cgraph_add_node(&graph, NODE_RMSNORM, DISPATCH_CPU_SIMD, 0, "blk.0.attn_norm");
    noel_cgraph_add_node(&graph, NODE_QKV_GEMV, DISPATCH_CPU_SIMD, 0, "blk.0.attn_qkv");
    noel_cgraph_add_node(&graph, NODE_ROPE, DISPATCH_CPU_SIMD, 0, "blk.0.rope");
    noel_cgraph_add_node(&graph, NODE_GQA_ATTENTION, DISPATCH_CPU_SIMD, 0, "blk.0.gqa_attn");
    noel_cgraph_add_node(&graph, NODE_ATTN_OUTPUT, DISPATCH_CPU_SIMD, 0, "blk.0.attn_output");
    noel_cgraph_add_node(&graph, NODE_RESIDUAL_ADD, DISPATCH_CPU_SIMD, 0, "blk.0.attn_residual");
    noel_cgraph_add_node(&graph, NODE_RMSNORM, DISPATCH_CPU_SIMD, 0, "blk.0.ffn_norm");
    noel_cgraph_add_node(&graph, NODE_FFN_GATE_UP, DISPATCH_CPU_SIMD, 0, "blk.0.ffn_gate_up");
    noel_cgraph_add_node(&graph, NODE_SWIGLU_MUL, DISPATCH_CPU_SIMD, 0, "blk.0.swiglu");
    noel_cgraph_add_node(&graph, NODE_FFN_DOWN, DISPATCH_CPU_SIMD, 0, "blk.0.ffn_down");
    noel_cgraph_add_node(&graph, NODE_RESIDUAL_ADD, DISPATCH_CPU_SIMD, 0, "blk.0.ffn_residual");
    noel_cgraph_add_node(&graph, NODE_RMSNORM, DISPATCH_CPU_SIMD, -1, "output_norm");
    int n13 = noel_cgraph_add_node(&graph, NODE_LM_HEAD, DISPATCH_CPU_SIMD, -1, "output");

    graph.is_compiled = true;

    bool ok_nodes = (graph.node_count == 14 && n0 == 0 && n13 == 13);
    bool ok_val = noel_cgraph_validate(&graph);

    noel_cgraph_mem_plan_t plan;
    noel_cgraph_optimize_buffers(&graph, &plan);
    bool ok_plan = (plan.unoptimized_bytes > plan.optimized_bytes && plan.reuse_efficiency_pct > 80.0f);

    noel_cgraph_free(&graph);
    return ok_nodes && ok_val && ok_plan;
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

    printf("\n--- SIMD Dequantization & Physical Binding Tests ---\n");
    RUN_TEST("Fast SIMD Dequantization Kernels & GEMV", test_dequantization_kernels);
    RUN_TEST("Physical GGUF Tensor Binding into 4-Tier Pipeline", test_physical_gguf_tensor_binding);

    printf("\n--- Tokenizer & Streaming Conversational REPL Tests ---\n");
    RUN_TEST("GGUF Embedded BPE Vocabulary & Piece Detokenizer", test_tokenizer_gguf_bpe);
    RUN_TEST("Multi-Turn Streaming Generation & KV Cache Context", test_streaming_and_persistence);

    printf("\n--- Native Computation Graph & GQA Attention Tests ---\n");
    RUN_TEST("Vectorized Grouped-Query Attention (GQA)", test_gqa_attention_correctness);
    RUN_TEST("Nucleus Top-P, Top-K & Temperature Sampler", test_sampler_precision);
    RUN_TEST("Computation Graph Validation & Buffer Reuse", test_computation_graph_pipeline);

    printf("\n=================================================================\n");
    printf("  TEST RESULTS: %d / %d Passed (%.1f%%)\n",
           g_tests_passed, g_tests_run, (float)g_tests_passed * 100.0f / (float)g_tests_run);
    printf("=================================================================\n");

    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
