#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine.h"

static void print_banner(void) {
    printf("=================================================================\n");
    printf("   NOEL INFERENCE ENGINE: HETEROGENEOUS TIERED INFERENCE         \n");
    printf("   Optimized for Intel Raptor Lake (AVX-VNNI) + RTX 4050 + NVMe  \n");
    printf("=================================================================\n\n");
}

int main(int argc, char **argv) {
    print_banner();

    /* Default configuration: 3.5B / 7B style MoE model */
    noel_config_t cfg = {
        .dim = 2048,
        .hidden_dim = 5632,
        .n_layers = 16,
        .n_heads = 16,
        .n_kv_heads = 4,
        .vocab_size = 32000,
        .seq_len = 1024,
        .n_experts = 8,           /* MoE (Kimi-k3 style) */
        .n_active_experts = 2,    /* Top-2 active per token */
        .hot_ratio = 0.20f,       /* 20% hot neurons in VRAM (PowerInfer style) */
        .norm_eps = 1e-5f,
        .rope_theta = 10000.0f,
        .execution_mode = NOEL_MODE_AUTO
    };

    const char *prompt = "In the deep expanse of space, an autonomous starship";
    int max_gen_tokens = 32;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            cfg.dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            cfg.n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
            cfg.n_experts = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            max_gen_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (strcmp(m, "gpu") == 0 || strcmp(m, "vram") == 0) {
                cfg.execution_mode = NOEL_MODE_MONOLITHIC_GPU;
            } else if (strcmp(m, "heterogeneous") == 0 || strcmp(m, "4tier") == 0) {
                cfg.execution_mode = NOEL_MODE_HETEROGENEOUS_4TIER;
            } else {
                cfg.execution_mode = NOEL_MODE_AUTO;
            }
        }
    }

    aether_engine_t engine;
    if (!aether_engine_init(&engine, &cfg, "data/weights.noel")) {
        fprintf(stderr, "Failed to initialize Noel Engine\n");
        return 1;
    }

    aether_engine_generate(&engine, prompt, max_gen_tokens);

    aether_engine_shutdown(&engine);
    return 0;
}
