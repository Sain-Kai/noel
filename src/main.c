#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine.h"
#include "gguf_loader.h"

static void print_banner(void) {
    printf("=================================================================\n");
    printf("   NOEL INFERENCE ENGINE: HETEROGENEOUS TIERED INFERENCE         \n");
    printf("   Optimized for Heterogeneous CPU (AVX-VNNI) + CUDA GPU + NVMe  \n");
    printf("=================================================================\n\n");
}

static bool repl_stream_cb(int32_t token_id, const char *piece, void *user_data) {
    (void)token_id;
    (void)user_data;
    if (piece && piece[0]) {
        fputs(piece, stdout);
        fflush(stdout);
    }
    return true;
}

static void run_interactive_repl(aether_engine_t *engine) {
    printf("\n=================================================================\n");
    printf("   NOEL INTERACTIVE CONVERSATION REPL (noel run)                 \n");
    printf("   Model: %-22s | SeqLen: %-5d\n",
           engine->gguf.model_name[0] ? engine->gguf.model_name : "Native GGUF",
           engine->config.seq_len);
    printf("   Params: Temp=%.2f, Top-P=%.2f, Top-K=%d, Rep=%.2f\n",
           engine->sampler.config.temperature,
           engine->sampler.config.top_p,
           engine->sampler.config.top_k,
           engine->sampler.config.repetition_penalty);
    printf("   Commands: /exit, /clear, /stats, /params, /temp, /top_p, /top_k, /rep, /graph\n");
    printf("=================================================================\n\n");

    char line[4096];
    char formatted_prompt[4096 + 128];

    while (1) {
        printf("\033[1;36m>>> \033[0m");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) {
            printf("\n");
            break;
        }

        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) continue;

        if (strcmp(line, "/exit") == 0 || strcmp(line, "/quit") == 0 || strcmp(line, "exit") == 0) {
            printf("Exiting Noel REPL. Goodbye!\n");
            break;
        }

        if (strcmp(line, "/clear") == 0 || strcmp(line, "/reset") == 0) {
            aether_engine_reset_context(engine);
            printf("\033[1;33m[Context reset to position 0]\033[0m\n\n");
            continue;
        }

        if (strcmp(line, "/stats") == 0) {
            printf("• Active Context Pos:  %d / %d tokens\n", engine->cur_pos, engine->config.seq_len);
            printf("• Engine Throughput:   %.2f tokens/second\n", engine->stats.tokens_per_sec);
            printf("• Graph Nodes:         %d compiled\n\n", engine->cgraph.node_count);
            continue;
        }

        if (strcmp(line, "/graph") == 0) {
            noel_cgraph_print(&engine->cgraph);
            printf("\n");
            continue;
        }

        if (strcmp(line, "/params") == 0) {
            printf("• Temperature:         %.2f\n", engine->sampler.config.temperature);
            printf("• Top-P (Nucleus):     %.2f\n", engine->sampler.config.top_p);
            printf("• Top-K Candidates:    %d\n", engine->sampler.config.top_k);
            printf("• Repetition Penalty:  %.2f\n\n", engine->sampler.config.repetition_penalty);
            continue;
        }

        if (strncmp(line, "/temp ", 6) == 0) {
            float val = (float)atof(line + 6);
            engine->sampler.config.temperature = val;
            printf("\033[1;33m[Sampling temperature updated to %.2f]\033[0m\n\n", val);
            continue;
        }

        if (strncmp(line, "/top_p ", 7) == 0 || strncmp(line, "/topp ", 6) == 0) {
            const char *arg = (line[5] == 'p') ? line + 7 : line + 6;
            float val = (float)atof(arg);
            engine->sampler.config.top_p = val;
            printf("\033[1;33m[Nucleus Top-P threshold updated to %.2f]\033[0m\n\n", val);
            continue;
        }

        if (strncmp(line, "/top_k ", 7) == 0 || strncmp(line, "/topk ", 6) == 0) {
            const char *arg = (line[5] == 'k') ? line + 7 : line + 6;
            int val = atoi(arg);
            engine->sampler.config.top_k = val;
            printf("\033[1;33m[Top-K candidate limit updated to %d]\033[0m\n\n", val);
            continue;
        }

        if (strncmp(line, "/rep ", 5) == 0) {
            float val = (float)atof(line + 5);
            engine->sampler.config.repetition_penalty = val;
            printf("\033[1;33m[Repetition penalty updated to %.2f]\033[0m\n\n", val);
            continue;
        }

        if (strcmp(line, "/help") == 0) {
            printf("Available REPL commands:\n");
            printf("  /temp <float>     Set sampling temperature (0.0 = greedy argmax)\n");
            printf("  /top_p <float>    Set nucleus sampling Top-P threshold (0.0 - 1.0)\n");
            printf("  /top_k <int>      Set Top-K candidate filtering limit\n");
            printf("  /rep <float>      Set repetition penalty (e.g. 1.1)\n");
            printf("  /params           Display active sampling parameters\n");
            printf("  /graph            Display compiled Noel computation graph IR\n");
            printf("  /stats            Display context position and token throughput\n");
            printf("  /clear            Reset conversational context to position 0\n");
            printf("  /exit             Exit REPL shell\n\n");
            continue;
        }

        /* Format prompt for instruct chat if ChatML markers exist in vocabulary */
        bool has_chatml = (aether_tokenizer_lookup(&engine->tokenizer, "<|im_start|>", 12) >= 0);
        if (has_chatml) {
            snprintf(formatted_prompt, sizeof(formatted_prompt),
                     "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", line);
        } else {
            snprintf(formatted_prompt, sizeof(formatted_prompt), "%s ", line);
        }

        printf("\033[1;32m");
        aether_engine_generate_stream(engine, formatted_prompt, 128,
                                      engine->sampler.config.temperature,
                                      engine->sampler.config.top_p,
                                      repl_stream_cb, NULL);
        printf("\033[0m\n\n");
    }
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

    const char *model_path = "data/weights.noel";
    const char *prompt = "In the deep expanse of space, an autonomous starship";
    int max_gen_tokens = 32;
    bool is_repl = false;
    float opt_temp = 0.0f;
    float opt_top_p = 0.9f;
    int opt_top_k = 40;
    float opt_rep_penalty = 1.1f;
    bool show_graph = false;

    /* Check for `noel run <model>` command */
    if (argc >= 2 && (strcmp(argv[1], "run") == 0 || strcmp(argv[1], "chat") == 0)) {
        is_repl = true;
        if (argc >= 3 && argv[2][0] != '-') {
            model_path = argv[2];
        }
    }

    /* First pass: extract --model or positional path to auto-configure architecture */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
            break;
        } else if (strcmp(argv[i], "--interactive") == 0 || strcmp(argv[i], "-i") == 0) {
            is_repl = true;
        }
    }

    if (model_path) {
        noel_gguf_t gguf;
        if (noel_gguf_open(&gguf, model_path)) {
            noel_gguf_print_summary(&gguf);
            cfg = gguf.config;
            noel_gguf_close(&gguf);
        }
    }

    /* Second pass: handle overrides */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            cfg.dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden-dim") == 0 && i + 1 < argc) {
            cfg.hidden_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            cfg.n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            cfg.n_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--kv-heads") == 0 && i + 1 < argc) {
            cfg.n_kv_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            cfg.vocab_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--experts") == 0 && i + 1 < argc) {
            cfg.n_experts = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            max_gen_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            opt_temp = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            opt_top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            opt_top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) {
            opt_rep_penalty = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--graph") == 0) {
            show_graph = true;
        } else if (strcmp(argv[i], "--interactive") == 0 || strcmp(argv[i], "-i") == 0) {
            is_repl = true;
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
    if (!aether_engine_init(&engine, &cfg, model_path)) {
        fprintf(stderr, "Failed to initialize Noel Engine\n");
        return 1;
    }

    engine.sampler.config.temperature = opt_temp;
    engine.sampler.config.top_p = opt_top_p;
    engine.sampler.config.top_k = opt_top_k;
    engine.sampler.config.repetition_penalty = opt_rep_penalty;

    if (show_graph) {
        noel_cgraph_print(&engine.cgraph);
    }

    if (is_repl) {
        run_interactive_repl(&engine);
    } else {
        aether_engine_generate(&engine, prompt, max_gen_tokens);
    }

    aether_engine_shutdown(&engine);
    return 0;
}
