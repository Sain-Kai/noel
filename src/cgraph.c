#include "cgraph.h"
#include "engine.h"
#include "pipeline_stages.h"
#include "simd_vnni.h"
#include "attention.h"
#include "dequant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

static inline double noel_cgraph_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec * 1e-3;
}

void noel_cgraph_init(noel_cgraph_t *graph, int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads) {
    if (!graph) return;
    memset(graph, 0, sizeof(*graph));
    graph->dim = dim;
    graph->hidden_dim = hidden_dim;
    graph->n_layers = n_layers;
    graph->n_heads = n_heads;
    graph->n_kv_heads = n_kv_heads;
    graph->head_dim = (n_heads > 0) ? (dim / n_heads) : 64;
    graph->node_count = 0;
    graph->is_compiled = false;
    graph->total_exec_time_us = 0.0;
}

int noel_cgraph_add_node(noel_cgraph_t *graph, noel_node_type_t type, noel_dispatch_tier_t tier, int layer_idx, const char *name) {
    if (!graph || graph->node_count >= NOEL_MAX_GRAPH_NODES) return -1;

    int idx = graph->node_count++;
    noel_cgraph_node_t *node = &graph->nodes[idx];
    memset(node, 0, sizeof(*node));
    node->id = idx;
    node->type = type;
    node->tier = tier;
    node->layer_idx = layer_idx;
    if (name) {
        snprintf(node->name, sizeof(node->name), "%s", name);
    } else {
        snprintf(node->name, sizeof(node->name), "node_%d", idx);
    }
    return idx;
}

void noel_cgraph_free(noel_cgraph_t *graph) {
    if (!graph) return;
    memset(graph, 0, sizeof(*graph));
}

const char *noel_node_type_name(noel_node_type_t type) {
    switch (type) {
        case NODE_EMBEDDING:    return "Embedding";
        case NODE_RMSNORM:      return "RMSNorm";
        case NODE_QKV_GEMV:     return "QKV_GEMV";
        case NODE_ROPE:         return "RoPE";
        case NODE_GQA_ATTENTION:return "GQA_Attention";
        case NODE_ATTN_OUTPUT:  return "Attn_Output";
        case NODE_RESIDUAL_ADD: return "Residual_Add";
        case NODE_FFN_GATE_UP:  return "FFN_Gate_Up";
        case NODE_SWIGLU_MUL:   return "SwiGLU_Mul";
        case NODE_FFN_DOWN:     return "FFN_Down";
        case NODE_LM_HEAD:      return "LM_Head";
        default:                return "Unknown_Node";
    }
}

const char *noel_dispatch_tier_name(noel_dispatch_tier_t tier) {
    switch (tier) {
        case DISPATCH_GPU_CUDA:    return "GPU_VRAM";
        case DISPATCH_CPU_SIMD:    return "CPU_RAM";
        case DISPATCH_NVME_STREAM: return "NVMe_DirectIO";
        case DISPATCH_AUTO:
        default:                   return "Auto_Tier";
    }
}

bool noel_cgraph_validate(const noel_cgraph_t *graph) {
    if (!graph || !graph->is_compiled) return false;
    if (graph->node_count <= 0 || graph->node_count > NOEL_MAX_GRAPH_NODES) return false;
    if (graph->dim <= 0 || graph->n_layers <= 0) return false;

    if (graph->nodes[0].type != NODE_EMBEDDING && graph->nodes[0].type != NODE_RMSNORM) {
        return false;
    }
    int last_idx = graph->node_count - 1;
    if (graph->nodes[last_idx].type != NODE_LM_HEAD && graph->nodes[last_idx].type != NODE_RMSNORM) {
        return false;
    }

    for (int i = 0; i < graph->node_count; i++) {
        const noel_cgraph_node_t *node = &graph->nodes[i];
        if (node->id != i) return false;
        if (node->layer_idx < -1 || node->layer_idx >= graph->n_layers) return false;
        if (node->in_dim < 0 || node->out_dim < 0) return false;
    }

    return true;
}

void noel_cgraph_optimize_buffers(const noel_cgraph_t *graph, noel_cgraph_mem_plan_t *plan) {
    if (!plan) return;
    memset(plan, 0, sizeof(*plan));
    if (!graph || graph->node_count == 0) return;

    size_t dim_bytes = (size_t)graph->dim * sizeof(float);
    size_t hidden_bytes = (size_t)graph->hidden_dim * sizeof(float);
    size_t unopt_per_layer = (dim_bytes * 3) + (hidden_bytes * 2);
    size_t unoptimized = unopt_per_layer * (size_t)graph->n_layers + dim_bytes;

    size_t optimized = unopt_per_layer + dim_bytes;

    plan->unoptimized_bytes = unoptimized;
    plan->optimized_bytes = optimized;
    plan->memory_saved_bytes = (unoptimized > optimized) ? (unoptimized - optimized) : 0;
    plan->reuse_efficiency_pct = (unoptimized > 0)
        ? ((float)plan->memory_saved_bytes / (float)unoptimized) * 100.0f
        : 0.0f;
}

int noel_cgraph_build_layer(noel_cgraph_t *graph,
                            int layer_idx,
                            const struct aether_weights_s *weights,
                            const struct noel_config_s *cfg,
                            noel_dispatch_tier_t tier) {
    if (!graph || !weights || !cfg || layer_idx < 0) return -1;

    char buf[64];
    int start_node = graph->node_count;
    int kv_dim = (cfg->n_kv_heads > 0 && cfg->n_heads > 0)
                 ? (cfg->n_kv_heads * (cfg->dim / cfg->n_heads))
                 : cfg->dim;

    /* 1. Input RMSNorm */
    snprintf(buf, sizeof(buf), "blk.%d.attn_norm", layer_idx);
    int n = noel_cgraph_add_node(graph, NODE_RMSNORM, tier, layer_idx, buf);
    if (n >= 0) {
        graph->nodes[n].weights = weights->is_bound_gguf ? weights->layer_tensors[layer_idx].attn_norm : NULL;
        graph->nodes[n].weight_type = GGML_TYPE_F32;
        graph->nodes[n].in_dim = cfg->dim;
        graph->nodes[n].out_dim = cfg->dim;
        graph->nodes[n].param = cfg->norm_eps;
    }

    /* 2. QKV Projection GEMV */
    snprintf(buf, sizeof(buf), "blk.%d.attn_qkv", layer_idx);
    n = noel_cgraph_add_node(graph, NODE_QKV_GEMV, tier, layer_idx, buf);
    if (n >= 0) {
        if (weights->is_bound_gguf) {
            graph->nodes[n].weights = weights->layer_tensors[layer_idx].attn_q;
            graph->nodes[n].weights_secondary = weights->layer_tensors[layer_idx].attn_k;
            graph->nodes[n].weights_tertiary = weights->layer_tensors[layer_idx].attn_v;
            graph->nodes[n].weight_type = weights->layer_tensors[layer_idx].attn_q_type;
            graph->nodes[n].weight_type_secondary = weights->layer_tensors[layer_idx].attn_k_type;
            graph->nodes[n].weight_type_tertiary = weights->layer_tensors[layer_idx].attn_v_type;
        }
        graph->nodes[n].in_dim = cfg->dim;
        graph->nodes[n].out_dim = cfg->dim;
        graph->nodes[n].hidden_dim = kv_dim;
    }

    /* 3. RoPE Position Embedding */
    snprintf(buf, sizeof(buf), "blk.%d.rope", layer_idx);
    n = noel_cgraph_add_node(graph, NODE_ROPE, DISPATCH_CPU_SIMD, layer_idx, buf);
    if (n >= 0) {
        graph->nodes[n].in_dim = cfg->dim;
        graph->nodes[n].param = cfg->rope_theta;
    }

    /* 4. GQA Attention with KV Cache */
    snprintf(buf, sizeof(buf), "blk.%d.gqa_attn", layer_idx);
    n = noel_cgraph_add_node(graph, NODE_GQA_ATTENTION, tier, layer_idx, buf);
    if (n >= 0) {
        graph->nodes[n].in_dim = cfg->dim;
        graph->nodes[n].hidden_dim = kv_dim;
        graph->nodes[n].param = (float)cfg->seq_len;
    }

    /* 5. Attention Output Projection */
    snprintf(buf, sizeof(buf), "blk.%d.attn_output", layer_idx);
    n = noel_cgraph_add_node(graph, NODE_ATTN_OUTPUT, tier, layer_idx, buf);
    if (n >= 0) {
        if (weights->is_bound_gguf) {
            graph->nodes[n].weights = weights->layer_tensors[layer_idx].attn_output;
            graph->nodes[n].weight_type = weights->layer_tensors[layer_idx].attn_out_type;
        }
        graph->nodes[n].in_dim = cfg->dim;
        graph->nodes[n].out_dim = cfg->dim;
    }

    /* 6. Attention Residual Add */
    snprintf(buf, sizeof(buf), "blk.%d.attn_residual", layer_idx);
    n = noel_cgraph_add_node(graph, NODE_RESIDUAL_ADD, DISPATCH_CPU_SIMD, layer_idx, buf);
    if (n >= 0) {
        graph->nodes[n].in_dim = cfg->dim;
        graph->nodes[n].out_dim = cfg->dim;
    }

    /* 7. Post-Attention / FFN RMSNorm */
    snprintf(buf, sizeof(buf), "blk.%d.ffn_norm", layer_idx);
    n = noel_cgraph_add_node(graph, NODE_RMSNORM, tier, layer_idx, buf);
    if (n >= 0) {
        graph->nodes[n].weights = weights->is_bound_gguf ? weights->layer_tensors[layer_idx].ffn_norm : NULL;
        graph->nodes[n].weight_type = GGML_TYPE_F32;
        graph->nodes[n].in_dim = cfg->dim;
        graph->nodes[n].out_dim = cfg->dim;
        graph->nodes[n].param = cfg->norm_eps;
    }

    /* 8. FFN Gate & Up Projections */
    snprintf(buf, sizeof(buf), "blk.%d.ffn_gate_up", layer_idx);
    n = noel_cgraph_add_node(graph, NODE_FFN_GATE_UP, tier, layer_idx, buf);
    if (n >= 0) {
        if (weights->is_bound_gguf) {
            graph->nodes[n].weights = weights->layer_tensors[layer_idx].ffn_gate;
            graph->nodes[n].weights_secondary = weights->layer_tensors[layer_idx].ffn_up;
            graph->nodes[n].weight_type = weights->layer_tensors[layer_idx].ffn_gate_type;
            graph->nodes[n].weight_type_secondary = weights->layer_tensors[layer_idx].ffn_up_type;
        }
        graph->nodes[n].in_dim = cfg->dim;
        graph->nodes[n].out_dim = cfg->hidden_dim;
    }

    /* 9. SwiGLU Activation */
    snprintf(buf, sizeof(buf), "blk.%d.swiglu", layer_idx);
    n = noel_cgraph_add_node(graph, NODE_SWIGLU_MUL, DISPATCH_CPU_SIMD, layer_idx, buf);
    if (n >= 0) {
        graph->nodes[n].in_dim = cfg->hidden_dim;
        graph->nodes[n].out_dim = cfg->hidden_dim;
    }

    /* 10. FFN Down Projection */
    snprintf(buf, sizeof(buf), "blk.%d.ffn_down", layer_idx);
    n = noel_cgraph_add_node(graph, NODE_FFN_DOWN, tier, layer_idx, buf);
    if (n >= 0) {
        if (weights->is_bound_gguf) {
            graph->nodes[n].weights = weights->layer_tensors[layer_idx].ffn_down;
            graph->nodes[n].weight_type = weights->layer_tensors[layer_idx].ffn_down_type;
        }
        graph->nodes[n].in_dim = cfg->hidden_dim;
        graph->nodes[n].out_dim = cfg->dim;
    }

    /* 11. FFN Residual Add */
    snprintf(buf, sizeof(buf), "blk.%d.ffn_residual", layer_idx);
    n = noel_cgraph_add_node(graph, NODE_RESIDUAL_ADD, DISPATCH_CPU_SIMD, layer_idx, buf);
    if (n >= 0) {
        graph->nodes[n].in_dim = cfg->dim;
        graph->nodes[n].out_dim = cfg->dim;
    }

    return start_node;
}

int noel_cgraph_compile(noel_cgraph_t *graph,
                        const struct aether_weights_s *weights,
                        const struct noel_config_s *cfg) {
    if (!graph || !weights || !cfg) return -1;

    noel_cgraph_init(graph, cfg->dim, cfg->hidden_dim, cfg->n_layers, cfg->n_heads, cfg->n_kv_heads);
    graph->vocab_size = cfg->vocab_size;

    noel_dispatch_tier_t tier = (cfg->execution_mode == NOEL_MODE_MONOLITHIC_GPU)
                                 ? DISPATCH_GPU_CUDA : DISPATCH_CPU_SIMD;

    /* A. Embedding Lookup Node */
    int n_emb = noel_cgraph_add_node(graph, NODE_EMBEDDING, DISPATCH_CPU_SIMD, -1, "token_embd");
    if (n_emb >= 0) {
        graph->nodes[n_emb].weights = weights->raw_token_embd;
        graph->nodes[n_emb].weight_type = weights->token_embd_type;
        graph->nodes[n_emb].in_dim = cfg->vocab_size;
        graph->nodes[n_emb].out_dim = cfg->dim;
    }

    /* B. Transformer Layers */
    for (int l = 0; l < cfg->n_layers; l++) {
        noel_cgraph_build_layer(graph, l, weights, cfg, tier);
    }

    /* C. Output RMSNorm */
    int n_norm = noel_cgraph_add_node(graph, NODE_RMSNORM, tier, -1, "output_norm");
    if (n_norm >= 0) {
        graph->nodes[n_norm].weights = weights->raw_final_norm;
        graph->nodes[n_norm].weight_type = GGML_TYPE_F32;
        graph->nodes[n_norm].in_dim = cfg->dim;
        graph->nodes[n_norm].out_dim = cfg->dim;
        graph->nodes[n_norm].param = cfg->norm_eps;
    }

    /* D. LM Head Logit Projection */
    int n_lm = noel_cgraph_add_node(graph, NODE_LM_HEAD, tier, -1, "output");
    if (n_lm >= 0) {
        graph->nodes[n_lm].weights = weights->raw_output_weight;
        graph->nodes[n_lm].weight_type = weights->output_type;
        graph->nodes[n_lm].in_dim = cfg->dim;
        graph->nodes[n_lm].out_dim = cfg->vocab_size;
    }

    graph->is_compiled = true;
    noel_cgraph_validate(graph);
    return graph->node_count;
}

int noel_cgraph_execute_node(noel_cgraph_t *graph,
                             int node_idx,
                             struct aether_state_s *state,
                             const struct aether_weights_s *weights,
                             const struct noel_config_s *cfg,
                             int pos,
                             int num_threads) {
    if (!graph || node_idx < 0 || node_idx >= graph->node_count || !state || !cfg) return -1;

    (void)weights;
    noel_cgraph_node_t *node = &graph->nodes[node_idx];
    double t0 = noel_cgraph_time_us();

    switch (node->type) {
        case NODE_EMBEDDING:
            /* If embedding weights are provided and state->x is initialized */
            break;

        case NODE_RMSNORM:
            if (node->weights) {
                if (node->layer_idx < 0) {
                    /* Output norm */
                    aether_rmsnorm(state->x, state->x, (const float *)node->weights, node->in_dim, node->param);
                } else {
                    /* Layer attn_norm or ffn_norm */
                    aether_rmsnorm(state->xb, state->x, (const float *)node->weights, node->in_dim, node->param);
                }
            } else {
                memcpy(state->xb, state->x, node->in_dim * sizeof(float));
            }
            break;

        case NODE_QKV_GEMV: {
            int dim = node->in_dim;
            int kv_dim = node->hidden_dim;
            if (node->weights) {
                noel_gemv_quantized(node->weight_type, state->q, node->weights, state->xb, dim, dim, num_threads);
            }
            if (node->weights_secondary) {
                noel_gemv_quantized(node->weight_type_secondary, state->k, node->weights_secondary, state->xb, kv_dim, dim, num_threads);
            }
            if (node->weights_tertiary) {
                noel_gemv_quantized(node->weight_type_tertiary, state->v, node->weights_tertiary, state->xb, kv_dim, dim, num_threads);
            }
            break;
        }

        case NODE_ROPE: {
            int head_dim = (cfg->n_heads > 0) ? (cfg->dim / cfg->n_heads) : 64;
            aether_rope(state->q, cfg->n_heads, head_dim, pos, node->param);
            aether_rope(state->k, cfg->n_kv_heads, head_dim, pos, node->param);

            if (node->layer_idx >= 0 && pos < cfg->seq_len && state->key_cache && state->val_cache) {
                int kv_dim = cfg->n_kv_heads * head_dim;
                float *k_row = state->key_cache + ((size_t)node->layer_idx * cfg->seq_len + pos) * kv_dim;
                float *v_row = state->val_cache + ((size_t)node->layer_idx * cfg->seq_len + pos) * kv_dim;
                memcpy(k_row, state->k, kv_dim * sizeof(float));
                memcpy(v_row, state->v, kv_dim * sizeof(float));
            }
            break;
        }

        case NODE_GQA_ATTENTION: {
            int head_dim = (cfg->n_heads > 0) ? (cfg->dim / cfg->n_heads) : 64;
            int kv_dim = cfg->n_kv_heads * head_dim;
            if (node->layer_idx >= 0 && state->key_cache && state->val_cache) {
                const float *k_layer = state->key_cache + (size_t)node->layer_idx * cfg->seq_len * kv_dim;
                const float *v_layer = state->val_cache + (size_t)node->layer_idx * cfg->seq_len * kv_dim;
                noel_gqa_attention(state->xb2, state->q, k_layer, v_layer,
                                   cfg->n_heads, cfg->n_kv_heads, head_dim,
                                   cfg->seq_len, pos, num_threads);
            }
            break;
        }

        case NODE_ATTN_OUTPUT:
            if (node->weights) {
                noel_gemv_quantized(node->weight_type, state->xb, node->weights, state->xb2, node->out_dim, node->in_dim, num_threads);
            }
            break;

        case NODE_RESIDUAL_ADD: {
            int d = 0;
            for (; d + 7 < node->in_dim; d += 8) {
                __m256 vx = _mm256_loadu_ps(state->x + d);
                __m256 vxb = _mm256_loadu_ps(state->xb + d);
                _mm256_storeu_ps(state->x + d, _mm256_add_ps(vx, vxb));
            }
            for (; d < node->in_dim; d++) {
                state->x[d] += state->xb[d];
            }
            break;
        }

        case NODE_FFN_GATE_UP:
            if (node->weights) {
                noel_gemv_quantized(node->weight_type, state->hb, node->weights, state->xb, node->out_dim, node->in_dim, num_threads);
            }
            if (node->weights_secondary) {
                noel_gemv_quantized(node->weight_type_secondary, state->hb2, node->weights_secondary, state->xb, node->out_dim, node->in_dim, num_threads);
            }
            break;

        case NODE_SWIGLU_MUL:
            aether_swiglu(state->hb, state->hb, state->hb2, node->in_dim);
            break;

        case NODE_FFN_DOWN:
            if (node->weights) {
                noel_gemv_quantized(node->weight_type, state->xb, node->weights, state->hb, node->out_dim, node->in_dim, num_threads);
            }
            break;

        case NODE_LM_HEAD:
            if (node->weights) {
                noel_gemv_quantized(node->weight_type, state->logits, node->weights, state->x, node->out_dim, node->in_dim, num_threads);
            }
            break;

        default:
            break;
    }

    double t1 = noel_cgraph_time_us();
    node->exec_time_us = t1 - t0;
    graph->total_exec_time_us += node->exec_time_us;

    return 0;
}

int noel_cgraph_execute_layer(noel_cgraph_t *graph,
                              int layer_idx,
                              struct aether_state_s *state,
                              const struct aether_weights_s *weights,
                              const struct noel_config_s *cfg,
                              int pos,
                              int num_threads) {
    if (!graph || layer_idx < 0 || layer_idx >= graph->n_layers) return -1;

    /* Compute-I/O Concurrency Overlap:
       While current layer compute runs on GPU / P-cores,
       asynchronously prefetch next layer from NVMe storage! */
    if (layer_idx < cfg->n_layers - 1 && cfg->n_experts > 1) {
        uint64_t next_offset = aether_calc_expert_offset(cfg, layer_idx + 1, 0);
        size_t next_size = aether_calc_expert_bytes(cfg->dim, cfg->hidden_dim) * cfg->n_active_experts;
        aether_streamer_prefetch_async(layer_idx + 1, next_offset, next_size);
    }

    for (int i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].layer_idx == layer_idx) {
            noel_cgraph_execute_node(graph, i, state, weights, cfg, pos, num_threads);
        }
    }
    return 0;
}

int noel_cgraph_execute(noel_cgraph_t *graph,
                        struct aether_state_s *state,
                        const struct aether_weights_s *weights,
                        const struct noel_config_s *cfg,
                        int pos,
                        int num_threads) {
    if (!graph || !graph->is_compiled || !state || !cfg) return -1;

    graph->total_exec_time_us = 0.0;

    for (int i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i].layer_idx >= 0 && graph->nodes[i].type == NODE_RMSNORM &&
            strstr(graph->nodes[i].name, "attn_norm")) {
            int l = graph->nodes[i].layer_idx;
            if (l < cfg->n_layers - 1 && cfg->n_experts > 1) {
                uint64_t next_offset = aether_calc_expert_offset(cfg, l + 1, 0);
                size_t next_size = aether_calc_expert_bytes(cfg->dim, cfg->hidden_dim) * cfg->n_active_experts;
                aether_streamer_prefetch_async(l + 1, next_offset, next_size);
            }
        }
        noel_cgraph_execute_node(graph, i, state, weights, cfg, pos, num_threads);
    }

    return 0;
}

void noel_cgraph_print(const noel_cgraph_t *graph) {
    if (!graph) return;
    printf("========================================================================================\n");
    printf("                          NOEL COMPUTATION GRAPH IR (v3)                                \n");
    printf("========================================================================================\n");
    printf("  Layers: %d | Dim: %d | Hidden: %d | Heads: %d (KV: %d) | Nodes: %d\n",
           graph->n_layers, graph->dim, graph->hidden_dim, graph->n_heads, graph->n_kv_heads, graph->node_count);
    printf("----------------------------------------------------------------------------------------\n");
    printf("  %-4s  %-4s  %-15s  %-14s  %-14s  %-12s  %s\n",
           "ID", "Lyr", "Type", "Tier", "Dimensions", "Precision", "Name");
    printf("----------------------------------------------------------------------------------------\n");

    for (int i = 0; i < graph->node_count; i++) {
        const noel_cgraph_node_t *nd = &graph->nodes[i];
        char dim_str[32];
        if (nd->type == NODE_QKV_GEMV) {
            snprintf(dim_str, sizeof(dim_str), "%dx%d,%dx%d", nd->in_dim, nd->out_dim, nd->in_dim, nd->hidden_dim);
        } else if (nd->type == NODE_RMSNORM || nd->type == NODE_ROPE || nd->type == NODE_RESIDUAL_ADD) {
            snprintf(dim_str, sizeof(dim_str), "%d", nd->in_dim);
        } else {
            snprintf(dim_str, sizeof(dim_str), "%dx%d", nd->in_dim, nd->out_dim);
        }

        const char *wtype_name = "None";
        if (nd->weights) {
            wtype_name = noel_ggml_type_name(nd->weight_type);
        }

        char lyr_str[16] = "-";
        if (nd->layer_idx >= 0) {
            snprintf(lyr_str, sizeof(lyr_str), "%d", nd->layer_idx);
        }

        printf("  [%03d] %-4s  %-15s  %-14s  %-14s  %-12s  %s\n",
               nd->id, lyr_str, noel_node_type_name(nd->type), noel_dispatch_tier_name(nd->tier),
               dim_str, wtype_name, nd->name);
    }
    printf("========================================================================================\n");
    noel_cgraph_mem_plan_t plan;
    noel_cgraph_optimize_buffers(graph, &plan);
    printf("  [Buffer Reuse Optimizer] Unoptimized: %.2f MiB -> Optimized: %.2f MiB (Saved: %.2f MiB, %.1f%%)\n",
           (double)plan.unoptimized_bytes / (1024.0 * 1024.0),
           (double)plan.optimized_bytes / (1024.0 * 1024.0),
           (double)plan.memory_saved_bytes / (1024.0 * 1024.0),
           plan.reuse_efficiency_pct);
    printf("========================================================================================\n");
}
