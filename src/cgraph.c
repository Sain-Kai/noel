#include "cgraph.h"
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    return graph->node_count;
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
}
