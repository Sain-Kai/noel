#include "cgraph.h"
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
