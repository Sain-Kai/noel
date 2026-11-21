#ifndef NOEL_CGRAPH_H
#define NOEL_CGRAPH_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "noel.h"
#include "gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NOEL_MAX_GRAPH_NODES 1024

/* Forward declarations */
struct aether_weights_s;
struct noel_config_s;
struct aether_state_s;

typedef enum {
    NODE_EMBEDDING,
    NODE_RMSNORM,
    NODE_QKV_GEMV,
    NODE_ROPE,
    NODE_GQA_ATTENTION,
    NODE_ATTN_OUTPUT,
    NODE_RESIDUAL_ADD,
    NODE_FFN_GATE_UP,
    NODE_SWIGLU_MUL,
    NODE_FFN_DOWN,
    NODE_LM_HEAD
} noel_node_type_t;

typedef enum {
    DISPATCH_AUTO = 0,
    DISPATCH_GPU_CUDA,
    DISPATCH_CPU_SIMD,
    DISPATCH_NVME_STREAM
} noel_dispatch_tier_t;

typedef struct {
    int id;
    noel_node_type_t type;
    noel_dispatch_tier_t tier;
    int layer_idx;
    char name[64];
    const void *weights;
    const void *weights_secondary;
    const void *weights_tertiary;
    noel_ggml_type_t weight_type;
    noel_ggml_type_t weight_type_secondary;
    noel_ggml_type_t weight_type_tertiary;
    int in_dim;
    int out_dim;
    int hidden_dim;
    float param;          /* e.g. norm_eps or rope_theta */
    double exec_time_us;  /* Telemetry: last execution duration */
} noel_cgraph_node_t;

typedef struct {
    noel_cgraph_node_t nodes[NOEL_MAX_GRAPH_NODES];
    int node_count;
    int n_layers;
    int dim;
    int hidden_dim;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int vocab_size;
    bool is_compiled;
    double total_exec_time_us;
} noel_cgraph_t;

/* Graph Lifecycle APIs */
void noel_cgraph_init(noel_cgraph_t *graph, int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads);
int noel_cgraph_add_node(noel_cgraph_t *graph, noel_node_type_t type, noel_dispatch_tier_t tier, int layer_idx, const char *name);
void noel_cgraph_free(noel_cgraph_t *graph);
const char *noel_node_type_name(noel_node_type_t type);
const char *noel_dispatch_tier_name(noel_dispatch_tier_t tier);

/* Layer & Full Transformer Compilation */
int noel_cgraph_build_layer(noel_cgraph_t *graph,
                            int layer_idx,
                            const struct aether_weights_s *weights,
                            const struct noel_config_s *cfg,
                            noel_dispatch_tier_t tier);

int noel_cgraph_compile(noel_cgraph_t *graph,
                        const struct aether_weights_s *weights,
                        const struct noel_config_s *cfg);

void noel_cgraph_print(const noel_cgraph_t *graph);

/* Asynchronous & Cross-Tier Graph Execution */
int noel_cgraph_execute_node(noel_cgraph_t *graph,
                             int node_idx,
                             struct aether_state_s *state,
                             const struct aether_weights_s *weights,
                             const struct noel_config_s *cfg,
                             int pos,
                             int num_threads);

int noel_cgraph_execute_layer(noel_cgraph_t *graph,
                              int layer_idx,
                              struct aether_state_s *state,
                              const struct aether_weights_s *weights,
                              const struct noel_config_s *cfg,
                              int pos,
                              int num_threads);

int noel_cgraph_execute(noel_cgraph_t *graph,
                        struct aether_state_s *state,
                        const struct aether_weights_s *weights,
                        const struct noel_config_s *cfg,
                        int pos,
                        int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* NOEL_CGRAPH_H */
