#ifndef AETHER_SPARSITY_H
#define AETHER_SPARSITY_H

#include "aether.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t expert_id;
    float weight;
} moe_routing_t;

typedef struct {
    int layer_idx;
    int total_neurons;
    int num_hot_neurons;
    int32_t *hot_indices;
    int32_t *cold_indices;
} neuron_partition_t;

/* Partition MLP neurons into hot (GPU VRAM resident) and cold (Host RAM / SSD) */
neuron_partition_t aether_partition_neurons(int layer, int total_neurons, float hot_ratio);

/* Predict active cold neurons for a given token activation vector (PowerInfer) */
int aether_predict_active_neurons(const float *x, int dim, const float *pred_weights,
                                 int total_cold, int32_t *out_active_indices, float threshold);

/* Route token to top-k experts (MoE routing for Kimi-k3 / Mixtral architectures) */
void aether_route_top_k(const float *gate_logits, int n_experts, int top_k, moe_routing_t *out_routing);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_SPARSITY_H */
