#include "sparsity.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

neuron_partition_t aether_partition_neurons(int layer, int total_neurons, float hot_ratio) {
    neuron_partition_t part;
    part.layer_idx = layer;
    part.total_neurons = total_neurons;
    part.num_hot_neurons = (int)(total_neurons * hot_ratio);
    if (part.num_hot_neurons < 1 && hot_ratio > 0.0f) part.num_hot_neurons = 1;
    if (part.num_hot_neurons > total_neurons) part.num_hot_neurons = total_neurons;

    int num_cold = total_neurons - part.num_hot_neurons;
    part.hot_indices = (int32_t*)malloc(part.num_hot_neurons * sizeof(int32_t));
    part.cold_indices = (int32_t*)malloc(num_cold * sizeof(int32_t));

    /* In PowerInfer, hot neurons are ranked by offline activation frequency */
    /* By default, top indices are selected as hot */
    for (int i = 0; i < part.num_hot_neurons; i++) {
        part.hot_indices[i] = i;
    }
    for (int i = 0; i < num_cold; i++) {
        part.cold_indices[i] = part.num_hot_neurons + i;
    }

    return part;
}

int aether_predict_active_neurons(const float *x, int dim, const float *pred_weights,
                                 int total_cold, int32_t *out_active_indices, float threshold) {
    int active_count = 0;
    for (int n = 0; n < total_cold; n++) {
        const float *w = pred_weights + n * dim;
        float dot = 0.0f;
        for (int i = 0; i < dim; i++) {
            dot += x[i] * w[i];
        }
        if (dot > threshold) {
            out_active_indices[active_count++] = n;
        }
    }
    return active_count;
}

void aether_route_top_k(const float *gate_logits, int n_experts, int top_k, moe_routing_t *out_routing) {
    /* Initialize with first top_k */
    for (int k = 0; k < top_k; k++) {
        out_routing[k].expert_id = -1;
        out_routing[k].weight = -1e9f;
    }

    for (int e = 0; e < n_experts; e++) {
        float val = gate_logits[e];
        /* Insert sort into top_k list */
        for (int k = 0; k < top_k; k++) {
            if (val > out_routing[k].weight) {
                for (int m = top_k - 1; m > k; m--) {
                    out_routing[m] = out_routing[m - 1];
                }
                out_routing[k].expert_id = e;
                out_routing[k].weight = val;
                break;
            }
        }
    }

    /* Softmax over the top-k selected experts */
    float max_logit = out_routing[0].weight;
    float sum_exp = 0.0f;
    for (int k = 0; k < top_k; k++) {
        out_routing[k].weight = expf(out_routing[k].weight - max_logit);
        sum_exp += out_routing[k].weight;
    }
    float inv_sum = sum_exp > 0.0f ? (1.0f / sum_exp) : 0.0f;
    for (int k = 0; k < top_k; k++) {
        out_routing[k].weight *= inv_sum;
    }
}
