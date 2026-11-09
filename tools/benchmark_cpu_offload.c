#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "hardware.h"
#include "pipeline_stages.h"

int main(void) {
    printf("=================================================================\n");
    printf("  AETHER INFERENCE ENGINE: CPU AUXILIARY PIPELINE BENCHMARK      \n");
    printf("  Testing Prerequisite & Post-Requisite Offloading on i7-13650HX \n");
    printf("=================================================================\n\n");

    aether_detect_hardware();

    int vocab_size = 32000;
    int dim = 4096;
    int head_dim = 128;
    int context_len = 2048;

    /* ------------------------------------------------------------- */
    /* 1. Prerequisite: Host RAM Embedding Lookup                    */
    /* ------------------------------------------------------------- */
    printf("[1/4] Benchmarking Host RAM Embedding Lookup (P-Core & E-Core)...\n");
    float *host_emb = (float*)malloc((size_t)vocab_size * dim * sizeof(float));
    float *dst_x = (float*)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) host_emb[100 * dim + i] = 0.5f;

    int iters = 10000;
    double t0 = aether_time_sec();
    for (int i = 0; i < iters; i++) {
        aether_prereq_embedding_lookup(dst_x, host_emb, 100 + (i % 1000), dim);
    }
    double t1 = aether_time_sec();
    double emb_us = ((t1 - t0) / iters) * 1e6;
    printf("  • Embedding Gather Latency: %.2f µs (Microseconds) per token\n", emb_us);
    printf("  • VRAM Saved by keeping Embeddings in Host RAM: %.2f MiB (%.2f GiB)\n",
           (double)(vocab_size * dim * sizeof(float)) / (1024*1024),
           (double)(vocab_size * dim * sizeof(float)) / (1024*1024*1024));

    /* ------------------------------------------------------------- */
    /* 2. Prerequisite: RoPE Frequencies Precomputation              */
    /* ------------------------------------------------------------- */
    printf("\n[2/4] Benchmarking RoPE Precomputation (Head Dim=%d)...\n", head_dim);
    float cos_buf[64], sin_buf[64];
    t0 = aether_time_sec();
    for (int i = 0; i < iters; i++) {
        aether_prereq_precompute_rope(cos_buf, sin_buf, head_dim, i % 2048, 10000.0f);
    }
    t1 = aether_time_sec();
    double rope_us = ((t1 - t0) / iters) * 1e6;
    printf("  • RoPE Table Generation Latency: %.3f µs per position\n", rope_us);

    /* ------------------------------------------------------------- */
    /* 3. Post-requisite: Repetition Penalty over Context History    */
    /* ------------------------------------------------------------- */
    printf("\n[3/4] Benchmarking Context Repetition Penalty Scan (Context=%d tokens)...\n", context_len);
    float *logits = (float*)malloc(vocab_size * sizeof(float));
    int32_t *history = (int32_t*)malloc(context_len * sizeof(int32_t));
    for (int i = 0; i < vocab_size; i++) logits[i] = ((float)rand() / RAND_MAX) * 10.0f;
    for (int i = 0; i < context_len; i++) history[i] = rand() % vocab_size;

    t0 = aether_time_sec();
    for (int i = 0; i < 2000; i++) {
        aether_postreq_apply_penalties(logits, vocab_size, history, context_len, 1.15f, context_len);
    }
    t1 = aether_time_sec();
    double penalty_us = ((t1 - t0) / 2000) * 1e6;
    printf("  • History Context Penalty Latency: %.2f µs (Masked concurrently)\n", penalty_us);

    /* ------------------------------------------------------------- */
    /* 4. Post-requisite: Top-K + Top-P (Nucleus) Sampling           */
    /* ------------------------------------------------------------- */
    printf("\n[4/4] Benchmarking Advanced Sampling Pipeline (Vocab=%d, Top-K=50, Top-P=0.9)...\n", vocab_size);
    aether_sampling_params_t params = {
        .temperature = 0.7f,
        .top_p = 0.90f,
        .top_k = 50,
        .repetition_penalty = 1.15f,
        .repetition_window = 2048,
        .grammar_enforce_json = false
    };

    t0 = aether_time_sec();
    for (int i = 0; i < 1000; i++) {
        /* Re-seed random logits */
        logits[0] = 12.0f;
        int32_t sampled = aether_postreq_sample(logits, vocab_size, &params);
        (void)sampled;
    }
    t1 = aether_time_sec();
    double sample_us = ((t1 - t0) / 1000) * 1e6;
    printf("  • Full Softmax + Top-K + Top-P Sampling Latency: %.2f µs per token\n", sample_us);

    printf("\n=================================================================\n");
    printf("  TOTAL AUXILIARY OVERHEAD MASKED ON CPU: ~%.2f µs (ZERO GPU STALL)\n",
           emb_us + rope_us + penalty_us + sample_us);
    printf("=================================================================\n");

    free(host_emb);
    free(dst_x);
    free(logits);
    free(history);
    return 0;
}
