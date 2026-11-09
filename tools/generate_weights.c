#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "aether.h"

int main(int argc, char **argv) {
    const char *out_path = "data/weights.noel";
    if (argc > 1) out_path = argv[1];

    printf("[Weight Gen] Creating 4096-aligned Noel model file at: %s\n", out_path);

    noel_config_t cfg = {
        .dim = 2048,
        .hidden_dim = 5632,
        .n_layers = 16,
        .n_heads = 16,
        .n_kv_heads = 4,
        .vocab_size = 32000,
        .seq_len = 1024,
        .n_experts = 8,
        .n_active_experts = 2,
        .hot_ratio = 0.20f,
        .norm_eps = 1e-5f,
        .rope_theta = 10000.0f
    };

    int fd = open(out_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open failed");
        return 1;
    }

    /* Write 4096-byte header */
    uint8_t header_buf[AETHER_ALIGNMENT] = {0};
    uint32_t magic = AETHER_MAGIC;
    uint32_t ver = AETHER_VERSION;
    memcpy(header_buf, &magic, sizeof(magic));
    memcpy(header_buf + 4, &ver, sizeof(ver));
    memcpy(header_buf + 8, &cfg, sizeof(cfg));

    if (write(fd, header_buf, AETHER_ALIGNMENT) != AETHER_ALIGNMENT) {
        perror("write header failed");
        close(fd);
        return 1;
    }

    /* Write simulated layer expert weights */
    size_t aligned_expert_bytes = aether_calc_expert_bytes(cfg.dim, cfg.hidden_dim);

    void *buf = NULL;
    if (posix_memalign(&buf, AETHER_ALIGNMENT, aligned_expert_bytes) != 0 || !buf) {
        perror("posix_memalign failed");
        close(fd);
        return 1;
    }
    memset(buf, 0x55, aligned_expert_bytes);

    printf("  • Writing %d layers x %d experts = %d expert matrices (%.2f MB each)...\n",
           cfg.n_layers, cfg.n_experts, cfg.n_layers * cfg.n_experts,
           (double)aligned_expert_bytes / (1024.0 * 1024.0));

    for (int l = 0; l < cfg.n_layers; l++) {
        for (int e = 0; e < cfg.n_experts; e++) {
            if (write(fd, buf, aligned_expert_bytes) != (ssize_t)aligned_expert_bytes) {
                perror("write expert failed");
                free(buf);
                close(fd);
                return 1;
            }
        }
    }

    free(buf);
    close(fd);

    printf("[Weight Gen] Done! Generated %s successfully.\n", out_path);
    return 0;
}
