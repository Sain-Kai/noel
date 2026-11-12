#include <stdio.h>
#include <stdlib.h>
#include "gguf_loader.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <path_to_gguf_model>\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    printf("[Noel] Inspecting GGUF file: %s\n", path);

    noel_gguf_t ctx;
    if (!noel_gguf_open(&ctx, path)) {
        fprintf(stderr, "[Error] Failed to parse GGUF file: %s\n", path);
        return 1;
    }

    noel_gguf_print_summary(&ctx);

    printf("\nFirst 10 Model Tensors:\n");
    printf("-----------------------------------------------------------------\n");
    printf(" %-4s | %-32s | %-8s | %-16s | %s\n", "Idx", "Tensor Name", "Type", "Shape [0..3]", "Size (MB)");
    printf("-----------------------------------------------------------------\n");
    for (uint64_t i = 0; i < ctx.n_tensors && i < 10; i++) {
        char shape_buf[64];
        if (ctx.tensors[i].n_dims == 1) {
            snprintf(shape_buf, sizeof(shape_buf), "[%lu]", ctx.tensors[i].ne[0]);
        } else if (ctx.tensors[i].n_dims == 2) {
            snprintf(shape_buf, sizeof(shape_buf), "[%lu, %lu]", ctx.tensors[i].ne[0], ctx.tensors[i].ne[1]);
        } else if (ctx.tensors[i].n_dims == 3) {
            snprintf(shape_buf, sizeof(shape_buf), "[%lu, %lu, %lu]", ctx.tensors[i].ne[0], ctx.tensors[i].ne[1], ctx.tensors[i].ne[2]);
        } else {
            snprintf(shape_buf, sizeof(shape_buf), "[%lu, %lu, %lu, %lu]", ctx.tensors[i].ne[0], ctx.tensors[i].ne[1], ctx.tensors[i].ne[2], ctx.tensors[i].ne[3]);
        }

        printf(" #%-3lu | %-32s | %-8s | %-16s | %.2f MB\n",
               i + 1,
               ctx.tensors[i].name,
               noel_ggml_type_name(ctx.tensors[i].type),
               shape_buf,
               (double)ctx.tensors[i].size_bytes / (1024.0 * 1024.0));
    }
    printf("-----------------------------------------------------------------\n");

    noel_gguf_close(&ctx);
    return 0;
}
