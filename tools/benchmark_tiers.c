#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <immintrin.h>
#include "hardware.h"
#include "cuda_backend.h"
#include "simd_vnni.h"

#define BENCH_BUF_SIZE (128ULL * 1024 * 1024) /* 128 MB */

int main(void) {
    printf("=================================================================\n");
    printf("  AETHER INFERENCE ENGINE: HARDWARE TIER MICRO-BENCHMARK SUITE   \n");
    printf("=================================================================\n");

    aether_detect_hardware();
    aether_pin_to_p_cores();

    /* ------------------------------------------------------------- */
    /* 1. Micro-benchmark: AVX2 vs AVX-VNNI GEMV Throughput on P-Cores */
    /* ------------------------------------------------------------- */
    printf("\n[1/4] Benchmarking CPU SIMD Micro-Kernels (CPU P-Cores)...\n");
    int rows = 4096;
    int cols = 4096;
    int nb = cols / AETHER_BLOCK_SIZE;

    block_q4_0 *W_q4 = NULL;
    posix_memalign((void**)&W_q4, 64, (size_t)rows * nb * sizeof(block_q4_0));
    float *x = NULL;
    posix_memalign((void**)&x, 64, cols * sizeof(float));
    float *y = NULL;
    posix_memalign((void**)&y, 64, rows * sizeof(float));

    for (int i = 0; i < cols; i++) x[i] = 1.0f;
    for (int i = 0; i < rows * nb; i++) {
        W_q4[i].scale = 0.05f;
        memset(W_q4[i].qs, 0x88, sizeof(W_q4[i].qs)); /* 8 in each nibble -> 0 after offset */
    }

    int warmup = 5;
    int iters = 50;

    for (int i = 0; i < warmup; i++) {
        aether_gemv_q4_0(y, W_q4, x, rows, cols, g_cpu_topo.num_p_cores);
    }

    double t0 = aether_time_sec();
    for (int i = 0; i < iters; i++) {
        aether_gemv_q4_0(y, W_q4, x, rows, cols, g_cpu_topo.num_p_cores);
    }
    double t1 = aether_time_sec();
    double gemv_time_ms = ((t1 - t0) / iters) * 1000.0;
    double gflops = (2.0 * rows * cols * 1e-9) / ((t1 - t0) / iters);

    printf("  • Q4_0 Matrix-Vector GEMV (4096 x 4096):\n");
    printf("    -> Latency:    %.3f ms per vector\n", gemv_time_ms);
    printf("    -> Throughput: %.2f GFLOPS (effective arithmetic density)\n", gflops);

    /* ------------------------------------------------------------- */
    /* 2. Micro-benchmark: Host RAM Memory Bandwidth                 */
    /* ------------------------------------------------------------- */
    printf("\n[2/4] Benchmarking Host System RAM Streaming Bandwidth...\n");
    void *ram_src = NULL;
    void *ram_dst = NULL;
    posix_memalign(&ram_src, 64, BENCH_BUF_SIZE);
    posix_memalign(&ram_dst, 64, BENCH_BUF_SIZE);
    memset(ram_src, 0xAA, BENCH_BUF_SIZE);

    t0 = aether_time_sec();
    for (int i = 0; i < 20; i++) {
        memcpy(ram_dst, ram_src, BENCH_BUF_SIZE);
    }
    t1 = aether_time_sec();
    double ram_bw_gbs = (20.0 * (double)BENCH_BUF_SIZE / (1024.0 * 1024.0 * 1024.0)) / (t1 - t0);
    printf("  • Sequential RAM Copy Bandwidth: %.2f GB/s\n", ram_bw_gbs);

    /* ------------------------------------------------------------- */
    /* 3. Micro-benchmark: NVMe SSD Read Bandwidth                   */
    /* ------------------------------------------------------------- */
    printf("\n[3/4] Benchmarking NVMe Direct Read Bandwidth...\n");
    const char *test_path = "/home/neymar-jr/noel/data/bench_nvme.tmp";
    int fd_w = open(test_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd_w >= 0) {
        write(fd_w, ram_src, BENCH_BUF_SIZE);
        fsync(fd_w);
        close(fd_w);

        int fd_r = open(test_path, O_RDONLY);
        if (fd_r >= 0) {
            t0 = aether_time_sec();
            for (int i = 0; i < 5; i++) {
                pread(fd_r, ram_dst, BENCH_BUF_SIZE, 0);
            }
            t1 = aether_time_sec();
            double nvme_bw = (5.0 * (double)BENCH_BUF_SIZE / (1024.0 * 1024.0 * 1024.0)) / (t1 - t0);
            printf("  • NVMe Read Bandwidth: %.2f GB/s (%.2f MB/s)\n", nvme_bw, nvme_bw * 1024.0);
            close(fd_r);
        }
        unlink(test_path);
    }

    /* ------------------------------------------------------------- */
    /* 4. Micro-benchmark: GPU PCIe H2D & VRAM Bandwidth            */
    /* ------------------------------------------------------------- */
    printf("\n[4/4] Benchmarking Dedicated GPU & PCIe Gen4 Transfer...\n");
    bool has_gpu = aether_cuda_init();
    if (has_gpu) {
        size_t gpu_bench_size = 64ULL * 1024 * 1024; /* 64 MB */
        void *d_bench = aether_cuda_alloc(gpu_bench_size);
        if (d_bench) {
            t0 = aether_time_sec();
            for (int i = 0; i < 20; i++) {
                aether_cuda_memcpy_h2d_async(d_bench, ram_src, gpu_bench_size);
                aether_cuda_sync();
            }
            t1 = aether_time_sec();
            double pcie_bw = (20.0 * (double)gpu_bench_size / (1024.0 * 1024.0 * 1024.0)) / (t1 - t0);
            printf("  • PCIe Gen4 Host-to-Device (H2D) Bandwidth: %.2f GB/s\n", pcie_bw);
            aether_cuda_free(d_bench);
        }
        aether_cuda_shutdown();
    } else {
        printf("  • GPU not available for this run.\n");
    }

    printf("\n=================================================================\n");
    printf("  BENCHMARK COMPLETED SUCCESSFULLY                               \n");
    printf("=================================================================\n");

    free(W_q4);
    free(x);
    free(y);
    free(ram_src);
    free(ram_dst);
    return 0;
}
