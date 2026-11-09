#include "hardware.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <cpuid.h>

cpu_topology_t g_cpu_topo = {0};

static bool check_avx_vnni(void) {
    unsigned int eax, ebx, ecx, edx;
    /* EAX=7, ECX=1 */
    __cpuid_count(7, 1, eax, ebx, ecx, edx);
    /* Bit 4 of EAX is AVX-VNNI */
    return (eax & (1 << 4)) != 0;
}

void aether_detect_hardware(void) {
    g_cpu_topo.total_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    CPU_ZERO(&g_cpu_topo.p_core_mask);
    CPU_ZERO(&g_cpu_topo.e_core_mask);

    g_cpu_topo.has_avx2 = __builtin_cpu_supports("avx2");
    g_cpu_topo.has_fma  = __builtin_cpu_supports("fma");
    g_cpu_topo.has_avx_vnni = check_avx_vnni();

    /* Classify P-cores vs E-cores by reading thread_siblings and frequencies */
    for (int cpu = 0; cpu < g_cpu_topo.total_cpus; cpu++) {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        FILE *f = fopen(path, "r");
        bool has_hyperthreading = false;
        if (f) {
            char line[64];
            if (fgets(line, sizeof(line), f)) {
                /* If line contains comma or hyphen, it shares a core with another thread -> P-core */
                if (strchr(line, ',') || strchr(line, '-')) {
                    has_hyperthreading = true;
                }
            }
            fclose(f);
        }

        /* Also check max frequency as a secondary heuristic */
        unsigned long max_freq = 0;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
        f = fopen(path, "r");
        if (f) {
            if (fscanf(f, "%lu", &max_freq) == 1) {
                /* On i7-13650HX, P-cores have higher base/max freq */
            }
            fclose(f);
        }

        if (has_hyperthreading || max_freq >= 4000000) {
            CPU_SET(cpu, &g_cpu_topo.p_core_mask);
            g_cpu_topo.num_p_cores++;
        } else {
            CPU_SET(cpu, &g_cpu_topo.e_core_mask);
            g_cpu_topo.num_e_cores++;
        }
    }

    g_cpu_topo.l1_cache_size = 48 * 1024;
    g_cpu_topo.l2_cache_size = 1280 * 1024;
    g_cpu_topo.l3_cache_size = 24 * 1024 * 1024;

    printf("[Hardware] Detected %d CPUs: %d P-core threads, %d E-core threads\n",
           g_cpu_topo.total_cpus, g_cpu_topo.num_p_cores, g_cpu_topo.num_e_cores);
    printf("[Hardware] Hardware Acceleration: AVX2=%s, FMA=%s, AVX-VNNI=%s\n",
           g_cpu_topo.has_avx2 ? "YES" : "NO",
           g_cpu_topo.has_fma ? "YES" : "NO",
           g_cpu_topo.has_avx_vnni ? "YES (vpdpbusd 32-op/cycle)" : "NO");
}

int aether_pin_to_p_cores(void) {
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &g_cpu_topo.p_core_mask);
}

int aether_pin_to_e_cores(void) {
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &g_cpu_topo.e_core_mask);
}

int aether_pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

double aether_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
