#ifndef AETHER_HARDWARE_H
#define AETHER_HARDWARE_H

#include <sched.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    int total_cpus;
    int num_p_cores;
    int num_e_cores;
    cpu_set_t p_core_mask;
    cpu_set_t e_core_mask;
    bool has_avx2;
    bool has_avx_vnni;
    bool has_fma;
    size_t l1_cache_size;
    size_t l2_cache_size;
    size_t l3_cache_size;
} cpu_topology_t;

/* Global hardware descriptor */
extern cpu_topology_t g_cpu_topo;

/* Detect system topology, P-cores vs E-cores, AVX-VNNI flags */
void aether_detect_hardware(void);

/* Pin the current calling thread to P-cores */
int aether_pin_to_p_cores(void);

/* Pin the current calling thread to E-cores (for async disk I/O) */
int aether_pin_to_e_cores(void);

/* Pin calling thread to a specific single core ID */
int aether_pin_to_core(int core_id);

/* High-resolution timer in seconds */
double aether_time_sec(void);

#endif /* AETHER_HARDWARE_H */
