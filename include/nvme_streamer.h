#ifndef AETHER_NVME_STREAMER_H
#define AETHER_NVME_STREAMER_H

#include "aether.h"
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int layer_idx;
    int expert_idx;
    uint64_t file_offset;
    size_t size_bytes;
    void *data; /* 4096-byte aligned */
    uint64_t last_accessed_tick;
    bool is_valid;
} lru_entry_t;

typedef struct {
    lru_entry_t *entries;
    int capacity;
    int count;
    uint64_t tick_counter;
    pthread_mutex_t lock;
} lru_cache_t;

/* Double-buffered prefetch staging slot */
typedef struct {
    void *buffer[2]; /* Two 4096-byte aligned buffers for double-buffering */
    size_t buffer_size;
    int current_slot;
    int prefetch_layer;
    bool prefetch_in_progress;
    pthread_t worker_thread;
    pthread_mutex_t lock;
    pthread_cond_t start_cond;
    pthread_cond_t done_cond;
    bool stop_worker;
} prefetch_pipeline_t;

typedef struct {
    int fd;
    const char *file_path;
    size_t file_size;
    lru_cache_t lru_cache;
    prefetch_pipeline_t pipeline;
} nvme_streamer_t;

extern nvme_streamer_t g_streamer;

/* Initialize the NVMe Direct I/O Streamer and launch background E-core worker */
bool aether_streamer_init(const char *weight_file, size_t max_cache_bytes, size_t layer_buffer_size);

/* Request asynchronous prefetch of the next layer's weights while current layer computes */
void aether_streamer_prefetch_async(int next_layer, uint64_t offset, size_t size);

/* Wait for prefetch of the layer to complete and return aligned pointer */
void *aether_streamer_wait_and_get(int layer, uint64_t offset, size_t size);

/* Check LRU cache or fetch from disk */
void *aether_streamer_get_expert(int layer, int expert, uint64_t offset, size_t size);

/* Shutdown streamer and free resources */
void aether_streamer_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_NVME_STREAMER_H */
