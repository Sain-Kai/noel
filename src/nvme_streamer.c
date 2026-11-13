#include "nvme_streamer.h"
#include "hardware.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

nvme_streamer_t g_streamer = {0};

/* Internal prefetch task state */
typedef struct {
    int target_layer;
    uint64_t target_offset;
    size_t target_size;
    int target_slot;
} prefetch_task_t;

static prefetch_task_t s_current_task;

static void *prefetch_worker_loop(void *arg) {
    (void)arg;
    /* Pin this background prefetch thread strictly to E-cores to preserve P-cores for AVX-VNNI */
    aether_pin_to_e_cores();

    while (1) {
        pthread_mutex_lock(&g_streamer.pipeline.lock);
        while (!g_streamer.pipeline.prefetch_in_progress && !g_streamer.pipeline.stop_worker) {
            pthread_cond_wait(&g_streamer.pipeline.start_cond, &g_streamer.pipeline.lock);
        }

        if (g_streamer.pipeline.stop_worker) {
            pthread_mutex_unlock(&g_streamer.pipeline.lock);
            break;
        }

        prefetch_task_t task = s_current_task;
        pthread_mutex_unlock(&g_streamer.pipeline.lock);

        /* Perform direct high-speed pread from NVMe if file is open */
        if (g_streamer.fd >= 0) {
            void *dst_buf = g_streamer.pipeline.buffer[task.target_slot];
            ssize_t bytes_read = pread(g_streamer.fd, dst_buf, task.target_size, task.target_offset);
            if (bytes_read < 0) {
                perror("[NVMe Streamer] pread failed");
            }
        }

        pthread_mutex_lock(&g_streamer.pipeline.lock);
        g_streamer.pipeline.prefetch_in_progress = false;
        pthread_cond_broadcast(&g_streamer.pipeline.done_cond);
        pthread_mutex_unlock(&g_streamer.pipeline.lock);
    }
    return NULL;
}

bool aether_streamer_init(const char *weight_file, size_t max_cache_bytes, size_t layer_buffer_size) {
    memset(&g_streamer, 0, sizeof(g_streamer));
    g_streamer.file_path = weight_file;

    g_streamer.fd = open(weight_file, O_RDONLY);
    if (g_streamer.fd < 0) {
        /* If weight file doesn't exist yet, we can create or allow memory-only benchmark */
        printf("[NVMe Streamer] Note: Weight file %s not opened directly (%m)\n", weight_file);
    } else {
        struct stat st;
        fstat(g_streamer.fd, &st);
        g_streamer.file_size = st.st_size;
        printf("[NVMe Streamer] Opened %s (Size: %.2f GiB)\n",
               weight_file, (double)st.st_size / (1024.0 * 1024.0 * 1024.0));
    }

    /* Allocate double-buffered prefetch slots with 4096-byte alignment */
    g_streamer.pipeline.buffer_size = layer_buffer_size;
    for (int i = 0; i < 2; i++) {
        int err = posix_memalign(&g_streamer.pipeline.buffer[i], AETHER_ALIGNMENT, layer_buffer_size);
        if (err != 0 || !g_streamer.pipeline.buffer[i]) {
            fprintf(stderr, "[NVMe Streamer] posix_memalign failed for staging slot %d\n", i);
            return false;
        }
        memset(g_streamer.pipeline.buffer[i], 0, layer_buffer_size);
    }

    pthread_mutex_init(&g_streamer.pipeline.lock, NULL);
    pthread_cond_init(&g_streamer.pipeline.start_cond, NULL);
    pthread_cond_init(&g_streamer.pipeline.done_cond, NULL);

    /* Initialize LRU cache */
    size_t avg_expert_size = 2 * 1024 * 1024; /* ~2 MB per expert slice */
    int capacity = max_cache_bytes / avg_expert_size;
    if (capacity < 16) capacity = 16;
    if (capacity > 2048) capacity = 2048;

    g_streamer.lru_cache.capacity = capacity;
    g_streamer.lru_cache.entries = (lru_entry_t*)calloc(capacity, sizeof(lru_entry_t));
    pthread_mutex_init(&g_streamer.lru_cache.lock, NULL);

    /* Launch asynchronous background prefetch worker on E-cores */
    pthread_create(&g_streamer.pipeline.worker_thread, NULL, prefetch_worker_loop, NULL);

    printf("[NVMe Streamer] Staging Double-Buffers: 2 x %.2f MiB, LRU Capacity: %d experts (Max ~%.2f GiB)\n",
           (double)layer_buffer_size / (1024.0 * 1024.0),
           capacity, (double)max_cache_bytes / (1024.0 * 1024.0 * 1024.0));
    return true;
}

void aether_streamer_prefetch_async(int next_layer, uint64_t offset, size_t size) {
    pthread_mutex_lock(&g_streamer.pipeline.lock);
    int next_slot = 1 - g_streamer.pipeline.current_slot;
    s_current_task.target_layer = next_layer;
    s_current_task.target_offset = offset;
    s_current_task.target_size = size;
    s_current_task.target_slot = next_slot;
    g_streamer.pipeline.prefetch_layer = next_layer;
    g_streamer.pipeline.prefetch_in_progress = true;

    pthread_cond_signal(&g_streamer.pipeline.start_cond);
    pthread_mutex_unlock(&g_streamer.pipeline.lock);
}

void *aether_streamer_wait_and_get(int layer, uint64_t offset, size_t size) {
    pthread_mutex_lock(&g_streamer.pipeline.lock);

    /* If this layer was the one prefetched, wait for completion */
    if (g_streamer.pipeline.prefetch_layer == layer) {
        while (g_streamer.pipeline.prefetch_in_progress) {
            pthread_cond_wait(&g_streamer.pipeline.done_cond, &g_streamer.pipeline.lock);
        }
        /* Swap slot */
        g_streamer.pipeline.current_slot = 1 - g_streamer.pipeline.current_slot;
        void *data = g_streamer.pipeline.buffer[g_streamer.pipeline.current_slot];
        pthread_mutex_unlock(&g_streamer.pipeline.lock);
        return data;
    }

    /* Otherwise synchronous read into current slot */
    pthread_mutex_unlock(&g_streamer.pipeline.lock);
    void *data = g_streamer.pipeline.buffer[g_streamer.pipeline.current_slot];
    if (g_streamer.fd >= 0) {
        pread(g_streamer.fd, data, size, offset);
    }
    return data;
}

void *aether_streamer_get_expert(int layer, int expert, uint64_t offset, size_t size) {
    pthread_mutex_lock(&g_streamer.lru_cache.lock);
    g_streamer.lru_cache.tick_counter++;

    /* Check LRU cache for hit */
    for (int i = 0; i < g_streamer.lru_cache.count; i++) {
        lru_entry_t *e = &g_streamer.lru_cache.entries[i];
        if (e->is_valid && e->layer_idx == layer && e->expert_idx == expert) {
            e->last_accessed_tick = g_streamer.lru_cache.tick_counter;
            pthread_mutex_unlock(&g_streamer.lru_cache.lock);
            return e->data;
        }
    }

    /* Cache miss: Find victim entry (LRU) */
    int victim = 0;
    if (g_streamer.lru_cache.count < g_streamer.lru_cache.capacity) {
        victim = g_streamer.lru_cache.count++;
        posix_memalign(&g_streamer.lru_cache.entries[victim].data, AETHER_ALIGNMENT, size);
    } else {
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < g_streamer.lru_cache.capacity; i++) {
            if (g_streamer.lru_cache.entries[i].last_accessed_tick < oldest) {
                oldest = g_streamer.lru_cache.entries[i].last_accessed_tick;
                victim = i;
            }
        }
        if (g_streamer.lru_cache.entries[victim].size_bytes < size) {
            free(g_streamer.lru_cache.entries[victim].data);
            posix_memalign(&g_streamer.lru_cache.entries[victim].data, AETHER_ALIGNMENT, size);
        }
    }

    lru_entry_t *entry = &g_streamer.lru_cache.entries[victim];
    entry->layer_idx = layer;
    entry->expert_idx = expert;
    entry->file_offset = offset;
    entry->size_bytes = size;
    entry->last_accessed_tick = g_streamer.lru_cache.tick_counter;
    entry->is_valid = true;

    /* Read from disk into allocated entry data */
    if (g_streamer.fd >= 0) {
        pread(g_streamer.fd, entry->data, size, offset);
    }

    pthread_mutex_unlock(&g_streamer.lru_cache.lock);
    return entry->data;
}

void aether_streamer_shutdown(void) {
    if (g_streamer.pipeline.worker_thread) {
        pthread_mutex_lock(&g_streamer.pipeline.lock);
        g_streamer.pipeline.stop_worker = true;
        pthread_cond_signal(&g_streamer.pipeline.start_cond);
        pthread_mutex_unlock(&g_streamer.pipeline.lock);

        pthread_join(g_streamer.pipeline.worker_thread, NULL);
        g_streamer.pipeline.worker_thread = 0;
    }

    for (int i = 0; i < 2; i++) {
        if (g_streamer.pipeline.buffer[i]) {
            free(g_streamer.pipeline.buffer[i]);
            g_streamer.pipeline.buffer[i] = NULL;
        }
    }

    for (int i = 0; i < g_streamer.lru_cache.count; i++) {
        if (g_streamer.lru_cache.entries[i].data) {
            free(g_streamer.lru_cache.entries[i].data);
            g_streamer.lru_cache.entries[i].data = NULL;
        }
    }
    if (g_streamer.lru_cache.entries) {
        free(g_streamer.lru_cache.entries);
        g_streamer.lru_cache.entries = NULL;
    }

    if (g_streamer.fd >= 0) {
        close(g_streamer.fd);
        g_streamer.fd = -1;
    }

    memset(&g_streamer, 0, sizeof(g_streamer));
    g_streamer.fd = -1;
}
