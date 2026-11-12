#ifndef NOEL_GGUF_LOADER_H
#define NOEL_GGUF_LOADER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "noel.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NOEL_GGUF_MAGIC 0x46554747ULL /* "GGUF" in little-endian */
#define NOEL_GGUF_MAX_TENSORS 4096
#define NOEL_GGUF_MAX_KV 2048
#define NOEL_GGUF_MAX_STR_LEN 256

/* GGML Quantization Tensor Types */
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
} noel_ggml_type_t;

/* GGUF Metadata Value Types */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} noel_gguf_type_t;

typedef struct {
    char name[NOEL_GGUF_MAX_STR_LEN];
    uint32_t n_dims;
    uint64_t ne[4];          /* Dimensions: ne[0]=cols, ne[1]=rows, etc. */
    uint64_t n_elements;     /* Total number of elements */
    noel_ggml_type_t type;   /* GGML tensor quantization type */
    uint64_t offset;         /* Byte offset relative to tensor data base */
    size_t size_bytes;       /* Byte size in file */
    const void *data;        /* Direct pointer into mmap file */
} noel_tensor_info_t;

typedef struct {
    char key[NOEL_GGUF_MAX_STR_LEN];
    noel_gguf_type_t type;
    union {
        uint64_t u64;
        int64_t i64;
        uint32_t u32;
        int32_t i32;
        float f32;
        double f64;
        bool b;
        char str[NOEL_GGUF_MAX_STR_LEN];
    } val;
} noel_gguf_kv_t;

typedef struct {
    int fd;
    size_t file_size;
    void *mmap_addr;         /* Zero-copy memory-mapped file base */
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint32_t alignment;
    uint64_t data_offset;    /* Absolute byte offset of tensor data in file */

    /* Model metadata & hyper-parameters */
    char arch[NOEL_GGUF_MAX_STR_LEN]; /* e.g., "qwen2", "llama", "qwen35" */
    char model_name[NOEL_GGUF_MAX_STR_LEN];
    noel_config_t config;    /* Auto-extracted Noel configuration */

    /* Tensor and KV arrays */
    noel_tensor_info_t *tensors;
    noel_gguf_kv_t *kvs;
} noel_gguf_t;

/* Public GGUF Loader API */
bool noel_gguf_open(noel_gguf_t *ctx, const char *filepath);
void noel_gguf_close(noel_gguf_t *ctx);

/* Extract model configuration into Noel config struct */
bool noel_gguf_extract_config(noel_gguf_t *ctx, noel_config_t *cfg);

/* Fast tensor lookup */
const noel_tensor_info_t *noel_gguf_find_tensor(const noel_gguf_t *ctx, const char *name);
const void *noel_gguf_get_tensor_data(const noel_gguf_t *ctx, const char *name, size_t *out_size, noel_ggml_type_t *out_type);

/* Helper to get KV pair by key */
const noel_gguf_kv_t *noel_gguf_find_kv(const noel_gguf_t *ctx, const char *key);
int64_t noel_gguf_get_kv_int(const noel_gguf_t *ctx, const char *key, int64_t default_val);
float noel_gguf_get_kv_float(const noel_gguf_t *ctx, const char *key, float default_val);
const char *noel_gguf_get_kv_string(const noel_gguf_t *ctx, const char *key, const char *default_val);

/* Print comprehensive inspection report */
void noel_gguf_print_summary(const noel_gguf_t *ctx);

/* Helper string for GGML tensor type */
const char *noel_ggml_type_name(noel_ggml_type_t type);

#ifdef __cplusplus
}
#endif

#endif /* NOEL_GGUF_LOADER_H */
