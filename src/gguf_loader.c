#include "gguf_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

const char *noel_ggml_type_name(noel_ggml_type_t type) {
    switch (type) {
        case GGML_TYPE_F32:     return "F32";
        case GGML_TYPE_F16:     return "F16";
        case GGML_TYPE_Q4_0:    return "Q4_0";
        case GGML_TYPE_Q4_1:    return "Q4_1";
        case GGML_TYPE_Q5_0:    return "Q5_0";
        case GGML_TYPE_Q5_1:    return "Q5_1";
        case GGML_TYPE_Q8_0:    return "Q8_0";
        case GGML_TYPE_Q8_1:    return "Q8_1";
        case GGML_TYPE_Q2_K:    return "Q2_K";
        case GGML_TYPE_Q3_K:    return "Q3_K";
        case GGML_TYPE_Q4_K:    return "Q4_K";
        case GGML_TYPE_Q5_K:    return "Q5_K";
        case GGML_TYPE_Q6_K:    return "Q6_K";
        case GGML_TYPE_Q8_K:    return "Q8_K";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_IQ2_XS:  return "IQ2_XS";
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case GGML_TYPE_IQ1_S:   return "IQ1_S";
        case GGML_TYPE_IQ4_NL:  return "IQ4_NL";
        case GGML_TYPE_IQ3_S:   return "IQ3_S";
        case GGML_TYPE_IQ2_S:   return "IQ2_S";
        case GGML_TYPE_IQ4_XS:  return "IQ4_XS";
        case GGML_TYPE_BF16:    return "BF16";
        default:                return "UNKNOWN";
    }
}

static size_t gguf_type_size(noel_gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default:                return 0;
    }
}

bool noel_gguf_open(noel_gguf_t *ctx, const char *filepath) {
    if (!ctx || !filepath) return false;
    memset(ctx, 0, sizeof(*ctx));

    ctx->fd = open(filepath, O_RDONLY);
    if (ctx->fd < 0) {
        perror("[GGUF Loader] Failed to open file");
        return false;
    }

    struct stat st;
    if (fstat(ctx->fd, &st) != 0) {
        perror("[GGUF Loader] fstat failed");
        close(ctx->fd);
        ctx->fd = -1;
        return false;
    }
    ctx->file_size = st.st_size;

    /* Zero-copy memory map */
    ctx->mmap_addr = mmap(NULL, ctx->file_size, PROT_READ, MAP_SHARED, ctx->fd, 0);
    if (ctx->mmap_addr == MAP_FAILED) {
        perror("[GGUF Loader] mmap failed");
        close(ctx->fd);
        ctx->fd = -1;
        return false;
    }

    /* Advise OS for sequential read during header scanning */
    madvise(ctx->mmap_addr, ctx->file_size < (64 * 1024 * 1024) ? ctx->file_size : (64 * 1024 * 1024), MADV_WILLNEED);

    const uint8_t *p = (const uint8_t*)ctx->mmap_addr;
    const uint8_t *end = p + ctx->file_size;

    /* Verify Magic: 0x46554747 "GGUF" */
    uint32_t magic = *(const uint32_t*)p;
    if (magic != NOEL_GGUF_MAGIC) {
        fprintf(stderr, "[GGUF Loader] Invalid magic: 0x%08X (expected 0x%08llX 'GGUF')\n",
                magic, (unsigned long long)NOEL_GGUF_MAGIC);
        noel_gguf_close(ctx);
        return false;
    }
    p += 4;

    /* Version */
    ctx->version = *(const uint32_t*)p;
    p += 4;
    if (ctx->version != 2 && ctx->version != 3) {
        fprintf(stderr, "[GGUF Loader] Unsupported GGUF version: %u\n", ctx->version);
        noel_gguf_close(ctx);
        return false;
    }

    /* Tensor count and Metadata KV count */
    ctx->n_tensors = *(const uint64_t*)p; p += 8;
    ctx->n_kv      = *(const uint64_t*)p; p += 8;

    if (ctx->n_tensors > NOEL_GGUF_MAX_TENSORS * 16 || ctx->n_kv > NOEL_GGUF_MAX_KV * 16) {
        fprintf(stderr, "[GGUF Loader] Sanity check failed: n_tensors=%lu, n_kv=%lu\n",
                ctx->n_tensors, ctx->n_kv);
        noel_gguf_close(ctx);
        return false;
    }

    ctx->alignment = 32; /* Default GGUF alignment */
    ctx->kvs = (noel_gguf_kv_t*)calloc(ctx->n_kv, sizeof(noel_gguf_kv_t));
    if (!ctx->kvs) {
        noel_gguf_close(ctx);
        return false;
    }

    /* Parse Key-Value Metadata */
    for (uint64_t i = 0; i < ctx->n_kv && p < end; i++) {
        uint64_t klen = *(const uint64_t*)p; p += 8;
        size_t cplen = klen < NOEL_GGUF_MAX_STR_LEN - 1 ? klen : NOEL_GGUF_MAX_STR_LEN - 1;
        memcpy(ctx->kvs[i].key, p, cplen);
        ctx->kvs[i].key[cplen] = '\0';
        p += klen;

        uint32_t vtype = *(const uint32_t*)p; p += 4;
        ctx->kvs[i].type = (noel_gguf_type_t)vtype;

        switch (vtype) {
            case GGUF_TYPE_UINT8:   ctx->kvs[i].val.u32 = *p; p += 1; break;
            case GGUF_TYPE_INT8:    ctx->kvs[i].val.i32 = *(const int8_t*)p; p += 1; break;
            case GGUF_TYPE_UINT16:  ctx->kvs[i].val.u32 = *(const uint16_t*)p; p += 2; break;
            case GGUF_TYPE_INT16:   ctx->kvs[i].val.i32 = *(const int16_t*)p; p += 2; break;
            case GGUF_TYPE_UINT32:  ctx->kvs[i].val.u32 = *(const uint32_t*)p; p += 4; break;
            case GGUF_TYPE_INT32:   ctx->kvs[i].val.i32 = *(const int32_t*)p; p += 4; break;
            case GGUF_TYPE_FLOAT32: ctx->kvs[i].val.f32 = *(const float*)p; p += 4; break;
            case GGUF_TYPE_BOOL:    ctx->kvs[i].val.b = (*p != 0); p += 1; break;
            case GGUF_TYPE_STRING: {
                uint64_t slen = *(const uint64_t*)p; p += 8;
                size_t scp = slen < NOEL_GGUF_MAX_STR_LEN - 1 ? slen : NOEL_GGUF_MAX_STR_LEN - 1;
                memcpy(ctx->kvs[i].val.str, p, scp);
                ctx->kvs[i].val.str[scp] = '\0';
                p += slen;
                break;
            }
            case GGUF_TYPE_ARRAY: {
                uint32_t item_type = *(const uint32_t*)p; p += 4;
                uint64_t count = *(const uint64_t*)p; p += 8;
                ctx->kvs[i].val.u64 = count; /* Record array count */
                if (item_type == GGUF_TYPE_STRING) {
                    for (uint64_t c = 0; c < count && p < end; c++) {
                        uint64_t s_len = *(const uint64_t*)p; p += 8 + s_len;
                    }
                } else {
                    size_t isize = gguf_type_size((noel_gguf_type_t)item_type);
                    p += count * isize;
                }
                break;
            }
            case GGUF_TYPE_UINT64:  ctx->kvs[i].val.u64 = *(const uint64_t*)p; p += 8; break;
            case GGUF_TYPE_INT64:   ctx->kvs[i].val.i64 = *(const int64_t*)p; p += 8; break;
            case GGUF_TYPE_FLOAT64: ctx->kvs[i].val.f64 = *(const double*)p; p += 8; break;
            default:
                fprintf(stderr, "[GGUF Loader] Unknown value type %u for key '%s'\n", vtype, ctx->kvs[i].key);
                break;
        }

        if (strcmp(ctx->kvs[i].key, "general.alignment") == 0) {
            ctx->alignment = ctx->kvs[i].val.u32;
        } else if (strcmp(ctx->kvs[i].key, "general.architecture") == 0) {
            snprintf(ctx->arch, sizeof(ctx->arch), "%s", ctx->kvs[i].val.str);
        } else if (strcmp(ctx->kvs[i].key, "general.name") == 0) {
            snprintf(ctx->model_name, sizeof(ctx->model_name), "%s", ctx->kvs[i].val.str);
        }
    }

    /* Parse Tensor Info */
    ctx->tensors = (noel_tensor_info_t*)calloc(ctx->n_tensors, sizeof(noel_tensor_info_t));
    if (!ctx->tensors) {
        noel_gguf_close(ctx);
        return false;
    }

    for (uint64_t i = 0; i < ctx->n_tensors && p < end; i++) {
        uint64_t nlen = *(const uint64_t*)p; p += 8;
        size_t cplen = nlen < NOEL_GGUF_MAX_STR_LEN - 1 ? nlen : NOEL_GGUF_MAX_STR_LEN - 1;
        memcpy(ctx->tensors[i].name, p, cplen);
        ctx->tensors[i].name[cplen] = '\0';
        p += nlen;

        uint32_t ndims = *(const uint32_t*)p; p += 4;
        ctx->tensors[i].n_dims = ndims;
        ctx->tensors[i].n_elements = 1;
        for (uint32_t d = 0; d < 4; d++) {
            if (d < ndims) {
                ctx->tensors[i].ne[d] = *(const uint64_t*)p; p += 8;
                ctx->tensors[i].n_elements *= ctx->tensors[i].ne[d];
            } else {
                ctx->tensors[i].ne[d] = 1;
            }
        }

        ctx->tensors[i].type   = (noel_ggml_type_t)(*(const uint32_t*)p); p += 4;
        ctx->tensors[i].offset = *(const uint64_t*)p; p += 8;
    }

    /* Calculate data base offset aligned to ctx->alignment */
    uint64_t cur_pos = (uint64_t)(p - (const uint8_t*)ctx->mmap_addr);
    ctx->data_offset = (cur_pos + ctx->alignment - 1) & ~((uint64_t)ctx->alignment - 1);

    /* Assign tensor data pointers and compute byte sizes */
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        ctx->tensors[i].data = (const uint8_t*)ctx->mmap_addr + ctx->data_offset + ctx->tensors[i].offset;
        if (i < ctx->n_tensors - 1) {
            ctx->tensors[i].size_bytes = (size_t)(ctx->tensors[i+1].offset - ctx->tensors[i].offset);
        } else {
            ctx->tensors[i].size_bytes = ctx->file_size - (ctx->data_offset + ctx->tensors[i].offset);
        }
    }

    /* Auto-extract normalized Noel model configuration */
    noel_gguf_extract_config(ctx, &ctx->config);

    return true;
}

void noel_gguf_close(noel_gguf_t *ctx) {
    if (!ctx) return;
    if (ctx->tensors) free(ctx->tensors);
    if (ctx->kvs) free(ctx->kvs);
    if (ctx->mmap_addr && ctx->mmap_addr != MAP_FAILED) {
        munmap(ctx->mmap_addr, ctx->file_size);
    }
    if (ctx->fd >= 0) close(ctx->fd);
    memset(ctx, 0, sizeof(*ctx));
    ctx->fd = -1;
}

const noel_gguf_kv_t *noel_gguf_find_kv(const noel_gguf_t *ctx, const char *key) {
    if (!ctx || !ctx->kvs || !key) return NULL;
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kvs[i].key, key) == 0) {
            return &ctx->kvs[i];
        }
    }
    return NULL;
}

int64_t noel_gguf_get_kv_int(const noel_gguf_t *ctx, const char *key, int64_t default_val) {
    const noel_gguf_kv_t *kv = noel_gguf_find_kv(ctx, key);
    if (!kv) return default_val;
    if (kv->type == GGUF_TYPE_UINT32) return (int64_t)kv->val.u32;
    if (kv->type == GGUF_TYPE_INT32)  return (int64_t)kv->val.i32;
    if (kv->type == GGUF_TYPE_UINT64) return (int64_t)kv->val.u64;
    if (kv->type == GGUF_TYPE_INT64)  return kv->val.i64;
    return default_val;
}

float noel_gguf_get_kv_float(const noel_gguf_t *ctx, const char *key, float default_val) {
    const noel_gguf_kv_t *kv = noel_gguf_find_kv(ctx, key);
    if (!kv) return default_val;
    if (kv->type == GGUF_TYPE_FLOAT32) return kv->val.f32;
    if (kv->type == GGUF_TYPE_FLOAT64) return (float)kv->val.f64;
    return default_val;
}

const char *noel_gguf_get_kv_string(const noel_gguf_t *ctx, const char *key, const char *default_val) {
    const noel_gguf_kv_t *kv = noel_gguf_find_kv(ctx, key);
    if (!kv || kv->type != GGUF_TYPE_STRING) return default_val;
    return kv->val.str;
}

const noel_tensor_info_t *noel_gguf_find_tensor(const noel_gguf_t *ctx, const char *name) {
    if (!ctx || !ctx->tensors || !name) return NULL;
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

const void *noel_gguf_get_tensor_data(const noel_gguf_t *ctx, const char *name, size_t *out_size, noel_ggml_type_t *out_type) {
    const noel_tensor_info_t *t = noel_gguf_find_tensor(ctx, name);
    if (!t) return NULL;
    if (out_size) *out_size = t->size_bytes;
    if (out_type) *out_type = t->type;
    return t->data;
}

bool noel_gguf_extract_config(noel_gguf_t *ctx, noel_config_t *cfg) {
    if (!ctx || !cfg) return false;
    memset(cfg, 0, sizeof(*cfg));

    const char *arch = ctx->arch[0] ? ctx->arch : "llama";
    char key_buf[512];

    #define GET_INT(field, key_suffix, def) \
        snprintf(key_buf, sizeof(key_buf), "%s.%s", arch, key_suffix); \
        cfg->field = (int)noel_gguf_get_kv_int(ctx, key_buf, def);

    #define GET_FLOAT(field, key_suffix, def) \
        snprintf(key_buf, sizeof(key_buf), "%s.%s", arch, key_suffix); \
        cfg->field = noel_gguf_get_kv_float(ctx, key_buf, def);

    GET_INT(n_layers, "block_count", 16);
    GET_INT(dim, "embedding_length", 2048);
    GET_INT(hidden_dim, "feed_forward_length", 5632);
    GET_INT(n_heads, "attention.head_count", 16);
    GET_INT(n_kv_heads, "attention.head_count_kv", cfg->n_heads);

    /* Attention head dimension */
    snprintf(key_buf, sizeof(key_buf), "%s.attention.key_length", arch);
    int key_len = (int)noel_gguf_get_kv_int(ctx, key_buf, 0);
    if (key_len > 0) {
        cfg->head_dim = key_len;
    } else if (cfg->n_heads > 0) {
        cfg->head_dim = cfg->dim / cfg->n_heads;
    } else {
        cfg->head_dim = 128;
    }

    /* Context length: Safe active sequence window (capped to 2048 default to avoid allocating tens of GBs) */
    snprintf(key_buf, sizeof(key_buf), "%s.context_length", arch);
    int model_ctx = (int)noel_gguf_get_kv_int(ctx, key_buf, 2048);
    cfg->seq_len = (model_ctx > 0 && model_ctx <= 2048) ? model_ctx : 2048;

    GET_FLOAT(norm_eps, "attention.layer_norm_rms_epsilon", 1e-5f);
    GET_FLOAT(rope_theta, "rope.freq_base", 10000.0f);

    /* Check vocabulary size from token_embd tensor shape or tokenizer array */
    const noel_tensor_info_t *embd = noel_gguf_find_tensor(ctx, "token_embd.weight");
    if (embd && embd->n_dims >= 2) {
        cfg->vocab_size = (int)embd->ne[1];
    } else {
        const noel_tensor_info_t *out_w = noel_gguf_find_tensor(ctx, "output.weight");
        if (out_w && out_w->n_dims >= 2) {
            cfg->vocab_size = (int)out_w->ne[1];
        } else {
            const noel_gguf_kv_t *tok_kv = noel_gguf_find_kv(ctx, "tokenizer.ggml.tokens");
            cfg->vocab_size = tok_kv ? (int)tok_kv->val.u64 : 32000;
        }
    }

    /* MoE parameters */
    snprintf(key_buf, sizeof(key_buf), "%s.expert_count", arch);
    cfg->n_experts = (int)noel_gguf_get_kv_int(ctx, key_buf, 1);
    if (cfg->n_experts <= 0) cfg->n_experts = 1;

    snprintf(key_buf, sizeof(key_buf), "%s.expert_used_count", arch);
    cfg->n_active_experts = (int)noel_gguf_get_kv_int(ctx, key_buf, 1);
    if (cfg->n_active_experts <= 0) cfg->n_active_experts = 1;

    cfg->hot_ratio = 0.20f;
    cfg->execution_mode = NOEL_MODE_AUTO;

    return true;
}

void noel_gguf_print_summary(const noel_gguf_t *ctx) {
    if (!ctx) return;

    printf("=================================================================\n");
    printf("         NOEL GGUF ZERO-COPY TENSOR INSPECTION REPORT            \n");
    printf("=================================================================\n");
    printf(" • Model Name:        %s\n", ctx->model_name[0] ? ctx->model_name : "Unnamed");
    printf(" • Architecture:      %s\n", ctx->arch);
    printf(" • GGUF Version:      %u\n", ctx->version);
    printf(" • File Size:         %.2f GiB (%zu bytes)\n",
           (double)ctx->file_size / (1024.0 * 1024.0 * 1024.0), ctx->file_size);
    printf(" • Metadata Pairs:    %lu key-values\n", ctx->n_kv);
    printf(" • Total Tensors:     %lu tensors (Data Offset: %lu bytes, Align: %u)\n",
           ctx->n_tensors, ctx->data_offset, ctx->alignment);
    printf("-----------------------------------------------------------------\n");
    printf(" [Extracted Hyperparameters]\n");
    printf(" • Hidden Dimension:  %d\n", ctx->config.dim);
    printf(" • Feed-Forward Dim:  %d\n", ctx->config.hidden_dim);
    printf(" • Number of Layers:  %d\n", ctx->config.n_layers);
    printf(" • Attention Heads:   %d (KV Heads: %d, Head Dim: %d)\n",
           ctx->config.n_heads, ctx->config.n_kv_heads, ctx->config.head_dim);
    printf(" • Vocabulary Size:   %d tokens\n", ctx->config.vocab_size);
    printf(" • Context Window:    %d tokens (Active KV Cache: %d)\n",
           (int)noel_gguf_get_kv_int(ctx, "context_length", ctx->config.seq_len), ctx->config.seq_len);
    printf(" • RMSNorm Epsilon:   %.2e\n", ctx->config.norm_eps);
    printf(" • RoPE Frequency:    %.1f\n", ctx->config.rope_theta);
    printf(" • Mixture of Experts:%d experts (%d active per token)\n",
           ctx->config.n_experts, ctx->config.n_active_experts);
    printf("=================================================================\n");
}
