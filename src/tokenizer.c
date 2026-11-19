#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline uint32_t fnv1a_hash(const char *data, size_t len) {
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint8_t)data[i];
        hash *= 16777619u;
    }
    return hash;
}

static void build_hash_table(aether_tokenizer_t *tok) {
    if (tok->vocab_size <= 0) return;

    int size = 1024;
    while (size < tok->vocab_size * 2) size <<= 1;

    tok->hash_size = size;
    tok->hash_table = (int*)malloc(size * sizeof(int));
    if (!tok->hash_table) return;

    for (int i = 0; i < size; i++) tok->hash_table[i] = -1;

    uint32_t mask = (uint32_t)(size - 1);
    for (int i = 0; i < tok->vocab_size; i++) {
        if (!tok->vocab[i]) continue;
        size_t len = strlen(tok->vocab[i]);
        uint32_t h = fnv1a_hash(tok->vocab[i], len);
        uint32_t idx = h & mask;
        while (tok->hash_table[idx] >= 0) {
            idx = (idx + 1) & mask;
        }
        tok->hash_table[idx] = i;
    }
}

int32_t aether_tokenizer_lookup(const aether_tokenizer_t *tok, const char *str, size_t len) {
    if (!tok || !tok->hash_table || tok->hash_size <= 0 || !str || len == 0) return -1;

    uint32_t mask = (uint32_t)(tok->hash_size - 1);
    uint32_t h = fnv1a_hash(str, len);
    uint32_t idx = h & mask;

    while (tok->hash_table[idx] >= 0) {
        int id = tok->hash_table[idx];
        const char *v = tok->vocab[id];
        if (v && strlen(v) == len && memcmp(v, str, len) == 0) {
            return id;
        }
        idx = (idx + 1) & mask;
    }
    return -1;
}

bool aether_tokenizer_init(aether_tokenizer_t *tok, int vocab_size) {
    tok->vocab_size = vocab_size;
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->pad_id = 0;

    tok->vocab = (char**)calloc(vocab_size, sizeof(char*));
    tok->scores = (float*)calloc(vocab_size, sizeof(float));

    /* Initialize byte fallback tokens for ASCII */
    for (int i = 0; i < 256 && i < vocab_size; i++) {
        char buf[16];
        if (i >= 32 && i <= 126) {
            snprintf(buf, sizeof(buf), "%c", (char)i);
        } else {
            snprintf(buf, sizeof(buf), "<0x%02X>", i);
        }
        tok->vocab[i] = strdup(buf);
    }

    build_hash_table(tok);
    return true;
}

bool aether_tokenizer_load_gguf(aether_tokenizer_t *tok, const noel_gguf_t *gguf) {
    if (!tok || !gguf) return false;

    int32_t vocab_size = 0;
    char **tokens = NULL;
    float *scores = NULL;

    if (!noel_gguf_get_token_table(gguf, &vocab_size, &tokens, &scores)) {
        return false;
    }

    /* Free any previous vocabulary */
    aether_tokenizer_free(tok);

    tok->vocab = tokens;
    tok->scores = scores;
    tok->vocab_size = vocab_size;
    tok->bos_id = gguf->bos_token_id >= 0 ? gguf->bos_token_id : 1;
    tok->eos_id = gguf->eos_token_id >= 0 ? gguf->eos_token_id : 2;
    tok->pad_id = gguf->pad_token_id >= 0 ? gguf->pad_token_id : 0;

    build_hash_table(tok);

    printf("[Tokenizer] Loaded %d embedded vocabulary tokens from GGUF (BOS=%d, EOS=%d)\n",
           vocab_size, tok->bos_id, tok->eos_id);
    return true;
}

int aether_tokenizer_encode(const aether_tokenizer_t *tok, const char *text, int32_t *tokens, int max_tokens) {
    if (!tok || !text || !tokens || max_tokens <= 0) return 0;
    int count = 0;

    if (tok->bos_id >= 0 && tok->bos_id != tok->eos_id && count < max_tokens) {
        tokens[count++] = tok->bos_id;
    }

    size_t text_len = strlen(text);
    size_t pos = 0;

    while (pos < text_len && count < max_tokens) {
        int best_len = 0;
        int32_t best_token = -1;

        size_t max_check = (text_len - pos < 48) ? (text_len - pos) : 48;
        for (size_t l = max_check; l >= 1; l--) {
            int32_t id = aether_tokenizer_lookup(tok, text + pos, l);
            if (id >= 0) {
                best_len = (int)l;
                best_token = id;
                break;
            }

            /* Try with BPE leading space marker 'Ġ' if text[pos] == ' ' */
            if (text[pos] == ' ' && l > 1) {
                char bpe_buf[64];
                bpe_buf[0] = (char)0xC4;
                bpe_buf[1] = (char)0xA0;
                memcpy(bpe_buf + 2, text + pos + 1, l - 1);
                int32_t bpe_id = aether_tokenizer_lookup(tok, bpe_buf, l + 1);
                if (bpe_id >= 0) {
                    best_len = (int)l;
                    best_token = bpe_id;
                    break;
                }
            }
        }

        if (best_token >= 0) {
            tokens[count++] = best_token;
            pos += best_len;
        } else {
            /* Byte fallback or hex format <0xXX> */
            uint8_t byte = (uint8_t)text[pos];
            char byte_str[16];
            snprintf(byte_str, sizeof(byte_str), "<0x%02X>", byte);
            int32_t id = aether_tokenizer_lookup(tok, byte_str, strlen(byte_str));
            if (id >= 0) {
                tokens[count++] = id;
            } else if (byte < (uint32_t)tok->vocab_size) {
                tokens[count++] = (int32_t)byte;
            }
            pos++;
        }
    }

    return count;
}

const char *aether_tokenizer_decode(const aether_tokenizer_t *tok, int32_t token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size || !tok->vocab || !tok->vocab[token_id]) {
        return "";
    }
    return tok->vocab[token_id];
}

const char *aether_tokenizer_decode_piece(const aether_tokenizer_t *tok, int32_t token_id, char *out_buf, size_t max_len) {
    if (!out_buf || max_len == 0) return "";
    out_buf[0] = '\0';

    if (token_id < 0 || token_id >= tok->vocab_size || !tok->vocab || !tok->vocab[token_id]) {
        return "";
    }

    if (token_id == tok->eos_id || token_id == tok->pad_id) {
        return "";
    }

    const char *raw = tok->vocab[token_id];

    /* Hex byte tokens <0xXX> */
    if (raw[0] == '<' && raw[1] == '0' && (raw[2] == 'x' || raw[2] == 'X') && raw[5] == '>') {
        unsigned int byte_val = 0;
        if (sscanf(raw + 3, "%02x", &byte_val) == 1 || sscanf(raw + 3, "%02X", &byte_val) == 1) {
            out_buf[0] = (char)byte_val;
            out_buf[1] = '\0';
            return out_buf;
        }
    }

    /* Translate special BPE / SentencePiece tokens */
    size_t out_idx = 0;
    size_t raw_len = strlen(raw);
    for (size_t i = 0; i < raw_len && out_idx + 4 < max_len; ) {
        /* BPE Space: 'Ġ' (0xC4 0xA0) */
        if ((uint8_t)raw[i] == 0xC4 && (uint8_t)raw[i+1] == 0xA0) {
            out_buf[out_idx++] = ' ';
            i += 2;
        }
        /* SentencePiece Space: ' ' (0xE2 0x96 0x81) */
        else if ((uint8_t)raw[i] == 0xE2 && (uint8_t)raw[i+1] == 0x96 && (uint8_t)raw[i+2] == 0x81) {
            out_buf[out_idx++] = ' ';
            i += 3;
        }
        /* BPE Newline: 'Ċ' (0xC4 0x8A) */
        else if ((uint8_t)raw[i] == 0xC4 && (uint8_t)raw[i+1] == 0x8A) {
            out_buf[out_idx++] = '\n';
            i += 2;
        }
        else {
            out_buf[out_idx++] = raw[i++];
        }
    }
    out_buf[out_idx] = '\0';
    return out_buf;
}

void aether_tokenizer_free(aether_tokenizer_t *tok) {
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i]) free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    if (tok->scores) free(tok->scores);
    if (tok->hash_table) free(tok->hash_table);
    memset(tok, 0, sizeof(*tok));
}
