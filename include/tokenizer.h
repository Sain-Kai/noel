#ifndef AETHER_TOKENIZER_H
#define AETHER_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "gguf_loader.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char **vocab;
    float *scores;
    int vocab_size;
    int bos_id;
    int eos_id;
    int pad_id;
    int *hash_table;
    int hash_size;
} aether_tokenizer_t;

/* Initialize standard LLaMA/Mistral compatible tokenizer */
bool aether_tokenizer_init(aether_tokenizer_t *tok, int vocab_size);

/* Load vocabulary directly from GGUF metadata */
bool aether_tokenizer_load_gguf(aether_tokenizer_t *tok, const noel_gguf_t *gguf);

/* Fast string lookup for token ID */
int32_t aether_tokenizer_lookup(const aether_tokenizer_t *tok, const char *str, size_t len);

/* Encode prompt string into token array using greedy prefix matching */
int aether_tokenizer_encode(const aether_tokenizer_t *tok, const char *text, int32_t *tokens, int max_tokens);

/* Decode token id into raw string */
const char *aether_tokenizer_decode(const aether_tokenizer_t *tok, int32_t token_id);

/* Decode token id into clean, human-readable UTF-8 piece (stripping BPE/SentencePiece markers) */
const char *aether_tokenizer_decode_piece(const aether_tokenizer_t *tok, int32_t token_id, char *out_buf, size_t max_len);

/* Free tokenizer memory */
void aether_tokenizer_free(aether_tokenizer_t *tok);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_TOKENIZER_H */
