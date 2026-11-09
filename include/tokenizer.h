#ifndef AETHER_TOKENIZER_H
#define AETHER_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>

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
} aether_tokenizer_t;

/* Initialize standard LLaMA/Mistral compatible tokenizer */
bool aether_tokenizer_init(aether_tokenizer_t *tok, int vocab_size);

/* Encode prompt string into token array */
int aether_tokenizer_encode(const aether_tokenizer_t *tok, const char *text, int32_t *tokens, int max_tokens);

/* Decode token id into string */
const char *aether_tokenizer_decode(const aether_tokenizer_t *tok, int32_t token_id);

/* Free tokenizer memory */
void aether_tokenizer_free(aether_tokenizer_t *tok);

#ifdef __cplusplus
}
#endif

#endif /* AETHER_TOKENIZER_H */
