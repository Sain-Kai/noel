#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    /* Free any previous mock vocabulary */
    aether_tokenizer_free(tok);

    tok->vocab = tokens;
    tok->scores = scores;
    tok->vocab_size = vocab_size;
    tok->bos_id = gguf->bos_token_id >= 0 ? gguf->bos_token_id : 1;
    tok->eos_id = gguf->eos_token_id >= 0 ? gguf->eos_token_id : 2;
    tok->pad_id = gguf->pad_token_id >= 0 ? gguf->pad_token_id : 0;

    printf("[Tokenizer] Loaded %d embedded vocabulary tokens from GGUF (BOS=%d, EOS=%d)\n",
           vocab_size, tok->bos_id, tok->eos_id);
    return true;
}

int aether_tokenizer_encode(const aether_tokenizer_t *tok, const char *text, int32_t *tokens, int max_tokens) {
    int count = 0;
    if (tok->bos_id >= 0 && count < max_tokens) {
        tokens[count++] = tok->bos_id;
    }

    int len = strlen(text);
    for (int i = 0; i < len && count < max_tokens; i++) {
        uint8_t byte = (uint8_t)text[i];
        tokens[count++] = (int32_t)byte;
    }

    return count;
}

const char *aether_tokenizer_decode(const aether_tokenizer_t *tok, int32_t token_id) {
    if (token_id < 0 || token_id >= tok->vocab_size || !tok->vocab[token_id]) {
        return "";
    }
    return tok->vocab[token_id];
}

void aether_tokenizer_free(aether_tokenizer_t *tok) {
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            if (tok->vocab[i]) free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    if (tok->scores) free(tok->scores);
    memset(tok, 0, sizeof(*tok));
}
