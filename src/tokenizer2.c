/* Gemma 4 BPE through the native tokenizer C API; not SentencePiece. MIT. */
#include "tokenizer.h"
#include "llama.h"
#include <limits.h>

void ei_tokenizer_init(ei_tokenizer *tok, const ei_model *model) {
    memset(tok, 0, sizeof *tok);
    tok->model = model;
    struct llama_model_params params = llama_model_default_params();
    params.no_alloc = true;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.n_gpu_layers = 0;
    tok->native_model = llama_model_load_from_file(model->source_path, params);
    if (!tok->native_model) ei_die("cannot initialize Gemma 4 tokenizer");
}

void ei_tokenizer_free(ei_tokenizer *tok) {
    if (tok->native_model) llama_model_free(tok->native_model);
    memset(tok, 0, sizeof *tok);
}

void ei_tokenize_spm(const ei_tokenizer *tok, const char *text, size_t len,
                     bool add_special, bool parse_special, ei_tokens *out) {
    memset(out, 0, sizeof *out);
    if (len > INT32_MAX) return;
    const struct llama_vocab *vocab = llama_model_get_vocab(tok->native_model);
    int32_t count = llama_tokenize(vocab, text, (int32_t)len, NULL, 0, add_special, parse_special);
    if (count == INT32_MIN) return;
    if (count < 0) count = -count;
    out->ids = ei_xmalloc((size_t)count * sizeof *out->ids);
    int32_t result = llama_tokenize(vocab, text, (int32_t)len, out->ids, count, add_special, parse_special);
    if (result < 0) ei_die("Gemma 4 tokenizer sizing changed");
    out->n = (size_t)result;
}

void ei_tokens_free(ei_tokens *tokens) {
    free(tokens->ids);
    memset(tokens, 0, sizeof *tokens);
}

int32_t ei_tokenizer_text_to_id(const ei_tokenizer *tok, const char *text, size_t len) {
    const struct llama_vocab *vocab = llama_model_get_vocab(tok->native_model);
    int32_t id;
    if (len > INT32_MAX || llama_tokenize(vocab, text, (int32_t)len, &id, 1, false, true) != 1) return -1;
    return id;
}
