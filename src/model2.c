/* EmbeddingGemma 2 metadata. Tensor execution lives in engine2.c. */
#include "model.h"

static const ei_kv *array(const ei_gguf *g, const char *key, uint32_t type) {
    const ei_kv *v = ei_gguf_kv(g, key, true);
    if (v->type != EI_GGUF_ARRAY || v->arr_type != type || v->arr_n != EI_VOCAB)
        ei_die("%s: invalid tokenizer array", key);
    return v;
}

void ei_model_load(ei_model *m, const char *path) {
    memset(m, 0, sizeof *m);
    m->source_path = ei_xmalloc(strlen(path) + 1);
    strcpy(m->source_path, path);
    ei_gguf_open(&m->gguf, path);
    const ei_gguf *g = &m->gguf;
    const ei_kv *arch = ei_gguf_kv(g, "general.architecture", true);
    const char *name = "gemma-embedding2";
    if (arch->type != EI_GGUF_STRING || arch->v.s.len != strlen(name) ||
        memcmp(arch->v.s.str, name, strlen(name)))
        ei_die("%s: expected gemma-embedding2 architecture", path);
    const ei_kv *tok = ei_gguf_kv(g, "tokenizer.ggml.model", true);
    if (tok->type != EI_GGUF_STRING || tok->v.s.len != 6 || memcmp(tok->v.s.str, "gemma4", 6))
        ei_die("%s: expected Gemma 4 BPE tokenizer", path);
    m->rms_eps = (float)ei_gguf_kv_f64(g, "gemma-embedding2.attention.layer_norm_rms_epsilon", 1e-6);
    m->rope_base_full = (float)ei_gguf_kv_f64(g, "gemma-embedding2.rope.freq_base", 1000000);
    m->rope_base_swa = (float)ei_gguf_kv_f64(g, "gemma-embedding2.rope.freq_base_swa", 10000);
    m->swa_window = (uint32_t)ei_gguf_kv_u64(g, "gemma-embedding2.attention.sliding_window", 512);
    m->swa_period = 6;
    if (ei_gguf_kv_u64(g, "gemma-embedding2.block_count", 0) != 24 ||
        ei_gguf_kv_u64(g, "gemma-embedding2.embedding_length", 0) != 512)
        ei_die("%s: unsupported EmbeddingGemma 2 model dimensions", path);
    m->tok_tokens = array(g, "tokenizer.ggml.tokens", EI_GGUF_STRING);
    m->tok_scores = ei_gguf_kv(g, "tokenizer.ggml.scores", false);
    m->tok_types = array(g, "tokenizer.ggml.token_type", EI_GGUF_I32);
    m->bos_id = (int32_t)ei_gguf_kv_u64(g, "tokenizer.ggml.bos_token_id", 2);
    m->eos_id = (int32_t)ei_gguf_kv_u64(g, "tokenizer.ggml.eos_token_id", 1);
    m->unk_id = (int32_t)ei_gguf_kv_u64(g, "tokenizer.ggml.unknown_token_id", 3);
    m->pad_id = (int32_t)ei_gguf_kv_u64(g, "tokenizer.ggml.padding_token_id", 0);
    m->add_bos = ei_gguf_kv_bool(g, "tokenizer.ggml.add_bos_token", true);
    m->add_eos = ei_gguf_kv_bool(g, "tokenizer.ggml.add_eos_token", true);
    m->add_space_prefix = ei_gguf_kv_bool(g, "tokenizer.ggml.add_space_prefix", false);
}

bool ei_layer_is_swa(const ei_model *m, int il) {
    return (uint32_t)il % m->swa_period != m->swa_period - 1;
}

void ei_model_free(ei_model *m) {
    ei_gguf_close(&m->gguf);
    free(m->source_path);
    memset(m, 0, sizeof *m);
}
