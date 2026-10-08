/* Small native reference runner for diagnosing graph differences. MIT.
 * Prints one normalized vector for the exact input string supplied. */
#include "llama.h"
#include "ggml-backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s MODEL TEXT\n", argv[0]); return 2; }
    ggml_backend_load_all();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 1024;
    cp.n_threads = 6; cp.n_threads_batch = 6;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int n = -llama_tokenize(vocab, argv[2], (int)strlen(argv[2]), NULL, 0, true, false);
    if (n <= 0 || n > 1024) return 2;
    llama_token *ids = malloc((size_t)n * sizeof *ids);
    if (!ids) return 1;
    if (llama_tokenize(vocab, argv[2], (int)strlen(argv[2]), ids, n, true, false) != n) return 1;
    struct llama_batch_ext *batch = llama_batch_ext_init(ctx);
    for (int i = 0; i < n; i++) {
        int idx = llama_batch_ext_add_token(batch, 0, ids[i]);
        llama_pos pos = i;
        llama_batch_ext_set_pos(batch, idx, &pos);
        llama_batch_ext_set_output_logits(batch, idx, true);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) return 1;
    const float *v = llama_get_embeddings_seq(ctx, 0);
    if (!v) return 1;
    double energy = 0;
    for (int i = 0; i < 768; i++) energy += (double)v[i] * v[i];
    printf("[");
    for (int i = 0; i < 768; i++) printf("%s%.9g", i ? "," : "", v[i] / sqrt(energy));
    printf("]\n");
    llama_batch_ext_free(batch);
    free(ids);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
