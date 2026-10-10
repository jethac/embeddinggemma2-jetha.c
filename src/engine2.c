/* Model-specific EmbeddingGemma 2 C graph. MIT; algorithm reference:
 * llama.cpp de7fa0a/src/models/gemma-embedding2.cpp (also MIT).
 * Keep the graph resident across identical batch shapes. No autoregressive KV
 * cache: this encoder attends bidirectionally to the complete input. */
#include "engine.h"
#include "media2.h"
#include "jpeg_decode2.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <math.h>
#include <pthread.h>
#include <time.h>

#define HIDDEN 512
#define FF 2048
#define LAYERS 24
#define GRAPH_NODES 4096
#define MAX_GRAPH_CACHE2 4

static void dependency_log(enum ggml_log_level level, const char *text, void *debug) {
    // Tensor inventories and graph-reuse messages can make synchronous log
    // sinks dominate startup/inference. Keep normal diagnostics by default.
    if (level == GGML_LOG_LEVEL_DEBUG && !debug) return;
    if (text) fputs(text, stderr);
}

typedef struct {
    ggml_backend_sched_t sched;
    struct ggml_context *graph_ctx;
    struct ggml_cgraph *graph;
    struct ggml_tensor *input, *positions, *full_mask, *local_mask, *pool, *output;
    struct ggml_context *static_ctx;
    ggml_backend_buffer_t static_buffer;
    size_t *input_offsets;
    bool layout_valid;
    size_t graph_tokens, graph_batch;
    bool graph_raw;
    uint64_t last_used;
} graph2;

typedef struct {
    struct ggml_context *weights;
    ggml_backend_buffer_t weight_buffer;
    struct ggml_context *qkv_ctx;
    ggml_backend_buffer_t qkv_buffer;
    struct ggml_tensor *qkv[LAYERS];
    ggml_backend_t backends[2];
    int n_backends;
    graph2 graphs[1 + MAX_GRAPH_CACHE2]; // Normal workspace plus bounded short-text shapes.
    bool graph_cache;
    bool raw_graph_cache;
    int graph_cache_slots;
    bool text_buckets;
    bool text_batch_buckets;
    bool cuda_global_attn;
    bool cuda_local_attn;
    bool cuda_local_range;
    bool reuse_inputs;
    bool fused_geglu;
    bool media_batch;
    bool media_pipeline;
    bool vision_clip_metadata;
    bool metal_media_flash_attn;
    bool audio_multishape_graph;
    bool jpeg_turbo;
    uint64_t graph_clock;
    int threads;
    bool profile;
    pthread_mutex_t mutex;
    pthread_mutex_t media_mutex;
    pthread_cond_t media_slot;
    size_t media_inflight;
    struct llama_model *vocab_model;
    mtmd_context *media;
} engine2;

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1000000.0;
}

static bool fail(char *err, size_t n, const char *msg) {
    if (err && n) snprintf(err, n, "%s", msg);
    return false;
}

uint64_t ei_engine_cache_fingerprint(const ei_engine *e, uint64_t fingerprint) {
    const engine2 *s = e->gemma2;
    // Numeric variants can change accumulation order. Keep persisted text
    // and HTTP responses separate from the original path and each other.
    const char *domains[] = {
        s->qkv_buffer ? "embeddinggemma2-packed-qkv-v1" : NULL,
        s->cuda_global_attn ? "embeddinggemma2-cuda-global-attn-v2" : NULL,
        s->cuda_local_attn ? "embeddinggemma2-cuda-local-attn-v1" : NULL,
        s->cuda_local_range ? "embeddinggemma2-cuda-local-range-v1" : NULL,
        s->fused_geglu ? "embeddinggemma2-geglu-v1" : NULL,
        s->media_batch ? "embeddinggemma2-media-batch-v1" : NULL,
        s->vision_clip_metadata ? "embeddinggemma2-vision-clip-metadata-v1" : NULL,
        s->metal_media_flash_attn ? "embeddinggemma2-metal-media-flash-attn-v1" : NULL,
        s->audio_multishape_graph ? "embeddinggemma2-audio-multishape-graph-v1" : NULL,
        s->jpeg_turbo ? "embeddinggemma2-jpeg-turbo-3.2.0-v1" : NULL,
        s->text_buckets ? "embeddinggemma2-text-buckets-v1" : NULL,
        s->text_batch_buckets ? "embeddinggemma2-text-batch-buckets-v1" : NULL,
    };
    for (size_t d = 0; d < sizeof domains / sizeof domains[0]; d++) {
        if (!domains[d]) continue;
        for (size_t i = 0; i <= strlen(domains[d]); i++) {
            fingerprint ^= (unsigned char)domains[d][i];
            fingerprint *= 1099511628211ull;
        }
    }
    return fingerprint;
}

static struct ggml_tensor *weight(engine2 *s, const char *name) {
    struct ggml_tensor *t = ggml_get_tensor(s->weights, name);
    if (!t) ei_die("missing EmbeddingGemma 2 tensor %s", name);
    return t;
}

static struct ggml_tensor *layer_weight(engine2 *s, int layer, const char *suffix) {
    char name[128];
    snprintf(name, sizeof name, "blk.%d.%s.weight", layer, suffix);
    return weight(s, name);
}

static struct ggml_tensor *norm(struct ggml_context *ctx, struct ggml_tensor *x,
                               struct ggml_tensor *w, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return w ? ggml_mul(ctx, x, w) : x;
}

static graph2 *select_graph(engine2 *s, size_t tokens, size_t batch, bool raw) {
    if (!s->graph_cache || (raw && !s->raw_graph_cache) || tokens > 256) return &s->graphs[0];
    graph2 *chosen = &s->graphs[1];
    for (int i = 1; i <= s->graph_cache_slots; i++) {
        graph2 *state = &s->graphs[i];
        if (state->graph && state->graph_tokens == tokens &&
            state->graph_batch == batch && state->graph_raw == raw) {
            chosen = state;
            break;
        }
        if (state->last_used < chosen->last_used) chosen = state;
    }
    chosen->last_used = ++s->graph_clock;
    return chosen;
}

static void free_static_inputs(graph2 *state) {
    if (state->static_buffer) ggml_backend_buffer_free(state->static_buffer);
    if (state->static_ctx) ggml_free(state->static_ctx);
    free(state->input_offsets);
    state->static_buffer = NULL;
    state->static_ctx = NULL;
    state->input_offsets = NULL;
    state->layout_valid = false;
}

static bool build_graph(ei_engine *e, graph2 *state, size_t tokens, size_t batch, bool raw,
                        char *err, size_t err_len) {
    engine2 *s = e->gemma2;
    if (!state->sched) {
        state->sched = ggml_backend_sched_new(s->backends, NULL, s->n_backends,
                                              GRAPH_NODES, false, true);
        if (!state->sched) return fail(err, err_len, "cannot initialize graph scheduler");
    }
    if (state->graph && state->graph_tokens == tokens && state->graph_batch == batch && state->graph_raw == raw)
        return true;
    ggml_backend_sched_reset(state->sched);
    if (state->graph_ctx) ggml_free(state->graph_ctx);
    state->graph = NULL;
    free_static_inputs(state);
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false),
        .no_alloc = true,
    };
    struct ggml_context *ctx = state->graph_ctx = ggml_init(params);
    if (!ctx) return fail(err, err_len, "cannot allocate graph metadata");
    struct ggml_cgraph *g = ggml_new_graph_custom(ctx, GRAPH_NODES, false);
    struct ggml_tensor *x;
    if (raw) {
        state->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, (int64_t)tokens);
        x = state->input;
    } else {
        state->input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)tokens);
        x = ggml_scale(ctx, ggml_get_rows(ctx, weight(s, "token_embd.weight"), state->input), sqrtf(HIDDEN));
    }
    ggml_set_name(state->input, "input");
    ggml_set_input(state->input);
    struct ggml_context *inputs_ctx = ctx;
    if (s->reuse_inputs) {
        struct ggml_init_params inputs_params = {
            .mem_size = 4 * ggml_tensor_overhead(), .no_alloc = true,
        };
        inputs_ctx = state->static_ctx = ggml_init(inputs_params);
        if (!inputs_ctx) return fail(err, err_len, "cannot allocate static input metadata");
    }
    state->positions = ggml_new_tensor_1d(inputs_ctx, GGML_TYPE_I32, (int64_t)tokens);
    ggml_set_input(state->positions);
    int64_t padded = ((int64_t)tokens + 31) / 32 * 32;
    // The pinned CUDA 512-wide flash kernel requires a 256-key stride. Pad
    // only above the explicit-attention memory limit; the shorter path is faster.
    int64_t full_keys = s->cuda_global_attn && tokens > 2048 ?
        ((int64_t)tokens + 255) / 256 * 256 : (int64_t)tokens;
    // CUDA's grouped-query and mask-tile scan paths require a 256-key stride.
    // Pad long local attention so its banded mask can skip work on those paths.
    int64_t local_keys = (s->cuda_local_attn || s->cuda_local_range) && tokens >= 1024 ?
        ((int64_t)tokens + 255) / 256 * 256 : (int64_t)tokens;
    enum ggml_type mask_type = s->reuse_inputs ? GGML_TYPE_F16 : GGML_TYPE_F32;
    state->full_mask = ggml_new_tensor_2d(inputs_ctx, mask_type, full_keys, padded);
    state->local_mask = ggml_new_tensor_2d(inputs_ctx, mask_type, local_keys, padded);
    ggml_set_input(state->full_mask);
    ggml_set_input(state->local_mask);
    struct ggml_tensor *full = s->reuse_inputs ? state->full_mask : ggml_cast(ctx, state->full_mask, GGML_TYPE_F16);
    struct ggml_tensor *local = s->reuse_inputs ? state->local_mask : ggml_cast(ctx, state->local_mask, GGML_TYPE_F16);
    state->pool = ggml_new_tensor_2d(inputs_ctx, GGML_TYPE_F32, (int64_t)tokens, (int64_t)batch);
    ggml_set_input(state->pool);
    if (s->reuse_inputs) {
        // Scheduler-managed input storage may be recycled after its last use.
        // Own these buffers so their contents survive between forwards, and
        // place them on the primary backend to avoid repeated GPU uploads.
        state->static_buffer = ggml_backend_alloc_ctx_tensors(inputs_ctx, s->backends[0]);
        if (!state->static_buffer) return fail(err, err_len, "cannot allocate static input buffers");
    }

    struct ggml_tensor *ple = ggml_scale(ctx,
        ggml_mul_mat(ctx, weight(s, "per_layer_model_proj.weight"), x), 1.0f / sqrtf(HIDDEN));
    ple = ggml_reshape_3d(ctx, ple, HIDDEN, LAYERS, (int64_t)tokens);
    ple = norm(ctx, ple, weight(s, "per_layer_proj_norm.weight"), e->model.rms_eps);

    for (int il = 0; il < LAYERS; il++) {
        bool swa = ei_layer_is_swa(&e->model, il);
        int dim = swa ? 256 : 512;
        int heads_kv = swa ? 2 : 1;
        float base = swa ? e->model.rope_base_swa : e->model.rope_base_full;
        struct ggml_tensor *a = norm(ctx, x, layer_weight(s, il, "attn_norm"), e->model.rms_eps);
        struct ggml_tensor *q, *k, *v;
        if (s->qkv[il]) {
            struct ggml_tensor *qkv = ggml_mul_mat(ctx, s->qkv[il], a);
            const size_t head_stride = (size_t)dim * sizeof(float);
            const size_t q_bytes = head_stride * 4;
            const size_t k_bytes = head_stride * (size_t)heads_kv;
            q = ggml_view_3d(ctx, qkv, dim, 4, (int64_t)tokens, head_stride, qkv->nb[1], 0);
            k = ggml_view_3d(ctx, qkv, dim, heads_kv, (int64_t)tokens, head_stride, qkv->nb[1], q_bytes);
            v = ggml_view_3d(ctx, qkv, dim, heads_kv, (int64_t)tokens, head_stride, qkv->nb[1], q_bytes + k_bytes);
        } else {
            q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, layer_weight(s, il, "attn_q"), a), dim, 4, (int64_t)tokens);
            k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, layer_weight(s, il, "attn_k"), a), dim, heads_kv, (int64_t)tokens);
            v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, layer_weight(s, il, "attn_v"), a), dim, heads_kv, (int64_t)tokens);
        }
        q = norm(ctx, q, layer_weight(s, il, "attn_q_norm"), e->model.rms_eps);
        k = norm(ctx, k, layer_weight(s, il, "attn_k_norm"), e->model.rms_eps);
        v = norm(ctx, v, NULL, e->model.rms_eps);
        q = ggml_rope_ext(ctx, q, state->positions, NULL, dim, GGML_ROPE_TYPE_NEOX, EI_N_CTX,
                          base, 1, 0, 1, 0, 0);
        k = ggml_rope_ext(ctx, k, state->positions, NULL, dim, GGML_ROPE_TYPE_NEOX, EI_N_CTX,
                          base, 1, 0, 1, 0, 0);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        struct ggml_tensor *k32 = ggml_permute(ctx, k, 0, 2, 1, 3);
        struct ggml_tensor *v32 = ggml_permute(ctx, v, 0, 2, 1, 3);
        k = ggml_cast(ctx, k32, GGML_TYPE_F16);
        v = ggml_cast(ctx, v32, GGML_TYPE_F16);
        a = ggml_flash_attn_ext(ctx, q, k, v, swa ? local : full, 1.0f, 0, 0);
        ggml_prec_set_acc(a, GGML_PREC_F32);
        int64_t keys = swa ? local_keys : full_keys;
        if (((swa && (s->cuda_local_attn || s->cuda_local_range)) || (!swa && s->cuda_global_attn)) && keys > (int64_t)tokens) {
            k = ggml_cast(ctx, ggml_pad(ctx, k32, 0, (int)(keys - tokens), 0, 0), GGML_TYPE_F16);
            v = ggml_cast(ctx, ggml_pad(ctx, v32, 0, (int)(keys - tokens), 0, 0), GGML_TYPE_F16);
            struct ggml_tensor *padded_attn = ggml_flash_attn_ext(ctx, q, k, v, swa ? local : full, 1.0f, 0, 0);
            ggml_prec_set_acc(padded_attn, GGML_PREC_F32);
            if (ggml_backend_supports_op(s->backends[0], padded_attn)) a = padded_attn;
        } else if (!swa && s->cuda_global_attn && tokens <= 2048 &&
            !ggml_backend_supports_op(s->backends[0], a)) {
            // CUDA's 512-wide flash kernel requires aligned K/V lengths.
            // Bound the explicit four-head score matrix to 64 MiB, and keep
            // larger batches on streaming attention, padding K/V above.
            struct ggml_tensor *scores = ggml_mul_mat(ctx, k, q);
            ggml_prec_set_acc(scores, GGML_PREC_F32);
            scores = ggml_soft_max_ext_inplace(ctx, scores, state->full_mask, 1.0f, 0);
            a = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), scores);
            ggml_prec_set_acc(a, GGML_PREC_F32);
            a = ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3));
        }
        if (swa && s->cuda_local_range && tokens >= 1024)
            ggml_flash_attn_ext_set_mask_range(a, true);
        a = ggml_reshape_2d(ctx, a, dim * 4, (int64_t)tokens);
        a = ggml_mul_mat(ctx, layer_weight(s, il, "attn_output"), a);
        x = ggml_add(ctx, x, norm(ctx, a, layer_weight(s, il, "post_attention_norm"), e->model.rms_eps));
        a = norm(ctx, x, layer_weight(s, il, "ffn_norm"), e->model.rms_eps);
        struct ggml_tensor *gate = ggml_mul_mat(ctx, layer_weight(s, il, "ffn_gate"), a);
        if (!s->fused_geglu) gate = ggml_gelu(ctx, gate);
        struct ggml_tensor *up = ggml_mul_mat(ctx, layer_weight(s, il, "ffn_up"), a);
        struct ggml_tensor *ffn = s->fused_geglu ? ggml_geglu_split(ctx, gate, up) : ggml_mul(ctx, gate, up);
        a = ggml_mul_mat(ctx, layer_weight(s, il, "ffn_down"), ffn);
        x = ggml_add(ctx, x, norm(ctx, a, layer_weight(s, il, "post_ffw_norm"), e->model.rms_eps));
        a = ggml_mul_mat(ctx, layer_weight(s, il, "inp_gate"), x);
        if (!s->fused_geglu) a = ggml_gelu(ctx, a);
        struct ggml_tensor *p = ggml_view_2d(ctx, ple, HIDDEN, (int64_t)tokens,
                                           ple->nb[2], (size_t)il * ple->nb[1]);
        struct ggml_tensor *layer_input = s->fused_geglu ? ggml_geglu_split(ctx, a, p) : ggml_mul(ctx, a, p);
        a = ggml_mul_mat(ctx, layer_weight(s, il, "proj"), layer_input);
        x = ggml_add(ctx, x, norm(ctx, a, layer_weight(s, il, "post_norm"), e->model.rms_eps));
        x = ggml_mul(ctx, x, layer_weight(s, il, "layer_output_scale"));
    }
    x = norm(ctx, x, weight(s, "output_norm.weight"), e->model.rms_eps);
    /* Linear projection commutes with mean pooling. Pool 512-wide activations
     * first to avoid projecting every token into 768 dimensions. */
    x = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)), state->pool);
    state->output = ggml_mul_mat(ctx, weight(s, "output.weight"), x);
    ggml_set_name(state->output, "embedding");
    ggml_set_output(state->output);
    ggml_build_forward_expand(g, state->output);
    if (!ggml_backend_sched_alloc_graph(state->sched, g))
        return fail(err, err_len, "cannot allocate inference graph");
    state->graph = g;
    state->graph_tokens = tokens;
    state->graph_batch = batch;
    state->graph_raw = raw;
    if (s->graph_cache && state != &s->graphs[0]) {
        size_t workspace = 0;
        for (int slot = 1; slot <= s->graph_cache_slots; slot++) {
            if (!s->graphs[slot].sched) continue;
            for (int backend = 0; backend < s->n_backends; backend++)
                workspace += ggml_backend_sched_get_buffer_size(s->graphs[slot].sched,
                                                                s->backends[backend]);
        }
        fprintf(stderr, "CUDA short-text graph cache: %zu tokens, %zu sequences, %.2f MiB total cached workspace\n",
                tokens, batch, (double)workspace / (1024 * 1024));
    }
    return true;
}

static void prepare_layout(ei_engine *e, graph2 *state, const size_t *offsets, size_t batch) {
    engine2 *s = e->gemma2;
    size_t actual = offsets[batch];
    size_t n = state->graph_tokens;
    size_t padded = (n + 31) / 32 * 32;
    size_t full_keys = (size_t)state->full_mask->ne[0];
    size_t local_keys = (size_t)state->local_mask->ne[0];
    int32_t *pos = ei_xmalloc(n * sizeof *pos);
    int32_t *seq = ei_xmalloc(n * sizeof *seq);
    size_t mask_keys = full_keys > local_keys ? full_keys : local_keys;
    void *mask = ei_xmalloc(mask_keys * padded * (s->reuse_inputs ? sizeof(uint16_t) : sizeof(float)));
    float *pool = ei_xcalloc(n * batch, sizeof *pool);
    for (size_t b = 0; b < batch; b++) {
        for (size_t t = offsets[b]; t < offsets[b + 1]; t++) {
            pos[t] = (int32_t)(t - offsets[b]);
            seq[t] = (int32_t)b;
            pool[b * n + t] = 1.0f / (float)(offsets[b + 1] - offsets[b]);
        }
    }
    // Padded queries need finite attention too: 0 * NaN would poison pooling.
    // Isolate them in a separate sequence, with zero pooling weights.
    for (size_t t = actual; t < n; t++) {
        pos[t] = 0;
        seq[t] = (int32_t)batch;
    }
    ggml_backend_tensor_set(state->positions, pos, 0, n * sizeof *pos);
    ggml_backend_tensor_set(state->pool, pool, 0, n * batch * sizeof *pool);
    if (s->reuse_inputs) {
        // These masks only contain zero and negative infinity. Their IEEE
        // half representations are exact; no per-element conversion is needed.
        uint16_t *half = mask;
        for (size_t q = 0; q < padded; q++) {
            for (size_t k = 0; k < n; k++)
                half[q * full_keys + k] = q < n && seq[q] == seq[k] ? 0 : 0xfc00;
            for (size_t k = n; k < full_keys; k++) half[q * full_keys + k] = 0xfc00;
        }
        ggml_backend_tensor_set(state->full_mask, half, 0, full_keys * padded * sizeof *half);
        if (full_keys == local_keys) {
            for (size_t q = 0; q < n; q++)
                for (size_t k = 0; k < n; k++)
                    if (abs(pos[q] - pos[k]) > (int)e->model.swa_window / 2) half[q * local_keys + k] = 0xfc00;
        } else {
            for (size_t q = 0; q < padded; q++) {
                for (size_t k = 0; k < n; k++)
                    half[q * local_keys + k] = q < n && seq[q] == seq[k] &&
                        abs(pos[q] - pos[k]) <= (int)e->model.swa_window / 2 ? 0 : 0xfc00;
                for (size_t k = n; k < local_keys; k++) half[q * local_keys + k] = 0xfc00;
            }
        }
        ggml_backend_tensor_set(state->local_mask, half, 0, local_keys * padded * sizeof *half);
    } else {
        float *full_mask = mask;
        for (size_t q = 0; q < padded; q++) {
            for (size_t k = 0; k < n; k++)
                full_mask[q * full_keys + k] = q < n && seq[q] == seq[k] ? 0.0f : -INFINITY;
            for (size_t k = n; k < full_keys; k++) full_mask[q * full_keys + k] = -INFINITY;
        }
        ggml_backend_tensor_set(state->full_mask, full_mask, 0, full_keys * padded * sizeof *full_mask);
        if (full_keys == local_keys) {
            for (size_t q = 0; q < n; q++)
                for (size_t k = 0; k < n; k++)
                    if (abs(pos[q] - pos[k]) > (int)e->model.swa_window / 2) full_mask[q * local_keys + k] = -INFINITY;
        } else {
            for (size_t q = 0; q < padded; q++) {
                for (size_t k = 0; k < n; k++)
                    full_mask[q * local_keys + k] = q < n && seq[q] == seq[k] &&
                        abs(pos[q] - pos[k]) <= (int)e->model.swa_window / 2 ? 0.0f : -INFINITY;
                for (size_t k = n; k < local_keys; k++) full_mask[q * local_keys + k] = -INFINITY;
            }
        }
        ggml_backend_tensor_set(state->local_mask, full_mask, 0, local_keys * padded * sizeof *full_mask);
    }
    free(pos); free(seq); free(mask); free(pool);
    if (s->reuse_inputs) {
        if (!state->input_offsets) state->input_offsets = ei_xmalloc((batch + 1) * sizeof *offsets);
        memcpy(state->input_offsets, offsets, (batch + 1) * sizeof *offsets);
        state->layout_valid = true;
    }
}

static bool compute(ei_engine *e, const void *input, bool raw, const size_t *offsets,
                    size_t batch, float *out, char *err, size_t err_len) {
    engine2 *s = e->gemma2;
    size_t n = offsets[batch];
    size_t graph_n = n;
    if (!raw && n <= 256 && ((s->text_buckets && batch == 1) ||
                            (s->text_batch_buckets && batch > 1))) {
        graph_n = 32;
        while (graph_n < n) graph_n *= 2;
    }
    graph2 *state = select_graph(s, graph_n, batch, raw);
    double started = s->profile ? now_ms() : 0;
    bool rebuilt = !state->graph || state->graph_tokens != graph_n || state->graph_batch != batch || state->graph_raw != raw;
    if (!build_graph(e, state, graph_n, batch, raw, err, err_len)) return false;
    double built = s->profile ? now_ms() : 0;
    bool reused = s->reuse_inputs && state->layout_valid &&
        memcmp(state->input_offsets, offsets, (batch + 1) * sizeof *offsets) == 0;
    int32_t padded_ids[256];
    if (graph_n != n) {
        memcpy(padded_ids, input, n * sizeof *padded_ids);
        memset(padded_ids + n, 0, (graph_n - n) * sizeof *padded_ids);
        ggml_backend_tensor_set(state->input, padded_ids, 0, graph_n * sizeof *padded_ids);
    } else {
        ggml_backend_tensor_set(state->input, input, 0, n * (raw ? HIDDEN * sizeof(float) : sizeof(int32_t)));
    }
    if (s->profile && rebuilt)
        fprintf(stderr, "backbone auxiliary inputs: positions=%s full_mask=%s local_mask=%s pool=%s, %.2f MiB dedicated\n",
                ggml_backend_buffer_name(state->positions->buffer), ggml_backend_buffer_name(state->full_mask->buffer),
                ggml_backend_buffer_name(state->local_mask->buffer), ggml_backend_buffer_name(state->pool->buffer),
                state->static_buffer ? (double)ggml_backend_buffer_get_size(state->static_buffer) / (1024 * 1024) : 0);
    if (!reused) prepare_layout(e, state, offsets, batch);
    double prepared = s->profile ? now_ms() : 0;
    if (ggml_backend_sched_graph_compute(state->sched, state->graph) != GGML_STATUS_SUCCESS) {
        state->layout_valid = false;
        return fail(err, err_len, "inference failed");
    }
    double computed = s->profile ? now_ms() : 0;
    ggml_backend_tensor_get(state->output, out, 0, batch * EI_N_EMBD * sizeof *out);
    for (size_t b = 0; b < batch; b++) {
        float *row = out + b * EI_N_EMBD;
        double energy = 0;
        for (int j = 0; j < EI_N_EMBD; j++) {
            if (!isfinite(row[j])) return fail(err, err_len, "model produced non-finite embeddings");
            energy += (double)row[j] * row[j];
        }
        if (energy <= 0) return fail(err, err_len, "model produced a zero embedding");
        ei_l2_normalize(row, EI_N_EMBD);
    }
    if (s->profile)
        fprintf(stderr, "backbone: tokens=%zu batch=%zu raw=%d rebuilt=%d layout_reused=%d build=%.3f prep=%.3f compute=%.3f output=%.3f ms\n",
                n, batch, raw, rebuilt, reused, built - started, prepared - built,
                computed - prepared, now_ms() - computed);
    return true;
}

void ei_engine_load_backend(ei_engine *e, const char *path, const char *backend) {
    const char *debug_logs = getenv("EI_DEBUG_LOG2");
    void *log_debug = debug_logs && strcmp(debug_logs, "1") == 0 ? (void *)"1" : NULL;
    ggml_log_set(dependency_log, log_debug);
    llama_log_set(dependency_log, log_debug);
    // Helpers have a separate logger for synchronous probe/frame messages.
    mtmd_helper_log_set(dependency_log, log_debug);
    memset(e, 0, sizeof *e);
    ei_model_load(&e->model, path);
    ggml_backend_load_all();
    ei_tokenizer_init(&e->tokenizer, &e->model);
    engine2 *s = e->gemma2 = ei_xcalloc(1, sizeof *s);
    s->profile = getenv("EI_PROFILE_BACKBONE2") != NULL;
    pthread_mutex_init(&s->mutex, NULL);
    pthread_mutex_init(&s->media_mutex, NULL);
    pthread_cond_init(&s->media_slot, NULL);
    s->threads = 6;
    const char *threads = getenv("EI_THREADS");
    if (threads) {
        char *end;
        long value = strtol(threads, &end, 10);
        if (*end || value < 1 || value > 256) ei_die("EI_THREADS must be 1..256");
        s->threads = (int)value;
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!cpu) ei_die("no compatible CPU backend");
    const char *requested = backend ? backend : "auto";
    if (strcmp(requested, "cpu") != 0) {
        ggml_backend_dev_t device = NULL;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            const char *name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(d));
            if (strcmp(requested, "auto") == 0 ||
                (strcmp(requested, "cuda") == 0 && strcmp(name, "CUDA") == 0) ||
                (strcmp(requested, "metal") == 0 && strcmp(name, "MTL") == 0) ||
                ((strcmp(requested, "rocm") == 0 || strcmp(requested, "hip") == 0) && strcmp(name, "ROCm") == 0) ||
                ((strcmp(requested, "xpu") == 0 || strcmp(requested, "sycl") == 0) && strcmp(name, "SYCL") == 0)) {
                device = d;
                break;
            }
        }
        if (device) s->backends[s->n_backends++] = ggml_backend_dev_init(device, NULL);
        else if (strcmp(requested, "auto")) ei_die("requested backend %s is unavailable", requested);
    }
    s->backends[s->n_backends++] = cpu;
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    ggml_backend_set_n_threads_t set_threads = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
    if (!set_threads) ei_die("CPU backend cannot configure threads");
    set_threads(cpu, s->threads);
    e->backend_name = ggml_backend_name(s->backends[0]);
    struct ggml_init_params params = {
        .mem_size = ggml_tensor_overhead() * (size_t)e->model.gguf.n_tensors,
        .no_alloc = true,
    };
    s->weights = ggml_init(params);
    if (!s->weights) ei_die("cannot allocate model metadata");
    for (uint64_t i = 0; i < e->model.gguf.n_tensors; i++) {
        const ei_tensor *t = e->model.gguf.tensors + i;
        int64_t ne[4];
        for (int j = 0; j < 4; j++) ne[j] = (int64_t)t->ne[j];
        struct ggml_tensor *w = ggml_new_tensor(s->weights, (enum ggml_type)t->type, (int)t->n_dims, ne);
        if (s->n_backends > 1 && w->type == GGML_TYPE_BF16 &&
            !ggml_backend_supports_op(s->backends[0], w)) {
            // Some Metal devices cannot even host a BF16 leaf in their scheduler.
            // Widen its values exactly rather than placing an unsupported tensor
            // in accelerator memory. Contiguous BF16 strides double for FP32.
            w->type = GGML_TYPE_F32;
            for (int j = 0; j < 4; j++) w->nb[j] *= 2;
            fprintf(stderr, "accelerator weight: widening %.*s from BF16 to FP32\n",
                    (int)t->name.len, t->name.str);
        }
        char name[128];
        if (t->name.len >= sizeof name) ei_die("tensor name too long");
        memcpy(name, t->name.str, (size_t)t->name.len); name[t->name.len] = 0;
        ggml_set_name(w, name);
    }
    if (s->n_backends == 1) {
        s->weight_buffer = ggml_backend_cpu_buffer_from_ptr((void *)e->model.gguf.map, e->model.gguf.map_len);
        for (struct ggml_tensor *w = ggml_get_first_tensor(s->weights); w; w = ggml_get_next_tensor(s->weights, w)) {
            w->buffer = s->weight_buffer;
            w->data = (void *)ei_gguf_tensor(&e->model.gguf, w->name, true)->data;
        }
    } else {
        s->weight_buffer = ggml_backend_alloc_ctx_tensors(s->weights, s->backends[0]);
        if (!s->weight_buffer) ei_die("cannot allocate accelerator weights");
        for (struct ggml_tensor *w = ggml_get_first_tensor(s->weights); w; w = ggml_get_next_tensor(s->weights, w)) {
            const ei_tensor *source = ei_gguf_tensor(&e->model.gguf, w->name, true);
            if (source->type == GGML_TYPE_BF16 && w->type == GGML_TYPE_F32) {
                float *expanded = ei_xmalloc(ggml_nbytes(w));
                ggml_bf16_to_fp32_row((const ggml_bf16_t *)source->data,
                                     expanded, ggml_nelements(w));
                ggml_backend_tensor_set(w, expanded, 0, ggml_nbytes(w));
                free(expanded);
            } else {
                ggml_backend_tensor_set(w, source->data, 0, ggml_nbytes(w));
            }
        }
    }
    ggml_backend_buffer_set_usage(s->weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const char *qkv_flag = getenv("EI_QKV2");
    if (qkv_flag && strcmp(qkv_flag, "1") == 0 &&
        (s->n_backends == 1 || strncmp(e->backend_name, "CUDA", 4) == 0)) {
        for (int il = 0; il < LAYERS; il++) {
            struct ggml_tensor *q = layer_weight(s, il, "attn_q");
            struct ggml_tensor *k = layer_weight(s, il, "attn_k");
            struct ggml_tensor *v = layer_weight(s, il, "attn_v");
            if (q->type != GGML_TYPE_Q8_0 || k->type != q->type || v->type != q->type ||
                q->ne[0] != HIDDEN || k->ne[0] != HIDDEN || v->ne[0] != HIDDEN)
                ei_die("EI_QKV2 requires matching Q8_0 projection weights");
        }
        struct ggml_init_params qkv_params = {
            .mem_size = ggml_tensor_overhead() * LAYERS, .no_alloc = true,
        };
        s->qkv_ctx = ggml_init(qkv_params);
        if (!s->qkv_ctx) ei_die("cannot allocate packed QKV metadata");
        for (int il = 0; il < LAYERS; il++) {
            struct ggml_tensor *q = layer_weight(s, il, "attn_q");
            struct ggml_tensor *k = layer_weight(s, il, "attn_k");
            struct ggml_tensor *v = layer_weight(s, il, "attn_v");
            s->qkv[il] = ggml_new_tensor_2d(s->qkv_ctx, q->type, HIDDEN, q->ne[1] + k->ne[1] + v->ne[1]);
        }
        s->qkv_buffer = ggml_backend_alloc_ctx_tensors(s->qkv_ctx, s->backends[0]);
        if (!s->qkv_buffer) ei_die("cannot allocate packed QKV weights");
        for (int il = 0; il < LAYERS; il++) {
            const char *suffixes[] = {"attn_q", "attn_k", "attn_v"};
            size_t offset = 0;
            for (int part = 0; part < 3; part++) {
                struct ggml_tensor *w = layer_weight(s, il, suffixes[part]);
                const ei_tensor *source = ei_gguf_tensor(&e->model.gguf, w->name, true);
                ggml_backend_tensor_set(s->qkv[il], source->data, offset, ggml_nbytes(w));
                offset += ggml_nbytes(w);
            }
        }
        ggml_backend_buffer_set_usage(s->qkv_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        fprintf(stderr, "packed QKV: %.2f MiB additional weights\n",
                (double)ggml_backend_buffer_get_size(s->qkv_buffer) / (1024 * 1024));
    }
    s->graphs[0].sched = ggml_backend_sched_new(s->backends, NULL, s->n_backends, GRAPH_NODES, false, true);
    if (!s->graphs[0].sched) ei_die("cannot initialize graph scheduler");
    const char *cache = getenv("EI_GRAPH_CACHE2");
    s->graph_cache = cache && strcmp(cache, "1") == 0 && strncmp(e->backend_name, "CUDA", 4) == 0;
    s->graph_cache_slots = 2;
    if (s->graph_cache) {
        const char *slots = getenv("EI_GRAPH_CACHE_SLOTS2");
        if (slots) {
            char *end = NULL;
            long count = strtol(slots, &end, 10);
            if (end == slots || *end || count < 2 || count > MAX_GRAPH_CACHE2)
                ei_die("EI_GRAPH_CACHE_SLOTS2 must be 2..%d", MAX_GRAPH_CACHE2);
            s->graph_cache_slots = (int)count;
        }
        fprintf(stderr, "CUDA short-text graph cache: %d shapes, at most 256 tokens\n", s->graph_cache_slots);
    }
    const char *raw_graph_cache = getenv("EI_RAW_GRAPH_CACHE2");
    s->raw_graph_cache = s->graph_cache && raw_graph_cache && strcmp(raw_graph_cache, "1") == 0;
    if (s->raw_graph_cache) fprintf(stderr, "CUDA raw-input graph cache: sharing the bounded short-input slots\n");
    const char *audio_cache = getenv("EI_AUDIO_GRAPH_CACHE2");
    const char *audio_slots = getenv("EI_AUDIO_GRAPH_CACHE_SLOTS2");
    s->audio_multishape_graph = audio_cache && strcmp(audio_cache, "1") == 0 &&
        audio_slots && audio_slots[0] >= '2' && audio_slots[0] <= '4' && !audio_slots[1] &&
        strncmp(e->backend_name, "CUDA", 4) == 0;
    const char *text_buckets = getenv("EI_TEXT_BUCKETS2");
    s->text_buckets = text_buckets && strcmp(text_buckets, "1") == 0 &&
        strncmp(e->backend_name, "CUDA", 4) == 0;
    if (s->text_buckets) fprintf(stderr, "CUDA text buckets: single sequences up to 256 tokens\n");
    const char *text_batch_buckets = getenv("EI_TEXT_BATCH_BUCKETS2");
    s->text_batch_buckets = text_batch_buckets && strcmp(text_batch_buckets, "1") == 0 &&
        strncmp(e->backend_name, "CUDA", 4) == 0;
    if (s->text_batch_buckets) fprintf(stderr, "CUDA text batch buckets: at most 256 aggregate tokens\n");
    const char *global_attn = getenv("EI_CUDA_GLOBAL_ATTN2");
    s->cuda_global_attn = global_attn && strcmp(global_attn, "1") == 0 &&
        strncmp(e->backend_name, "CUDA", 4) == 0;
    if (s->cuda_global_attn) fprintf(stderr, "CUDA global attention fallback: explicit scores through 2048 aggregate tokens; padded flash above\n");
    const char *local_attn = getenv("EI_CUDA_LOCAL_ATTN2");
    s->cuda_local_attn = local_attn && strcmp(local_attn, "1") == 0 &&
        strncmp(e->backend_name, "CUDA", 4) == 0;
    if (s->cuda_local_attn) fprintf(stderr, "CUDA local attention: padded keys from 1024 aggregate tokens\n");
    const char *local_range = getenv("EI_CUDA_LOCAL_RANGE2");
    s->cuda_local_range = local_range && strcmp(local_range, "1") == 0 &&
        strncmp(e->backend_name, "CUDA", 4) == 0;
    if (s->cuda_local_range) fprintf(stderr, "CUDA local mask range: skip fully masked leading key tiles from 1024 tokens\n");
    const char *reuse_inputs = getenv("EI_REUSE_INPUTS2");
    s->reuse_inputs = reuse_inputs && strcmp(reuse_inputs, "1") == 0;
    if (s->reuse_inputs) fprintf(stderr, "Backbone static input reuse: dedicated FP16 masks, matching sequence boundaries\n");
    const char *geglu = getenv("EI_GEGLU2");
    s->fused_geglu = geglu && strcmp(geglu, "1") == 0 &&
        strncmp(e->backend_name, "CUDA", 4) == 0;
    if (s->fused_geglu) fprintf(stderr, "Fused GeGLU: FFN and layer-input GELU/multiply\n");
    const char *media_batch = getenv("EI_MEDIA_BATCH2");
    s->media_batch = media_batch && strcmp(media_batch, "1") == 0;
    if (s->media_batch) fprintf(stderr, "Multimodal backbone batching: up to 1024 tokens, inputs up to 512\n");
    const char *media_pipeline = getenv("EI_MEDIA_PIPELINE2");
    s->media_pipeline = media_pipeline && strcmp(media_pipeline, "1") == 0;
    const char *vision_clip_metadata = getenv("EI_VISION_CLIP_METADATA2");
    s->vision_clip_metadata = vision_clip_metadata && strcmp(vision_clip_metadata, "1") == 0;
    if (s->vision_clip_metadata) fprintf(stderr, "Vision clipping: explicit metadata only\n");
    const char *media_flash_attn = getenv("EI_METAL_MEDIA_FLASH_ATTN2");
    s->metal_media_flash_attn = media_flash_attn && strcmp(media_flash_attn, "1") == 0 &&
        s->n_backends > 1 && strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(
            ggml_backend_get_device(s->backends[0]))), "MTL") == 0;
    if (s->metal_media_flash_attn) fprintf(stderr, "Metal media flash attention: forced on; unsupported ops may use CPU\n");
    if (s->media_pipeline) fprintf(stderr, "Multimodal pipeline: up to two singleton raw inputs\n");
    const char *jpeg_turbo = getenv("EI_JPEG_TURBO2");
    s->jpeg_turbo = jpeg_turbo && strcmp(jpeg_turbo, "1") == 0;
    if (s->jpeg_turbo) fprintf(stderr, "JPEG decoding: libjpeg-turbo 3.2.0\n");
    fprintf(stderr, "EmbeddingGemma 2: %s, %d CPU threads\n", e->backend_name, s->threads);
}

void ei_engine_load(ei_engine *e, const char *path) {
    ei_engine_load_backend(e, path, getenv("EI_BACKEND"));
}

const char *ei_engine_backend(const ei_engine *e) { return e->backend_name; }
int32_t ei_engine_threads(const ei_engine *e) { return ((engine2 *)e->gemma2)->threads; }

bool ei_engine_reserve(ei_engine *e, size_t tokens, size_t batch, char *err, size_t n) {
    (void)e;
    if (!tokens || tokens > 16384 || !batch || batch > 256)
        return fail(err, n, "batch reservation must be 1..16384 tokens and 1..256 requests");
    return true;
}

bool ei_engine_embed_tokens_batch(ei_engine *e, const int32_t *ids, const size_t *offsets,
                                  size_t batch, float *out, char *err, size_t n) {
    if (!ids || !offsets || !batch || offsets[0] != 0 || offsets[batch] > 16384)
        return fail(err, n, "invalid batch");
    for (size_t b = 0; b < batch; b++)
        if (offsets[b + 1] <= offsets[b] || offsets[b + 1] - offsets[b] > EI_N_CTX)
            return fail(err, n, "each input must contain 1..8192 tokens");
    for (size_t i = 0; i < offsets[batch]; i++)
        if (ids[i] < 0 || ids[i] >= EI_VOCAB) return fail(err, n, "invalid token ID");
    engine2 *s = e->gemma2;
    pthread_mutex_lock(&s->mutex);
    bool ok = compute(e, ids, false, offsets, batch, out, err, n);
    pthread_mutex_unlock(&s->mutex);
    return ok;
}

bool ei_engine_embed_tokens(ei_engine *e, const int32_t *ids, size_t tokens,
                            float out[EI_N_EMBD], char *err, size_t n) {
    size_t offsets[] = {0, tokens};
    return ei_engine_embed_tokens_batch(e, ids, offsets, 1, out, err, n);
}

bool ei_engine_embed(ei_engine *e, const char *text, size_t len,
                     float out[EI_N_EMBD], char *err, size_t n) {
    ei_tokens tokens = {0};
    ei_tokenize_spm(&e->tokenizer, text, len, true, false, &tokens);
    bool ok = ei_engine_embed_tokens(e, tokens.ids, tokens.n, out, err, n);
    ei_tokens_free(&tokens);
    return ok;
}

void ei_engine_free(ei_engine *e) {
    engine2 *s = e->gemma2;
    if (s) {
        if (s->media) mtmd_free(s->media);
        for (int i = 0; i <= MAX_GRAPH_CACHE2; i++) {
            if (s->graphs[i].sched) ggml_backend_sched_free(s->graphs[i].sched);
            if (s->graphs[i].graph_ctx) ggml_free(s->graphs[i].graph_ctx);
            free_static_inputs(&s->graphs[i]);
        }
        if (s->qkv_buffer) ggml_backend_buffer_free(s->qkv_buffer);
        if (s->qkv_ctx) ggml_free(s->qkv_ctx);
        ggml_backend_buffer_free(s->weight_buffer);
        ggml_free(s->weights);
        for (int i = 0; i < s->n_backends; i++) ggml_backend_free(s->backends[i]);
        pthread_mutex_destroy(&s->mutex);
        pthread_mutex_destroy(&s->media_mutex);
        pthread_cond_destroy(&s->media_slot);
        free(s);
    }
    ei_tokenizer_free(&e->tokenizer);
    ei_model_free(&e->model);
    memset(e, 0, sizeof *e);
}

bool ei_engine_load_media(ei_engine *e, const char *model_path, const char *mmproj_path,
                          bool load_vision, bool load_audio,
                          char *err, size_t err_len) {
    engine2 *s = e->gemma2;
    (void)model_path;
    s->vocab_model = e->tokenizer.native_model;
    struct mtmd_context_params media_params = mtmd_context_params_default();
    media_params.use_gpu = s->n_backends > 1;
    media_params.device = s->n_backends > 1 ? ggml_backend_get_device(s->backends[0]) : NULL;
    media_params.n_threads = s->threads;
    media_params.warmup = false;
    media_params.load_vision = load_vision;
    media_params.load_audio = load_audio;
    if (s->metal_media_flash_attn) media_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    s->media = mtmd_init_from_file(mmproj_path, s->vocab_model, media_params);
    if (!s->media) return fail(err, err_len, "cannot initialize modality encoders");
    if ((load_vision && !mtmd_support_vision(s->media)) ||
        (load_audio && !mtmd_support_audio(s->media))) {
        mtmd_free(s->media);
        s->media = NULL;
        return fail(err, err_len, "mmproj does not contain the requested encoders");
    }
    fprintf(stderr, "media encoders: vision=%d audio=%d video=%d\n",
            mtmd_support_vision(s->media), mtmd_support_audio(s->media),
            mtmd_helper_support_video(s->media));
    return true;
}

static bool embed_token_rows(const ei_engine *e, const int32_t *ids, size_t n, float *out) {
    const ei_tensor *table = ei_gguf_tensor(&e->model.gguf, "token_embd.weight", true);
    for (size_t i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= EI_VOCAB) return false;
        float *row = out + i * HIDDEN;
        if (table->type == EI_T_Q8_0) {
            ei_dequantize_row_q8_0_scaled(table, ids[i], sqrtf(HIDDEN), row);
        } else {
            const unsigned char *p = (const unsigned char *)table->data + (size_t)ids[i] * ei_tensor_row_bytes(table);
            for (int j = 0; j < HIDDEN; j++) {
                float v;
                if (table->type == EI_T_F32) memcpy(&v, p + j * 4, 4);
                else if (table->type == EI_T_BF16) {
                    uint16_t lo; memcpy(&lo, p + j * 2, 2);
                    uint32_t bits = (uint32_t)lo << 16; memcpy(&v, &bits, 4);
                } else if (table->type == EI_T_F16) {
                    uint16_t h; memcpy(&h, p + j * 2, 2); v = ei_fp16_to_fp32(h);
                } else return false;
                row[j] = v * sqrtf(HIDDEN);
            }
        }
    }
    return true;
}

typedef struct {
    mtmd_helper_video *ctx;
    size_t frames;
    size_t *total_frames;
    bool profile;
    double read_ms;
} video_input2;

static int read_video_frame2(size_t index, void *user, mtmd_bitmap **bitmap, char **text) {
    (void)index;
    video_input2 *video = user;
    double started = video->profile ? now_ms() : 0;
    int result;
    // The generic helper adds "Video:". This model's visual-only input has
    // adjacent image-boundary blocks and no prose or timestamp prefix.
    do {
        result = mtmd_helper_video_read_next(video->ctx, bitmap, text);
        if (*text) { free(*text); *text = NULL; }
    } while (!result && !*bitmap);
    if (video->profile) video->read_ms += now_ms() - started;
    if (!result && *bitmap) {
        if (++video->frames > 32 || ++*video->total_frames > 32) {
            mtmd_bitmap_free(*bitmap);
            *bitmap = NULL;
            return -2;
        }
        mtmd_bitmap_set_patch_budget(*bitmap, 140);
    }
    return result;
}

/* Caller holds media_mutex. Only one decoded input and one bounded pending raw
 * batch are resident; encoder contexts never execute concurrently. */
static bool prepare_parts(ei_engine *e, const ei_media_part *parts, size_t n_parts,
                          float **prepared, size_t *tokens, double *encoder_ms,
                          char *err, size_t err_len) {
    engine2 *s = e->gemma2;
    if (!s->media) return fail(err, err_len, "media encoders are not loaded; start with --mmproj PATH");
    if (!n_parts || n_parts > 64) return fail(err, err_len, "content must contain 1..64 parts");
    // Reject unavailable modalities before decoding or allocating media buffers.
    for (size_t i = 0; i < n_parts; i++) {
        if ((parts[i].type == EI_PART_IMAGE || parts[i].type == EI_PART_VIDEO) &&
            !mtmd_support_vision(s->media)) return fail(err, err_len, "vision encoder is not loaded");
        if (parts[i].type == EI_PART_AUDIO && !mtmd_support_audio(s->media))
            return fail(err, err_len, "audio encoder is not loaded");
    }
    mtmd_input_part *inputs = ei_xcalloc(n_parts, sizeof *inputs);
    const mtmd_input_part *input_refs[64];
    mtmd_input_text *texts = ei_xcalloc(n_parts, sizeof *texts);
    struct mtmd_helper_bitmap_wrapper *media = ei_xcalloc(n_parts, sizeof *media);
    video_input2 videos[64] = {0};
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    float *raw = NULL;
    bool ok = false;
    /* Keep one media request's decoded/preprocessed buffers resident at a time.
     * The media encoder has its own backend/scheduler. Its graph and immutable
     * token-table reads do not use the backbone's scratch buffers; text can
     * progress during decoding and encoding. Lock order is media, then backbone,
     * and text takes only the backbone lock. */
    double start = now_ms();
    bool phase_profile = getenv("EI_PROFILE_MEDIA2") != NULL;
    double decode_end = 0, preprocess_end = 0;
    size_t decoded_bytes = 0;
    size_t video_frames = 0;
    size_t decoded_video_frames = 0;
    const size_t decoded_limit = 128u * 1024u * 1024u;
    for (size_t i = 0; i < n_parts; i++) {
        input_refs[i] = &inputs[i];
        if (parts[i].type == EI_PART_TEXT) {
            texts[i] = (mtmd_input_text){(const char *)parts[i].data, parts[i].size, false, false};
            inputs[i].text = &texts[i];
        } else {
            struct mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
            // Every encoder executes from fresh input. Exact response caching
            // compares the full request; media IDs are never used for reuse.
            opt.compute_id = false;
            if (decoded_bytes >= decoded_limit) {
                fail(err, err_len, "media exceeds 128 MiB decoded input budget"); goto done;
            }
            opt.max_image_pixels = 16777216;
            // Audio produces at least one token per 640 samples at 16 kHz.
            opt.max_audio_samples = (size_t)EI_N_CTX * 640;
            opt.max_decoded_bytes = decoded_limit - decoded_bytes;
            opt.video_params.max_frame_bytes = opt.max_decoded_bytes;
            opt.video_params.probe_timeout_ms = 10000;
            opt.video_params.decode_timeout_ms = 30000;
            opt.video_params.fps_target = parts[i].fps > 0 ? parts[i].fps : 1.0f;
            opt.video_params.timestamp_interval_ms = 0;
            if (parts[i].type == EI_PART_VIDEO) {
                const char *probe = getenv("EI_VIDEO_PROBE2");
                if (probe && *probe) opt.video_params.probe_bin = probe;
                media[i].video_ctx = mtmd_helper_video_init_from_buf(s->media, parts[i].data, parts[i].size, opt.video_params);
                if (!media[i].video_ctx) { fail(err, err_len, "cannot decode video input"); goto done; }
                struct mtmd_helper_video_info info = mtmd_helper_video_get_info(media[i].video_ctx);
                if (!info.width || !info.height || (uint64_t)info.width * info.height > 16777216 ||
                    info.n_frames <= 0 || (size_t)info.n_frames > 32 - video_frames) {
                    fail(err, err_len, "input must contain at most 32 video frames of at most 16 megapixels"); goto done;
                }
                video_frames += (size_t)info.n_frames;
                decoded_bytes += (size_t)info.width * info.height * 3;
                videos[i].ctx = media[i].video_ctx;
                videos[i].profile = phase_profile;
                videos[i].total_frames = &decoded_video_frames;
                media[i].bitmap = mtmd_bitmap_init_lazy(s->media, NULL, &videos[i], read_video_frame2);
            } else {
                unsigned char *rgb = NULL;
                int width = 0, height = 0;
                int decoded = parts[i].type == EI_PART_IMAGE && s->jpeg_turbo
                    ? ei_jpeg_decode2(parts[i].data, parts[i].size, opt.max_image_pixels,
                                      opt.max_decoded_bytes, &rgb, &width, &height) : 0;
                if (decoded < 0) { fail(err, err_len, "cannot decode JPEG or decoded size limit exceeded"); goto done; }
                if (decoded > 0) {
                    media[i].bitmap = mtmd_bitmap_init(width, height, rgb);
                    free(rgb);
                } else {
                    media[i] = mtmd_helper_bitmap_init_from_buf(s->media, parts[i].data, parts[i].size, false, opt);
                }
            }
            if (!media[i].bitmap) { fail(err, err_len, "cannot decode media input or decoded size limit exceeded"); goto done; }
            if (parts[i].type != EI_PART_VIDEO && (media[i].video_ctx ||
                ((parts[i].type == EI_PART_AUDIO) != mtmd_bitmap_is_audio(media[i].bitmap)))) {
                fail(err, err_len, "media bytes do not match content type"); goto done;
            }
            if (parts[i].type != EI_PART_VIDEO) decoded_bytes += mtmd_bitmap_get_n_bytes(media[i].bitmap);
            if (parts[i].type == EI_PART_IMAGE) mtmd_bitmap_set_patch_budget(media[i].bitmap, 280);
            inputs[i].bitmap = media[i].bitmap;
        }
    }
    if (phase_profile) decode_end = now_ms();
    if (mtmd_tokenize_from_parts(s->media, chunks, input_refs, n_parts, true) != 0) {
        fail(err, err_len, "media preprocessing failed"); goto done;
    }
    size_t count = mtmd_helper_get_n_tokens(chunks);
    if (phase_profile) preprocess_end = now_ms();
    if (!count || count > EI_N_CTX) { fail(err, err_len, "multimodal input exceeds 8192-token context"); goto done; }
    raw = ei_xmalloc(count * HIDDEN * sizeof *raw);
    size_t cursor = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
        size_t nt = mtmd_input_chunk_get_n_tokens(chunk);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t text_n;
            const int32_t *ids = mtmd_input_chunk_get_tokens_text(chunk, &text_n);
            if (text_n != nt || !embed_token_rows(e, ids, nt, raw + cursor * HIDDEN)) {
                fail(err, err_len, "invalid media boundary tokens"); goto done;
            }
        } else {
            if (mtmd_encode_chunk(s->media, chunk) != 0) {
                fail(err, err_len, "modality encoder failed"); goto done;
            }
            memcpy(raw + cursor * HIDDEN, mtmd_get_output_embd(s->media), nt * HIDDEN * sizeof *raw);
        }
        cursor += nt;
    }
    *encoder_ms = now_ms() - start;
    if (phase_profile) fprintf(stderr, "media phases: decode %.2f ms, preprocess %.2f ms, encode/assemble %.2f ms\n",
                              decode_end - start, preprocess_end - decode_end,
                              *encoder_ms - (preprocess_end - start));
    if (phase_profile && decoded_video_frames) {
        double read_ms = 0;
        for (size_t i = 0; i < n_parts; i++) read_ms += videos[i].read_ms;
        // Lazy video reads happen inside preprocessing. Separate pipe/decode
        // waits from image preprocessing before choosing the next optimization.
        fprintf(stderr, "video phases: %zu frames, frame reads %.2f ms, remaining preprocess %.2f ms\n",
                decoded_video_frames, read_ms, preprocess_end - decode_end - read_ms);
    }
    *tokens = count;
    *prepared = raw;
    raw = NULL;
    ok = true;
done:
    free(raw);
    mtmd_input_chunks_free(chunks);
    for (size_t i = 0; i < n_parts; i++) {
        if (media[i].bitmap) mtmd_bitmap_free(media[i].bitmap);
        if (media[i].video_ctx) mtmd_helper_video_free(media[i].video_ctx);
    }
    free(media); free(texts); free(inputs);
    return ok;
}

static bool compute_media(ei_engine *e, const float *raw, const size_t *offsets,
                           size_t batch, float *out, double *elapsed,
                           char *err, size_t err_len) {
    engine2 *s = e->gemma2;
    double start = now_ms();
    pthread_mutex_lock(&s->mutex);
    bool ok = compute(e, raw, true, offsets, batch, out, err, err_len);
    pthread_mutex_unlock(&s->mutex);
    *elapsed += now_ms() - start;
    return ok;
}

bool ei_engine_embed_parts(ei_engine *e, const ei_media_part *parts, size_t n_parts,
                           float out[EI_N_EMBD], size_t *tokens, double *encoder_ms,
                           double *backbone_ms, char *err, size_t err_len) {
    engine2 *s = e->gemma2;
    float *raw = NULL;
    *backbone_ms = 0;
    pthread_mutex_lock(&s->media_mutex);
    if (s->media_pipeline) {
        // Reserve before decoding. Two owned raw inputs cost at most 32 MiB;
        // request-array batching retains its own bounded 18 MiB workspace.
        // Only one input's decoded media is resident under media_mutex.
        while (s->media_inflight == 2)
            pthread_cond_wait(&s->media_slot, &s->media_mutex);
        s->media_inflight++;
    }
    bool ok = prepare_parts(e, parts, n_parts, &raw, tokens, encoder_ms, err, err_len);
    if (s->media_pipeline) pthread_mutex_unlock(&s->media_mutex);
    if (ok) {
        size_t offsets[] = {0, *tokens};
        ok = compute_media(e, raw, offsets, 1, out, backbone_ms, err, err_len);
    }
    free(raw);
    if (s->media_pipeline) {
        pthread_mutex_lock(&s->media_mutex);
        s->media_inflight--;
        pthread_cond_signal(&s->media_slot);
    }
    pthread_mutex_unlock(&s->media_mutex);
    return ok;
}

bool ei_engine_media_batch_enabled(const ei_engine *e) {
    return ((const engine2 *)e->gemma2)->media_batch;
}

bool ei_engine_embed_parts_batch(ei_engine *e, const ei_media_part *const *parts,
                                 const size_t *n_parts, size_t batch, float *out,
                                 size_t *tokens, double *encoder_ms, double *backbone_ms,
                                 char *err, size_t err_len) {
    if (!parts || !n_parts || !batch || batch > 256)
        return fail(err, err_len, "media batch must contain 1..256 inputs");
    engine2 *s = e->gemma2;
    // Larger groups lose to extra masked attention work in paired CUDA trials.
    // Keep at most 2 MiB pending rows plus the current input (at most 16 MiB).
    // Long inputs keep the original full-context forward unchanged.
    const size_t limit = 1024, sequence_limit = 512;
    float *pending;
    size_t offsets[257] = {0};
    size_t count = 0, first = 0;
    bool ok = true;
    *encoder_ms = *backbone_ms = 0;
    pthread_mutex_lock(&s->media_mutex);
    pending = ei_xmalloc(limit * HIDDEN * sizeof *pending);
    for (size_t i = 0; ok && i < batch; i++) {
        float *raw = NULL;
        double elapsed = 0;
        ok = prepare_parts(e, parts[i], n_parts[i], &raw, &tokens[i], &elapsed, err, err_len);
        *encoder_ms += elapsed;
        if (!ok) { free(raw); break; }
        if (count && (offsets[count] + tokens[i] > limit || tokens[i] > sequence_limit)) {
            ok = compute_media(e, pending, offsets, count, out + first * EI_N_EMBD,
                               backbone_ms, err, err_len);
            count = 0;
            offsets[0] = 0;
        }
        if (ok && tokens[i] > sequence_limit) {
            size_t single[] = {0, tokens[i]};
            ok = compute_media(e, raw, single, 1, out + i * EI_N_EMBD,
                               backbone_ms, err, err_len);
        } else if (ok) {
            if (!count) first = i;
            memcpy(pending + offsets[count] * HIDDEN, raw, tokens[i] * HIDDEN * sizeof *raw);
            offsets[count + 1] = offsets[count] + tokens[i];
            count++;
        }
        free(raw);
    }
    if (ok && count) ok = compute_media(e, pending, offsets, count, out + first * EI_N_EMBD,
                                        backbone_ms, err, err_len);
    pthread_mutex_unlock(&s->media_mutex);
    free(pending);
    return ok;
}
