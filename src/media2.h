#ifndef EI_MEDIA2_H
#define EI_MEDIA2_H
#include "engine.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef enum { EI_PART_TEXT, EI_PART_IMAGE, EI_PART_AUDIO, EI_PART_VIDEO } ei_part_type;
typedef struct {
    ei_part_type type;
    const unsigned char *data;
    size_t size;
    float fps;
} ei_media_part;

bool ei_engine_load_media(ei_engine *e, const char *model_path, const char *mmproj_path,
                          char *err, size_t err_len);
bool ei_engine_embed_parts(ei_engine *e, const ei_media_part *parts, size_t n_parts,
                           float out[EI_N_EMBD], size_t *tokens, double *encoder_ms,
                           double *backbone_ms, char *err, size_t err_len);
bool ei_multimodal_request(ei_engine *e, const char *body, size_t body_len, bool openai,
                           size_t max_batch, char **response, char *err, size_t err_len);

#ifdef __cplusplus
}
#endif
#endif
