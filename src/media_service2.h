#ifndef EI_MEDIA_SERVICE2_H
#define EI_MEDIA_SERVICE2_H
#include "media2.h"
#include "response_cache.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ei_media_service ei_media_service;
typedef enum { EI_MEDIA_OK, EI_MEDIA_INVALID, EI_MEDIA_BUSY } ei_media_result;

/* Bound unique pending work by count and encoded key bytes. Identical callers
 * share one in-flight result, independently of the completed response cache. */
ei_media_service *ei_media_service_create(ei_engine *engine, size_t max_batch,
                                         size_t max_jobs, size_t max_key_bytes);
/* Call after clients have finished, or wait for existing submissions to finish. */
void ei_media_service_free(ei_media_service *service);
ei_media_result ei_media_service_submit(ei_media_service *service,
    const char *body, size_t body_len, bool openai, ei_response_cache *cache, char **response,
    char *err, size_t err_len);
#ifdef __cplusplus
}
#endif
#endif
