/* Bounded media admission and duplicate singleflight. MIT.
 * Keep decoded media out of waiting duplicates: the leader alone parses and
 * executes the request. Unique leaders use the engine's media mutex; backbone
 * execution shares the text mutex. Raw backbone batching is separate work. */
#include "media_service2.h"
#include <pthread.h>

typedef struct media_job media_job;
struct media_job {
    uint64_t hash;
    size_t key_len, users;
    bool openai, ready;
    ei_media_result result;
    char *response;
    char error[256];
    pthread_cond_t done;
    media_job *next;
    char key[];
};

struct ei_media_service {
    ei_engine *engine;
    size_t max_batch, max_jobs, max_key_bytes;
    size_t jobs, key_bytes;
    bool stopping;
    media_job *buckets[128];
    pthread_mutex_t mutex;
    pthread_cond_t idle;
};

static size_t job_bytes(const media_job *job) {
    return sizeof(*job) + job->key_len + 1;
}

static uint64_t request_hash(const char *body, size_t len, bool openai) {
    uint64_t h = (1469598103934665603ull ^ (openai ? 3u : 2u)) * 1099511628211ull;
    for (size_t i = 0; i < len; i++) {
        h ^= (unsigned char)body[i];
        h *= 1099511628211ull;
    }
    return h;
}

static media_job **find_job(ei_media_service *s, uint64_t hash,
                            const char *body, size_t len, bool openai) {
    media_job **slot = &s->buckets[hash & 127u];
    while (*slot) {
        media_job *job = *slot;
        if (job->hash == hash && job->key_len == len && job->openai == openai &&
            memcmp(job->key, body, len) == 0) break;
        slot = &job->next;
    }
    return slot;
}

ei_media_service *ei_media_service_create(ei_engine *engine, size_t max_batch,
                                         size_t max_jobs, size_t max_key_bytes) {
    if (!engine || !max_batch || !max_jobs || !max_key_bytes) return NULL;
    ei_media_service *s = ei_xcalloc(1, sizeof(*s));
    s->engine = engine;
    s->max_batch = max_batch;
    s->max_jobs = max_jobs;
    s->max_key_bytes = max_key_bytes;
    if (pthread_mutex_init(&s->mutex, NULL) || pthread_cond_init(&s->idle, NULL))
        ei_die("cannot initialize media admission");
    return s;
}

void ei_media_service_free(ei_media_service *s) {
    if (!s) return;
    pthread_mutex_lock(&s->mutex);
    s->stopping = true;
    while (s->jobs) pthread_cond_wait(&s->idle, &s->mutex);
    pthread_mutex_unlock(&s->mutex);
    pthread_cond_destroy(&s->idle);
    pthread_mutex_destroy(&s->mutex);
    free(s);
}

ei_media_result ei_media_service_submit(ei_media_service *s,
    const char *body, size_t len, bool openai, char **response,
    char *err, size_t err_len) {
    *response = NULL;
    uint64_t hash = request_hash(body, len, openai);
    pthread_mutex_lock(&s->mutex);
    if (s->stopping) {
        snprintf(err, err_len, "media service is shutting down");
        pthread_mutex_unlock(&s->mutex);
        return EI_MEDIA_BUSY;
    }
    media_job **slot = find_job(s, hash, body, len, openai);
    media_job *job = *slot;
    if (job) {
        job->users++;
        while (!job->ready) pthread_cond_wait(&job->done, &s->mutex);
    } else {
        if (len > SIZE_MAX - sizeof(*job) - 1 || s->jobs >= s->max_jobs ||
            sizeof(*job) + len + 1 > s->max_key_bytes - s->key_bytes) {
            snprintf(err, err_len, "media work queue is full; retry later");
            pthread_mutex_unlock(&s->mutex);
            return EI_MEDIA_BUSY;
        }
        job = ei_xcalloc(1, sizeof(*job) + len + 1);
        job->hash = hash;
        job->key_len = len;
        job->openai = openai;
        job->users = 1;
        memcpy(job->key, body, len);
        if (pthread_cond_init(&job->done, NULL)) ei_die("cannot initialize media singleflight");
        *slot = job;
        s->jobs++;
        s->key_bytes += job_bytes(job);
        pthread_mutex_unlock(&s->mutex);

        char *text = NULL;
        char error[256] = {0};
        bool ok = ei_multimodal_request(s->engine, job->key, len, openai,
                                        s->max_batch, &text, error, sizeof(error));
        pthread_mutex_lock(&s->mutex);
        job->result = ok ? EI_MEDIA_OK : EI_MEDIA_INVALID;
        job->response = ok ? text : NULL;
        if (!ok) {
            free(text);
            snprintf(job->error, sizeof(job->error), "%s",
                     error[0] ? error : "multimodal inference failed");
        }
        job->ready = true;
        pthread_cond_broadcast(&job->done);
    }
    ei_media_result result = job->result;
    if (result == EI_MEDIA_OK) {
        if (job->users == 1) {
            *response = job->response;
            job->response = NULL;
        } else {
            size_t n = strlen(job->response) + 1;
            *response = ei_xmalloc(n);
            memcpy(*response, job->response, n);
        }
    } else {
        snprintf(err, err_len, "%s", job->error);
    }
    if (--job->users == 0) {
        slot = find_job(s, hash, job->key, len, openai);
        *slot = job->next;
        s->jobs--;
        s->key_bytes -= job_bytes(job);
        pthread_cond_destroy(&job->done);
        free(job->response);
        free(job);
        if (!s->jobs) pthread_cond_broadcast(&s->idle);
    }
    pthread_mutex_unlock(&s->mutex);
    return result;
}
