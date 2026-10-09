/* Safety invariants at the public admission API. The controlled callback holds
 * real submissions pending so quota checks do not depend on inference speed. */
#define _POSIX_C_SOURCE 200809L
#include "media_service2.h"
#include <pthread.h>
#include <time.h>
#include <errno.h>

static pthread_mutex_t gate = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
static bool released;
static size_t started;
static int failures;

static void check(bool ok, const char *name) {
    if (!ok) { fprintf(stderr, "media admission: %s\n", name); failures++; }
}

bool ei_multimodal_request(ei_engine *engine, const char *body, size_t len,
    bool openai, size_t max_batch, char **response, char *err, size_t err_len) {
    (void)engine; (void)len; (void)max_batch;
    pthread_mutex_lock(&gate);
    started++;
    pthread_cond_broadcast(&changed);
    while (!released) pthread_cond_wait(&changed, &gate);
    pthread_mutex_unlock(&gate);
    if (*body == '!') { snprintf(err, err_len, "controlled failure"); return false; }
    *response = ei_xmalloc(16);
    snprintf(*response, 16, "%s", openai ? "openai" : "native");
    return true;
}

typedef struct {
    ei_media_service *service;
    const char *key;
    bool openai;
    ei_media_result result;
    char *response;
    char error[256];
    pthread_t thread;
} caller;

static void *submit(void *opaque) {
    caller *c = opaque;
    c->result = ei_media_service_submit(c->service, c->key, strlen(c->key),
        c->openai, &c->response, c->error, sizeof(c->error));
    return NULL;
}

static void close_gate(void) {
    pthread_mutex_lock(&gate); released = false; started = 0; pthread_mutex_unlock(&gate);
}
static void open_gate(void) {
    pthread_mutex_lock(&gate); released = true; pthread_cond_broadcast(&changed); pthread_mutex_unlock(&gate);
}
static void wait_started(size_t count) {
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += 5;
    pthread_mutex_lock(&gate);
    while (started < count) {
        if (pthread_cond_timedwait(&changed, &gate, &deadline) == ETIMEDOUT) {
            check(false, "callback did not start (capacity or API namespace)"); break;
        }
    }
    pthread_mutex_unlock(&gate);
}
static void launch(caller *c) {
    if (pthread_create(&c->thread, NULL, submit, c)) ei_die("test thread failed");
}
static void join(caller *c, const char *expected) {
    pthread_join(c->thread, NULL);
    check(c->result == EI_MEDIA_OK && c->response && !strcmp(c->response, expected), "wrong response");
    free(c->response);
}

int main(void) {
    ei_engine engine = {0};
    char *text = NULL, err[256];
    ei_media_service *s = ei_media_service_create(&engine, 256, 1, 4096);
    close_gate();
    caller first = {.service=s, .key="first"}; launch(&first); wait_started(1);
    check(ei_media_service_submit(s, "second", 6, false, &text, err, sizeof(err)) == EI_MEDIA_BUSY,
          "unique job capacity was exceeded");
    open_gate(); join(&first, "native");
    check(ei_media_service_submit(s, "second", 6, false, &text, err, sizeof(err)) == EI_MEDIA_OK,
          "completed job did not release capacity");
    free(text);
    check(ei_media_service_submit(s, "!", 1, false, &text, err, sizeof(err)) == EI_MEDIA_INVALID &&
          text == NULL && !strcmp(err, "controlled failure"), "inference error was not preserved");
    ei_media_service_free(s);

    char large_a[2049], large_b[2049];
    memset(large_a, 'a', 2048); large_a[2048] = 0;
    memset(large_b, 'b', 2048); large_b[2048] = 0;
    s = ei_media_service_create(&engine, 256, 64, 4096);
    close_gate();
    caller large = {.service=s, .key=large_a}; launch(&large); wait_started(1);
    check(ei_media_service_submit(s, large_b, 2048, false, &text, err, sizeof(err)) == EI_MEDIA_BUSY,
          "encoded key byte capacity was exceeded");
    open_gate(); join(&large, "native"); ei_media_service_free(s);

    s = ei_media_service_create(&engine, 256, 64, 4096);
    close_gate();
    caller native = {.service=s, .key="same body"}; launch(&native); wait_started(1);
    caller openai = {.service=s, .key="same body", .openai=true}; launch(&openai); wait_started(2);
    open_gate(); join(&native, "native"); join(&openai, "openai");
    ei_media_service_free(s);
    if (!failures) puts("Media admission, error, and API isolation checks passed");
    return failures ? 1 : 0;
}
