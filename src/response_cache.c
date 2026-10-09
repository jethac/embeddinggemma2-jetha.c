#define _POSIX_C_SOURCE 200809L
#include "response_cache.h"

#include <pthread.h>
#include <unistd.h>
#include "windows_compat.h"
#ifdef EI_GEMMA2
#include "request_hash2.h"
#endif

typedef struct response_cache_entry response_cache_entry;

struct response_cache_entry {
    uint64_t hash;
    size_t key_len;
    size_t value_len;
    size_t users;
    response_cache_entry *hash_next;
    response_cache_entry *lru_prev;
    response_cache_entry *lru_next;
    char data[];
};

struct ei_response_cache {
    size_t max_bytes;
    size_t used_bytes;
    size_t bucket_count;
    response_cache_entry **buckets;
    response_cache_entry *lru_head;
    response_cache_entry *lru_tail;
    pthread_mutex_t mutex;
};

static uint64_t hash_bytes(const char *data, size_t len) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < len; i++) {
        hash ^= (unsigned char)data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint64_t lookup_hash(const char *data, size_t len) {
#ifdef EI_GEMMA2
    return ei_request_hash2(data, len, 0);
#else
    return hash_bytes(data, len);
#endif
}

static size_t next_power_of_two(size_t value) {
    size_t result = 1;
    while (result < value && result <= SIZE_MAX / 2u) result *= 2u;
    return result < value ? value : result;
}

static size_t entry_bytes(const response_cache_entry *entry) {
    return sizeof(*entry) + entry->key_len + entry->value_len + 1u;
}

static char *entry_value(response_cache_entry *entry) {
    return entry->data + entry->key_len;
}

static response_cache_entry **entry_slot(ei_response_cache *cache,
                                         uint64_t hash,
                                         const char *key, size_t key_len) {
    response_cache_entry **slot =
        &cache->buckets[hash & (cache->bucket_count - 1u)];
    while (*slot) {
        response_cache_entry *entry = *slot;
        if (entry->hash == hash && entry->key_len == key_len &&
            memcmp(entry->data, key, key_len) == 0) {
            break;
        }
        slot = &entry->hash_next;
    }
    return slot;
}

static void lru_remove(ei_response_cache *cache,
                       response_cache_entry *entry) {
    if (entry->lru_prev) entry->lru_prev->lru_next = entry->lru_next;
    else cache->lru_head = entry->lru_next;
    if (entry->lru_next) entry->lru_next->lru_prev = entry->lru_prev;
    else cache->lru_tail = entry->lru_prev;
    entry->lru_prev = NULL;
    entry->lru_next = NULL;
}

static void lru_push_front(ei_response_cache *cache,
                           response_cache_entry *entry) {
    if (cache->lru_head == entry) return;
    if (entry->lru_prev || entry->lru_next || cache->lru_tail == entry) {
        lru_remove(cache, entry);
    }
    entry->lru_prev = NULL;
    entry->lru_next = cache->lru_head;
    if (cache->lru_head) cache->lru_head->lru_prev = entry;
    else cache->lru_tail = entry;
    cache->lru_head = entry;
}

static void remove_entry(ei_response_cache *cache,
                         response_cache_entry *entry) {
    response_cache_entry **slot = entry_slot(
        cache, entry->hash, entry->data, entry->key_len);
    if (*slot == entry) *slot = entry->hash_next;
    lru_remove(cache, entry);
    cache->used_bytes -= entry_bytes(entry);
    free(entry);
}

ei_response_cache *ei_response_cache_create(size_t max_bytes,
                                            size_t bucket_count) {
    if (max_bytes == 0) return NULL;
    ei_response_cache *cache = ei_xcalloc(1, sizeof(*cache));
    cache->max_bytes = max_bytes;
    cache->bucket_count = next_power_of_two(bucket_count > 0 ? bucket_count : 64);
    cache->buckets = ei_xcalloc(cache->bucket_count, sizeof(*cache->buckets));
    if (pthread_mutex_init(&cache->mutex, NULL) != 0) {
        ei_die("failed to initialize HTTP response cache");
    }
    return cache;
}

void ei_response_cache_free(ei_response_cache *cache) {
    if (!cache) return;
    response_cache_entry *entry = cache->lru_head;
    while (entry) {
        response_cache_entry *next = entry->lru_next;
        free(entry);
        entry = next;
    }
    pthread_mutex_destroy(&cache->mutex);
    free(cache->buckets);
    free(cache);
}

bool ei_response_cache_acquire(ei_response_cache *cache,
                               const char *key, size_t key_len,
                               ei_response_cache_value *value) {
    if (!cache || !key || !value) return false;
    uint64_t hash = lookup_hash(key, key_len);
    pthread_mutex_lock(&cache->mutex);
    response_cache_entry *entry = *entry_slot(cache, hash, key, key_len);
    if (!entry) {
        pthread_mutex_unlock(&cache->mutex);
        return false;
    }
    entry->users++;
    lru_push_front(cache, entry);
    *value = (ei_response_cache_value){
        .data = entry_value(entry), .len = entry->value_len, .entry = entry,
    };
    pthread_mutex_unlock(&cache->mutex);
    return true;
}

void ei_response_cache_release(ei_response_cache *cache,
                               ei_response_cache_value *value) {
    if (!cache || !value || !value->entry) return;
    pthread_mutex_lock(&cache->mutex);
    response_cache_entry *entry = value->entry;
    entry->users--;
    pthread_mutex_unlock(&cache->mutex);
    *value = (ei_response_cache_value){0};
}

void ei_response_cache_insert(ei_response_cache *cache,
                              const char *key, size_t key_len,
                              const char *value, size_t value_len) {
    if (!cache || !key || !value ||
        key_len > SIZE_MAX - value_len - sizeof(response_cache_entry) - 1u) {
        return;
    }
    size_t bytes = sizeof(response_cache_entry) + key_len + value_len + 1u;
    if (bytes > cache->max_bytes) return;

    response_cache_entry *candidate = ei_xcalloc(1, bytes);
    candidate->hash = lookup_hash(key, key_len);
    candidate->key_len = key_len;
    candidate->value_len = value_len;
    memcpy(candidate->data, key, key_len);
    memcpy(entry_value(candidate), value, value_len);
    entry_value(candidate)[value_len] = '\0';

    pthread_mutex_lock(&cache->mutex);
    response_cache_entry **slot = entry_slot(
        cache, candidate->hash, key, key_len);
    if (*slot) {
        pthread_mutex_unlock(&cache->mutex);
        free(candidate);
        return;
    }
    while (cache->used_bytes > cache->max_bytes - bytes) {
        response_cache_entry *victim = cache->lru_tail;
        while (victim && victim->users != 0) victim = victim->lru_prev;
        if (!victim) {
            pthread_mutex_unlock(&cache->mutex);
            free(candidate);
            return;
        }
        remove_entry(cache, victim);
    }
    slot = entry_slot(cache, candidate->hash, key, key_len);
    candidate->hash_next = *slot;
    *slot = candidate;
    cache->used_bytes += bytes;
    lru_push_front(cache, candidate);
    pthread_mutex_unlock(&cache->mutex);
}

/* Little-endian restart format: magic[8], identity u64, then records of
 * key_len u64, value_len u64, checksum u64, key bytes, value bytes. Recompute
 * the checksum and ordinary full-key hash on load. A truncated/corrupt record
 * ends the usable prefix; no declared length may exceed the resident budget. */
static const char RESPONSE_MAGIC[8] = {'E','I','H','T','T','P','0','2'};

static uint64_t record_checksum(const char *key, size_t key_len,
                                const char *value, size_t value_len) {
    uint64_t lens[2] = {key_len, value_len};
    uint64_t hash = hash_bytes((const char *)lens, sizeof lens);
    for (size_t i = 0; i < key_len; i++) {
        hash ^= (unsigned char)key[i]; hash *= 1099511628211ull;
    }
    for (size_t i = 0; i < value_len; i++) {
        hash ^= (unsigned char)value[i]; hash *= 1099511628211ull;
    }
    return hash;
}

void ei_response_cache_load(ei_response_cache *cache, const char *path, uint64_t identity) {
    if (!cache || !path) return;
    FILE *file = fopen(path, "rb");
    if (!file) return;
    char magic[8]; uint64_t saved_identity;
    if (fread(magic, 1, sizeof magic, file) != sizeof magic ||
        memcmp(magic, RESPONSE_MAGIC, sizeof magic) ||
        fread(&saved_identity, sizeof saved_identity, 1, file) != 1 || saved_identity != identity) {
        fclose(file); return;
    }
    size_t loaded = 0;
    for (;;) {
        uint64_t key_len, value_len, checksum;
        size_t remaining = cache->max_bytes - cache->used_bytes;
        if (fread(&key_len, sizeof key_len, 1, file) != 1 ||
            fread(&value_len, sizeof value_len, 1, file) != 1 ||
            fread(&checksum, sizeof checksum, 1, file) != 1 ||
            remaining <= sizeof(response_cache_entry) + 1 ||
            !key_len || !value_len || key_len > remaining - sizeof(response_cache_entry) - 1 ||
            value_len > remaining - sizeof(response_cache_entry) - 1 - key_len) break;
        char *data = ei_xmalloc((size_t)(key_len + value_len));
        bool ok = fread(data, 1, (size_t)(key_len + value_len), file) == key_len + value_len &&
            checksum == record_checksum(data, (size_t)key_len, data + key_len, (size_t)value_len) &&
            !memchr(data + key_len, '\0', (size_t)value_len);
        if (ok) {
            ei_response_cache_insert(cache, data, (size_t)key_len, data + key_len, (size_t)value_len);
            loaded++;
        }
        free(data);
        if (!ok) break;
    }
    fclose(file);
    if (loaded) fprintf(stderr, "loaded %zu cached HTTP responses from %s\n", loaded, path);
}

void ei_response_cache_save(ei_response_cache *cache, const char *path, uint64_t identity) {
    if (!cache || !path) return;
    size_t len = strlen(path);
    char *tmp = ei_xmalloc(len + 8);
    memcpy(tmp, path, len); memcpy(tmp + len, ".XXXXXX", 8);
    int fd = mkstemp(tmp);
    if (fd < 0) { free(tmp); return; }
    FILE *file = fdopen(fd, "wb");
    if (!file) { close(fd); unlink(tmp); free(tmp); return; }
    pthread_mutex_lock(&cache->mutex);
    bool ok = fwrite(RESPONSE_MAGIC, 1, sizeof RESPONSE_MAGIC, file) == sizeof RESPONSE_MAGIC &&
              fwrite(&identity, sizeof identity, 1, file) == 1;
    size_t count = 0;
    // Oldest first so loading by insertion restores the existing LRU order.
    for (response_cache_entry *entry = cache->lru_tail; ok && entry; entry = entry->lru_prev) {
        uint64_t lens[2] = {entry->key_len, entry->value_len};
        uint64_t checksum = record_checksum(entry->data, entry->key_len,
                                            entry_value(entry), entry->value_len);
        ok = fwrite(lens, sizeof lens, 1, file) == 1 &&
             fwrite(&checksum, sizeof checksum, 1, file) == 1 &&
             fwrite(entry->data, 1, entry->key_len + entry->value_len, file) == entry->key_len + entry->value_len;
        if (ok) count++;
    }
    pthread_mutex_unlock(&cache->mutex);
    if (fclose(file)) ok = false;
    if (ok && ei_replace_file(tmp, path) == 0)
        fprintf(stderr, "persisted %zu cached HTTP responses to %s\n", count, path);
    else unlink(tmp);
    free(tmp);
}
