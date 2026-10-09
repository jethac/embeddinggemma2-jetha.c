/* Restart-cache safety: exact API keys, identity, bounded reads, damaged files
 * and atomic replacement. Any failure blocks enabling persistence. */
#define _POSIX_C_SOURCE 200809L
#include "response_cache.h"
#include "cache_fingerprint.h"
#include <unistd.h>
#include "windows_compat.h"

static int failures;
static void check(bool ok, const char *name) {
    if (!ok) { fprintf(stderr, "response persistence: %s\n", name); failures++; }
}
static bool has(ei_response_cache *c, const char *key, const char *expected) {
    ei_response_cache_value value;
    if (!ei_response_cache_acquire(c, key, strlen(key), &value)) return false;
    bool ok = value.len == strlen(expected) && !memcmp(value.data, expected, value.len);
    ei_response_cache_release(c, &value);
    return ok;
}
static ei_response_cache *load(const char *path, uint64_t identity) {
    ei_response_cache *c = ei_response_cache_create(4096, 16);
    ei_response_cache_load(c, path, identity);
    return c;
}
int main(void) {
    char path[] = "response-persistence-check2.XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) ei_die("cannot create cache test file");
    close(fd);
    ei_response_cache *c = ei_response_cache_create(4096, 16);
    ei_response_cache_insert(c, "\2image", 6, "native", 6);
    ei_response_cache_insert(c, "\3image", 6, "openai", 6);
    ei_response_cache_save(c, path, 123);
    ei_response_cache *other = load(path, 123);
    check(has(other, "\2image", "native") && has(other, "\3image", "openai"), "API keys or responses changed");
    check(!has(other, "\2changed", "native"), "different media reused a result");
    ei_response_cache_free(other);
    other = load(path, 124);
    check(!has(other, "\2image", "native"), "stale identity was admitted");
    ei_response_cache_free(other); ei_response_cache_free(c);

    c = ei_response_cache_create(4096, 16);
    ei_response_cache_insert(c, "new", 3, "replacement", 11);
    ei_response_cache_save(c, path, 123);
    other = load(path, 123);
    check(has(other, "new", "replacement") && !has(other, "\2image", "native"), "replacement retained stale records");
    ei_response_cache_free(other);
    FILE *file = fopen(path, "r+b");
    if (!file) ei_die("cannot alter cache test file");
    fseek(file, 43, SEEK_SET); // first value byte after header, record and 3-byte key
    fputc('X', file); fclose(file);
    other = load(path, 123);
    check(!has(other, "new", "replacement"), "damaged response was admitted");
    ei_response_cache_free(other);

    ei_response_cache_save(c, path, 123);
    file = fopen(path, "r+b");
    uint64_t too_large = UINT64_MAX;
    fseek(file, 16, SEEK_SET); fwrite(&too_large, sizeof too_large, 1, file); fclose(file);
    other = load(path, 123);
    check(!has(other, "new", "replacement"), "oversized declared key was admitted");
    ei_response_cache_free(other);
    file = fopen(path, "wb"); fwrite("EIHTTP02", 1, 8, file); fclose(file);
    other = load(path, 123);
    check(!has(other, "new", "replacement"), "truncated header was admitted");
    ei_response_cache_free(other); ei_response_cache_free(c);
    file = fopen(path, "wb");
    for (size_t i = 0; i < 80000; i++) fputc('a', file);
    fclose(file);
    uint64_t before = ei_cache_fingerprint_file(path);
    file = fopen(path, "r+b"); fseek(file, 70000, SEEK_SET); fputc('b', file); fclose(file);
    check(before != ei_cache_fingerprint_file(path), "weight changes beyond the GGUF header reused identity");
    unlink(path);
    if (!failures) puts("Response persistence safety checks passed");
    return failures ? 1 : 0;
}
