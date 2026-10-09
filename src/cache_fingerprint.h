/* Runtime cache identity for the complete weight file. MIT. */
#ifndef EI_CACHE_FINGERPRINT_H
#define EI_CACHE_FINGERPRINT_H
#include "common.h"

static inline uint64_t ei_cache_fingerprint_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) ei_die("cannot open cached model weights: %s", path);
    uint64_t hash = 1469598103934665603ull, bytes = 0;
    unsigned char buffer[65536];
    size_t got;
    while ((got = fread(buffer, 1, sizeof buffer, file)) != 0) {
        bytes += got;
        for (size_t i = 0; i < got; i++) {
            hash ^= buffer[i]; hash *= 1099511628211ull;
        }
    }
    if (ferror(file)) ei_die("cannot fingerprint model weights: %s", path);
    fclose(file);
    for (int i = 0; i < 8; i++) {
        hash ^= (bytes >> (i * 8)) & 0xffu; hash *= 1099511628211ull;
    }
    return hash;
}
#endif
