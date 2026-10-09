/* In-memory request indexing only. Full keys remain the equality check. MIT. */
#ifndef EI_REQUEST_HASH2_H
#define EI_REQUEST_HASH2_H
#include <stddef.h>
#include <stdint.h>
#define XXH_INLINE_ALL
#include "hash/xxhash/xxhash.h"

static inline uint64_t ei_request_hash2(const char *data, size_t len, uint64_t seed) {
    return XXH3_64bits_withSeed(data, len, seed);
}
#endif
