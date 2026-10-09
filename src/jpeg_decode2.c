#include "jpeg_decode2.h"
#include <turbojpeg.h>
#include <stdint.h>
#include <stdlib.h>

int ei_jpeg_decode2(const unsigned char *data, size_t size, size_t max_pixels,
                    size_t max_bytes, unsigned char **rgb, int *width, int *height) {
    *rgb = NULL;
    if (size < 3 || data[0] != 0xff || data[1] != 0xd8 || data[2] != 0xff) return 0;
    tjhandle decoder = tj3Init(TJINIT_DECOMPRESS);
    if (!decoder) return -1;
    int result = -1;
    unsigned char *out = NULL;
    if (tj3Set(decoder, TJPARAM_MAXMEMORY, 128) ||
        tj3DecompressHeader(decoder, data, size)) goto done;
    int w = tj3Get(decoder, TJPARAM_JPEGWIDTH);
    int h = tj3Get(decoder, TJPARAM_JPEGHEIGHT);
    uint64_t pixels = w > 0 && h > 0 ? (uint64_t)w * (uint64_t)h : 0;
    if (!pixels || pixels > SIZE_MAX / 3 || (max_pixels && pixels > max_pixels) ||
        (max_bytes && pixels > max_bytes / 3)) goto done;
    int colorspace = tj3Get(decoder, TJPARAM_COLORSPACE);
    if (tj3Get(decoder, TJPARAM_PRECISION) != 8 ||
        (colorspace != TJCS_RGB && colorspace != TJCS_YCbCr && colorspace != TJCS_GRAY)) {
        result = 0;
        goto done;
    }
    out = malloc((size_t)pixels * 3);
    if (!out || tj3Decompress8(decoder, data, size, out, 0, TJPF_RGB)) goto done;
    *rgb = out;
    *width = w;
    *height = h;
    out = NULL;
    result = 1;
done:
    free(out);
    tj3Destroy(decoder);
    return result;
}
