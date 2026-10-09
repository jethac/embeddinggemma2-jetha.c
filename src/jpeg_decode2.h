#ifndef EI_JPEG_DECODE2_H
#define EI_JPEG_DECODE2_H
#include <stddef.h>
/* 1: owned RGB bytes, 0: retain the existing decoder, -1: invalid/over budget. */
int ei_jpeg_decode2(const unsigned char *data, size_t size, size_t max_pixels,
                    size_t max_bytes, unsigned char **rgb, int *width, int *height);
#endif
