/* Safety: optional IDs must preserve default SHA-256 and decoded bounds/data. */
#include "mtmd-helper.h"
#include <cstdio>
#include <cstring>

int main() {
    const unsigned char image[] = {'P','6','\n','2',' ','1','\n','2','5','5','\n',220,30,30,30,220,30};
    auto opt = mtmd_helper_init_opt_default();
    if (!opt.compute_id) return 1;
    auto hashed = mtmd_helper_bitmap_init_from_buf(nullptr, image, sizeof image, false, opt);
    if (!hashed.bitmap || std::strcmp(mtmd_bitmap_get_id(hashed.bitmap),
        "be69bd94b81121955756770d60ea0ef1b3bf383f26d920d23fd31fa2f0bba219")) return 1;
    opt.compute_id = false;
    auto plain = mtmd_helper_bitmap_init_from_buf(nullptr, image, sizeof image, false, opt);
    if (!plain.bitmap || *mtmd_bitmap_get_id(plain.bitmap) ||
        mtmd_bitmap_get_nx(plain.bitmap) != 2 || mtmd_bitmap_get_ny(plain.bitmap) != 1 ||
        mtmd_bitmap_get_n_bytes(plain.bitmap) != 6 ||
        std::memcmp(mtmd_bitmap_get_data(hashed.bitmap), mtmd_bitmap_get_data(plain.bitmap), 6)) return 1;
    mtmd_bitmap_free(hashed.bitmap);
    mtmd_bitmap_free(plain.bitmap);
    opt.max_image_pixels = 1;
    auto limited = mtmd_helper_bitmap_init_from_buf(nullptr, image, sizeof image, false, opt);
    if (limited.bitmap) return 1;
    std::puts("Default media SHA-256 ID, opt-out bytes and pixel bound: passed");
    return 0;
}
