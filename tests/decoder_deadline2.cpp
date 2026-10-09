/* Safety invariant: timeouts cannot become clean EOF or partial embeddings. */
#include "mtmd-helper.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#ifdef _WIN32
#include <windows.h>
static void mode(const char *value) { _putenv_s("EI_TEST_DECODER_MODE", value); }
#else
#include <unistd.h>
static void mode(const char *value) { setenv("EI_TEST_DECODER_MODE", value, 1); }
#endif
static int failures;
static void check(bool ok, const char *name) {
    if (!ok) { std::fprintf(stderr, "decoder deadline: %s\n", name); failures++; }
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    auto params = mtmd_helper_video_init_params_default();
    params.ffmpeg_bin_dir = argv[1];
    params.timestamp_interval_ms = 0;
    params.probe_timeout_ms = 1000;
    params.decode_timeout_ms = 500;
    const unsigned char input[] = {'c','l','i','p'};
    for (const char *test : {"probe", "decode", "partial", "normal"}) {
        mode(test);
        auto start = std::chrono::steady_clock::now();
        auto *ctx = mtmd_helper_video_init_from_buf(nullptr, input, sizeof input, params);
        if (test[0] == 'p' && test[1] == 'r') {
            check(!ctx, "stalled probe was accepted");
        } else if (!ctx) {
            check(false, "valid probe failed before decode");
        } else {
            int result = 0, frames = 0;
            do {
                mtmd_bitmap *bitmap = nullptr; char *text = nullptr;
                result = mtmd_helper_video_read_next(ctx, &bitmap, &text);
                if (bitmap) { frames++; mtmd_bitmap_free(bitmap); }
                std::free(text);
            } while (!result);
            if (test[0] == 'n') check(result == -1 && frames == 2, "normal decode did not finish at clean EOF");
            else {
                check(result == -2, "timeout was mistaken for clean EOF");
                check(frames == (test[0] == 'p' ? 1 : 0), "unexpected partial frame count");
            }
        }
        mtmd_helper_video_free(ctx);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count();
        check(ms < 2500, "subprocess exceeded the configured deadline");
        std::printf("%s decoder check: %lld ms\n", test, (long long)ms);
    }
    return failures ? 1 : 0;
}
