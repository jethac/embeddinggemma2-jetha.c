/* MIT licensed. A small, process-isolated libavformat metadata probe.
 * The parent supplies the input and enforces the existing probe deadline.
 * Keep decoding/sampling in ffmpeg so this path does not change frame pixels. */
#include <libavformat/avformat.h>
#include <libavutil/log.h>
#include <inttypes.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    AVFormatContext *ctx = NULL;
    AVDictionary *opts = NULL;
    av_log_set_level(AV_LOG_QUIET);
    av_dict_set(&opts, "read_ahead_limit", "-1", 0);
    int opened = avformat_open_input(&ctx, argv[argc - 1], NULL, &opts);
    av_dict_free(&opts);
    if (opened < 0) return 1;
    if (avformat_find_stream_info(ctx, NULL) < 0) {
        avformat_close_input(&ctx); return 1;
    }
    // Match ffprobe's -select_streams v:0, rather than the "best" stream.
    AVStream *video = NULL;
    for (unsigned i = 0; i < ctx->nb_streams; i++) {
        if (ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video = ctx->streams[i]; break;
        }
    }
    if (!video) { avformat_close_input(&ctx); return 1; }
    puts("[STREAM]");
    printf("width=%d\nheight=%d\nr_frame_rate=%d/%d\n", video->codecpar->width,
           video->codecpar->height, video->r_frame_rate.num, video->r_frame_rate.den);
    if (video->duration == AV_NOPTS_VALUE) puts("duration=N/A");
    else printf("duration=%.6f\n", video->duration * av_q2d(video->time_base));
    if (video->nb_frames) printf("nb_frames=%" PRId64 "\n", video->nb_frames);
    else puts("nb_frames=N/A");
    puts("[/STREAM]\n[FORMAT]");
    if (ctx->duration == AV_NOPTS_VALUE) puts("duration=N/A");
    else printf("duration=%.6f\n", (double)ctx->duration / AV_TIME_BASE);
    puts("[/FORMAT]");
    avformat_close_input(&ctx);
    return 0;
}
