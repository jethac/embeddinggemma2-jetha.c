/* Controlled decoder process for the observed blocking-pipe regression. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#define PID GetCurrentProcessId()
#else
#include <unistd.h>
#define PID getpid()
#endif
static void stall(void) {
    const char *path = getenv("EI_TEST_DECODER_PID_FILE");
    FILE *file = path ? fopen(path, "w") : NULL;
    if (file) { fprintf(file, "%lu", (unsigned long)PID); fclose(file); }
#ifdef _WIN32
    Sleep(60000);
#else
    sleep(60);
#endif
}
int main(int argc, char **argv) {
    (void)argc;
    const char *mode = getenv("EI_TEST_DECODER_MODE");
    if (strstr(argv[0], "ffprobe")) {
        if (mode && !strcmp(mode, "probe")) stall();
        puts("[STREAM]\nwidth=96\nheight=96\nr_frame_rate=1/1\nnb_frames=2\nduration=2\n[/STREAM]");
        return 0;
    }
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    unsigned char frame[96 * 96 * 3];
    memset(frame, 127, sizeof frame);
    if (mode && !strcmp(mode, "decode")) stall();
    fwrite(frame, 1, sizeof frame, stdout); fflush(stdout);
    if (mode && !strcmp(mode, "partial")) stall();
    fwrite(frame, 1, sizeof frame, stdout);
    return 0;
}
