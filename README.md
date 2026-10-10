# embeddinggemma2-jetha.c

Experimental EmbeddingGemma 2 server in C. Based on
[QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c).
This repository uses a separate name and release identity. See [GOAL.md](GOAL.md).

Text, image, audio, video, and mixed inputs are implemented. Outputs have 128,
256, 512, or 768 dimensions. The context limit is 8192 tokens.

| Target | Verified execution |
|---|---|
| Windows x86-64 CPU | All five input types on Xeon W-2135 |
| Linux x86-64 CPU | All five input types; runtime CPU dispatch |
| Linux CUDA | All five input types on RTX 5060 Ti under WSL |
| macOS ARM CPU | All five input types in CI |
| macOS Metal | All five input types on Apple Paravirtual GPU in CI |
| ROCm, Intel XPU | Unverified |
| GB10, Strix Halo GPU, Strix Halo NPU | Unverified; NPU support is absent |

GGML can use CPU fallback. Metal results do not prove physical Apple Silicon
performance or execution of all operations on the GPU. Complete reference
coverage, cross-request media batching, and release validation remain unfinished.
There are no binary releases. Build from source.

## Build and run

Install CMake 3.24 or newer, a C/C++ compiler, Git, and Python 3.9 or newer.
Install NASM on x86. Video and WebP require `ffmpeg` and `ffprobe` on `PATH`.
CMake downloads pinned dependencies. The model downloader saves separate Q8_0
weight files.

For Linux or macOS CPU:

```sh
git clone https://github.com/jethac/embeddinggemma2-jetha.c.git
cd embeddinggemma2-jetha.c
python3 scripts/download-model2.py
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_OPENMP=OFF
cmake --build build-cmake --target embeddinggemma2-jetha -j 6
cmake --install build-cmake --prefix local-install
local-install/bin/embeddinggemma2-jetha --bind 127.0.0.1 --port 42667 \
  --backend cpu --model model/embeddinggemma-2-Q8_0.gguf \
  --mmproj model/mmproj-embeddinggemma-2-Q8_0.gguf
```

For Windows, use an MSYS2 MinGW64 shell. Install the toolchain:

```sh
pacman -S --needed git mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake \
  mingw-w64-x86_64-ninja mingw-w64-x86_64-python \
  mingw-w64-x86_64-ffmpeg mingw-w64-x86_64-nasm
```

Use the commands above with `python` instead of `python3`. Add `-G Ninja` to
the CMake configuration. Run `local-install/bin/embeddinggemma2-jetha.exe`.
The installation includes the MinGW runtime DLLs.

| Accelerator | CMake option | Server option | Requirements |
|---|---|---|---|
| CUDA | `-DGGML_CUDA=ON` | `--backend cuda` | CUDA toolkit for builds; compatible NVIDIA driver |
| Metal | `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON` | `--backend metal` | macOS with a Metal device |

Use a separate build directory for each backend. Replace `-DGGML_METAL=OFF`
when building Metal. CUDA development uses CUDA 13.0. In WSL, keep the weights
on the Linux filesystem: use `--directory ~/embeddinggemma2-models` with the
downloader and pass those paths to the server.

Keep the entire installation directory when you move it. Shared libraries,
CPU plugins, and license notices are required. On x86-64, the default build
checks CPU and OS support before selecting scalar, AVX, AVX2, or AVX-512 kernels.
Do not disable `EI_CPU_DISPATCH` for portable builds.

Omit `--mmproj` for text only. Use `--media-encoders vision` for image/video or
`--media-encoders audio` for audio. The default loads both encoders.

## Send requests

With the server running:

```sh
python examples/embed.py --text "task: search result | query: what powers the cell"
python examples/embed.py --image picture.jpg --dimensions 256
python examples/embed.py --audio recording.wav
python examples/embed.py --video clip.mp4
python examples/embed.py --text "A description" --image picture.jpg --audio recording.wav
python examples/journey2.py
```

Add `--url http://HOST:PORT` to select another server. The journey command sends
synthetic inputs for all five input types. Add `--compare-url URL` to compare
with another running service.

Text routes are `/api/embed` and `/v1/embeddings`. Media uses an object with
ordered `content` parts, or an array of those objects. POST this JSON to either
route; replace `BASE64_BYTES` with encoded file bytes:

```json
{"input":{"content":[{"type":"text","text":"A red square"},{"type":"image","data":"BASE64_BYTES"}]},"dimensions":256}
```

Part types are `text`, `image`, `audio`, and `video`. Media accepts base64 bytes
or data URLs. Video accepts `fps`, which defaults to 1. The server normalizes
vectors after truncation. `encoding_format` accepts `float` or `base64` float32
bytes. Native responses contain `embeddings` and `usage`. OpenAI responses
contain `data` and `usage`.

| Limit | Value |
|---|---|
| Complete input | 8192 tokens |
| Image or video frame | 16 megapixels |
| Sampled video frames per input | 32 |
| Audio after 16 kHz resampling | 5,242,880 samples; 327.68 seconds |
| Retained decoded input | 128 MiB; workspaces and decoder copies are additional |
| Unique pending media requests | 64; 128 MiB of keys and bookkeeping |
| External probe / decoder deadline | 10 / 30 seconds; not a whole-request deadline |

Excess pending work returns HTTP 503. Identical media bodies on the same route
share one inference. Unique media requests are not batched across requests.
Short media arrays can use optional backbone batching. See
[optimization options and measurements](perf/optimization_status.md#embeddinggemma-2-development-results).

## Cache

Add `--persistent-cache-path cache.bin` to save exact results on graceful
shutdown. HTTP response persistence also requires a nonzero
`--response-cache-mb` value; its default is 64. Stop with SIGTERM or Ctrl-C on
Unix, or Ctrl-C/Ctrl-Break on Windows. Forced termination does not save data.

Cache files contain request bodies and media bytes. Reuse requires matching
weights, backend, encoder selection, batch limits, and numeric options. Changed
bodies and API routes have separate keys. Cache hits are not inference speedups.

## Performance versus llama.cpp

The baseline is pinned to `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`.
Comparisons use matching Q8_0 weights, GGML kernels, threads, inputs, token
counts, and 768 dimensions. Result, response, and prompt caches are off.
Each cell uses fresh servers, warmup, both engine orders, and cosine >=0.999.
A ratio above 1 means this server was faster.

The [CPU text run](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/37892946034)
used EPYC 9V74, Ubuntu 24.04, GCC 13.3, and two inference threads.
Other CPU load was 0.8-0.9%. The geometric mean was 1.003x: no measured CPU
speed advantage. Minimum cosine was 0.999970.

| Tokens | Clients | Ours embeddings/s | llama.cpp embeddings/s | Ratio | Ratio by order |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 16.75 | 16.81 | 0.996x | 0.973–1.019x |
| 32 | 4 | 18.31 | 18.16 | 1.008x | 1.007–1.009x |
| 256 | 1 | 2.26 | 2.25 | 1.006x | 1.004–1.007x |
| 256 | 4 | 2.26 | 2.24 | 1.008x | 1.005–1.010x |
| 1024 | 1 | 0.44 | 0.44 | 1.008x | 1.003–1.014x |
| 1024 | 4 | 0.41 | 0.41 | 0.994x | 0.991–0.998x |

The [Metal run](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38004147312)
used Apple M1 (Virtual), three cores, 7 GB RAM, and an Apple Paravirtual GPU.
Both engines used three threads. Minimum cosine was 0.999301. The geometric
mean was 0.801x: one win and nine losses. Other CPU load was 0.58-149.32%;
engine-order rates varied. This is a virtual GPU result.

| Input | Clients | Ours emb/s | llama.cpp emb/s | Ratio |
|---|---:|---:|---:|---:|
| Text | 1 | 2.4354 | 2.6003 | 0.937× |
| Text | 4 | 4.6006 | 6.6475 | 0.692× |
| Image | 1 | 0.0898 | 0.1175 | 0.764× |
| Image | 4 | 0.1129 | 0.1623 | 0.696× |
| Audio | 1 | 1.1562 | 1.0309 | 1.122× |
| Audio | 4 | 1.5689 | 2.0867 | 0.752× |
| Video | 1 | 0.1368 | 0.1565 | 0.874× |
| Video | 4 | 0.1264 | 0.1713 | 0.738× |
| Mixed | 1 | 0.0980 | 0.1180 | 0.831× |
| Mixed | 4 | 0.0959 | 0.1382 | 0.694× |

Earlier WSL text results were 1.01x CPU and 1.16x CUDA geometric means. Their
load guard missed Windows CPU contention, so those results are provisional.
The corrected guard has rejected later CUDA timing attempts before measurement.
No current quiet-host CUDA media throughput result is available.

CPU media runs had image parity and audio/video/mixed losses. One four-client
audio cell failed the 0.999 quality gate and has no published rate. Optional
VNNI audio results include wins on EPYC 9V45 and losses on Xeon 6973P-C.
Profiling was enabled in the EPYC run. Do not combine results from different CPUs.
[All completed cells, failures, and development latency measurements](perf/optimization_status.md#embeddinggemma-2-development-results)
include the provisional WSL results and optimization comparisons.

Reproduce CPU text or complete Metal media comparisons:

```sh
gh workflow run ci.yml --repo jethac/embeddinggemma2-jetha.c -f benchmark_cpu=true
gh workflow run ci.yml --repo jethac/embeddinggemma2-jetha.c -f benchmark_metal_media=true -f media_modalities=text,image,audio,video,mixed
```

For local runs, use [the text harness](perf/compare_llamacpp2.py) or
[the media harness](perf/compare_media_llamacpp2.py). Build `llama-server` from
the pinned dependency with matching GGML options. Under WSL, Windows
`python.exe` must be on `PATH` for host-load measurement. Leave phase profiling
off when measuring throughput.

## License and releases

[MIT](LICENSE). Preserve the QuixiAI copyright notice. Windows host changes
come from [jethac/embeddinggemma.c](https://github.com/jethac/embeddinggemma.c).
Model weights have their own license. Runtime packages include dependency notices.
This software is based in part on the work of the Independent JPEG Group.

[RELEASE.md](RELEASE.md) defines this repository's release matrix and staging.
`install.sh` and `install.ps1` target future releases; use source installation now.
The inherited Makefile and contributing guide describe the 300M implementation.
[UPSTREAM_README.md](UPSTREAM_README.md) retains its documentation and results.
Those results do not apply to EmbeddingGemma 2.
