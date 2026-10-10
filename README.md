# embeddinggemma2-jetha.c

Experimental EmbeddingGemma 2 server in C. Based on
[QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c).
Inputs: text, image, audio, video, or mixed. Output: 128, 256, 512, or 768
dimensions. Maximum input: 8192 tokens.

| Backend | Tested hardware |
|---|---|
| Windows CPU | Xeon W-2135 |
| Linux CPU / CUDA | Xeon W-2135 / RTX 5060 Ti under WSL |
| macOS CPU / Metal | ARM CI / Apple Paravirtual GPU |

All five input types work on these backends. GPU execution can use CPU fallback.
ROCm, XPU, GB10, and Strix Halo are unverified. NPU support is absent.
No binary releases are available.

## Build

Requires CMake >=3.24, a C/C++ compiler, Git, Python >=3.9, and NASM on x86.
Video and WebP require `ffmpeg` and `ffprobe` on `PATH`.

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

On Windows, use MSYS2 MinGW64. Install these packages:

```sh
pacman -S --needed git mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake \
  mingw-w64-x86_64-ninja mingw-w64-x86_64-python \
  mingw-w64-x86_64-ffmpeg mingw-w64-x86_64-nasm
```

Use the commands above with `python` instead of `python3`. Add `-G Ninja` to
the CMake configuration. Run `local-install/bin/embeddinggemma2-jetha.exe`.

| Accelerator | CMake option | Server option | Requirements |
|---|---|---|---|
| CUDA | `-DGGML_CUDA=ON` | `--backend cuda` | CUDA toolkit for builds; compatible NVIDIA driver |
| Metal | `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON` | `--backend metal` | macOS with a Metal device |

Use separate build directories. Replace `-DGGML_METAL=OFF` for Metal.
Move the entire installation directory. x86-64 builds select scalar, AVX, AVX2,
or AVX-512 at runtime.

Omit `--mmproj` for text only. Use `--media-encoders vision` for image/video or
`--media-encoders audio` for audio.

## Use

```sh
python examples/embed.py --text "task: search result | query: what powers the cell"
python examples/embed.py --image picture.jpg --dimensions 256
python examples/embed.py --audio recording.wav
python examples/embed.py --video clip.mp4
python examples/embed.py --text "A description" --image picture.jpg --audio recording.wav
python examples/journey2.py
```

Use `--url http://HOST:PORT` for another server. `journey2.py` sends all five
input types; `--compare-url URL` compares outputs with another service.

POST to `/api/embed` or `/v1/embeddings`. Media accepts ordered `content` parts
or an array of input objects. Replace `BASE64_BYTES` with encoded file bytes:

```json
{"input":{"content":[{"type":"text","text":"A red square"},{"type":"image","data":"BASE64_BYTES"}]},"dimensions":256}
```

Part types: `text`, `image`, `audio`, `video`. Media accepts base64 or data URLs.
Video `fps` defaults to 1. Outputs are normalized after truncation.
`encoding_format`: `float` or `base64` float32 bytes.

Queue overflow returns HTTP 503.

Use `--persistent-cache-path cache.bin` to save results on graceful shutdown.
Keep `--response-cache-mb` above zero. Cache files contain request and media data.

## Performance versus llama.cpp

Baseline: `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`.
Matching Q8_0 weights, GGML kernels, threads, inputs, tokens, and 768 dimensions.
Caches off. Fresh servers, warmup, both engine orders, cosine >=0.999.
Ratio >1 means this server was faster.

[CPU text run](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/37892946034):
EPYC 9V74, Ubuntu 24.04, GCC 13.3, two threads. Other CPU load: 0.8–0.9%.
Minimum cosine: 0.999970. Geometric mean: **1.003x**.

| Tokens | Clients | Ours embeddings/s | llama.cpp embeddings/s | Ratio | Ratio by order |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 16.75 | 16.81 | 0.996x | 0.973–1.019x |
| 32 | 4 | 18.31 | 18.16 | 1.008x | 1.007–1.009x |
| 256 | 1 | 2.26 | 2.25 | 1.006x | 1.004–1.007x |
| 256 | 4 | 2.26 | 2.24 | 1.008x | 1.005–1.010x |
| 1024 | 1 | 0.44 | 0.44 | 1.008x | 1.003–1.014x |
| 1024 | 4 | 0.41 | 0.41 | 0.994x | 0.991–0.998x |

[Metal run](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38004147312):
Apple M1 (Virtual), Paravirtual GPU, three threads, 7 GB RAM.
Minimum cosine: 0.999301. Other CPU load: 0.58–149.32%.
Geometric mean: **0.801x**. These results do not measure a physical Apple GPU.
Encoder attention paths differ; see the [routing check](perf/optimization_status.md#metal-encoder-routing).

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

No CUDA media result is available from a host with low competing load.
See [other measurements and options](perf/optimization_status.md#embeddinggemma-2-development-results).
Reproduce with the [text harness](perf/compare_llamacpp2.py) or
[media harness](perf/compare_media_llamacpp2.py) and the pinned `llama-server`.

## License

[MIT](LICENSE). Retains QuixiAI attribution. Windows host code:
[jethac/embeddinggemma.c](https://github.com/jethac/embeddinggemma.c).
Weights have a separate license. Packages include dependency notices.
This software is based in part on the work of the Independent JPEG Group.

[Scope](GOAL.md) · [Release process](RELEASE.md) · [Upstream README](UPSTREAM_README.md)
