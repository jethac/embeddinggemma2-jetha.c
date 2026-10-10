# embeddinggemma2-jetha.c

EmbeddingGemma 2 server in C. Based on [embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c).

Inputs: text, image, audio, video, mixed. Dimensions: 128, 256, 512, 768.
Input limit: 8192 tokens.

Tested: Windows CPU, WSL CPU/CUDA, macOS CI CPU/Metal. GPU paths can use CPU fallback.
ROCm, XPU, GB10, and Strix Halo are unverified. No NPU support or binary releases.

## Build

Install CMake >=3.24, a C/C++ compiler, Git, Python >=3.9, and NASM on x86.
For video and WebP, add `ffmpeg` and `ffprobe` to `PATH`.

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

Windows: use MSYS2 MinGW64.

```sh
pacman -S --needed git mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake \
  mingw-w64-x86_64-ninja mingw-w64-x86_64-python \
  mingw-w64-x86_64-ffmpeg mingw-w64-x86_64-nasm
```

Use `python`, add `-G Ninja`, and run `local-install/bin/embeddinggemma2-jetha.exe`.

| Backend | CMake options | Server option | Requires |
|---|---|---|---|
| CUDA | `-DGGML_CUDA=ON` | `--backend cuda` | CUDA toolkit for builds; compatible NVIDIA driver |
| Metal | `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON` | `--backend metal` | macOS with a Metal device |

Use a separate build directory for each backend. Remove `-DGGML_METAL=OFF` for Metal.
Move the whole installation directory. x86-64 builds select scalar, AVX, AVX2, or AVX-512 at runtime.
For text only, omit `--mmproj`.

## Use

```sh
python examples/embed.py --text "task: search result | query: what powers the cell"
python examples/embed.py --image picture.jpg --dimensions 256
python examples/embed.py --audio recording.wav
python examples/embed.py --video clip.mp4
python examples/embed.py --text "A description" --image picture.jpg --audio recording.wav
```

Set `--url http://HOST:PORT` to change the server.
Run `python examples/journey2.py` to try all five input types.

API: `/api/embed` or `/v1/embeddings`. See [request examples](examples/embed.py).
Outputs are normalized. Queue overflow returns HTTP 503.

## Performance versus llama.cpp

llama.cpp: `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`.
Same Q8_0 weights, GGML kernels, threads, inputs, tokens, and 768 dimensions.
Caches off. Warmup and both engine orders. Cosine >=0.999. Ratio >1: faster here.

[CPU text run](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/37892946034):
EPYC 9V74, Ubuntu 24.04, two threads. Other CPU load: 0.8–0.9%. Geometric mean: **1.003x**.

| Tokens | Clients | This server, emb/s | llama.cpp, emb/s | Ratio | Range by order |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 16.75 | 16.81 | 0.996x | 0.973–1.019x |
| 32 | 4 | 18.31 | 18.16 | 1.008x | 1.007–1.009x |
| 256 | 1 | 2.26 | 2.25 | 1.006x | 1.004–1.007x |
| 256 | 4 | 2.26 | 2.24 | 1.008x | 1.005–1.010x |
| 1024 | 1 | 0.44 | 0.44 | 1.008x | 1.003–1.014x |
| 1024 | 4 | 0.41 | 0.41 | 0.994x | 0.991–0.998x |

[Metal image run](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38020441663):
Virtual M1, Paravirtual GPU, three threads. `EI_METAL_MEDIA_FLASH_ATTN2=1`.
Vision attention used CPU fallback. Other CPU load: 1.46–97.42%.

| Input | Clients | This server, emb/s | llama.cpp, emb/s | Ratio | Range by order |
|---|---:|---:|---:|---:|---:|
| Image | 1 | 0.1302 | 0.1097 | 1.187x | 1.072–1.317x |
| Image | 4 | 0.1915 | 0.1752 | 1.093x | 0.996–1.225x |

Default: AUTO. Its [earlier run](perf/optimization_status.md#complete-virtual-metal-comparison)
had a 0.801x geometric mean across all five inputs. Other inputs were not timed with this option.
No CUDA media comparison passed the limit for other CPU load.

[Measurements and options](perf/optimization_status.md#embeddinggemma-2-development-results) ·
[Text benchmark](perf/compare_llamacpp2.py) · [Media benchmark](perf/compare_media_llamacpp2.py)

## License

[MIT](LICENSE). QuixiAI attribution retained. Windows host code:
[jethac/embeddinggemma.c](https://github.com/jethac/embeddinggemma.c).
Weights have a separate license. Packages include dependency notices.
This software is based in part on the work of the Independent JPEG Group.

[Scope](GOAL.md) · [Release process](RELEASE.md) · [Upstream README](UPSTREAM_README.md)
