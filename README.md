# embeddinggemma2-jetha.c

EmbeddingGemma 2 server in C. Inputs: text, image, audio, video, mixed. Output: 128/256/512/768 dimensions. Maximum: 8192 tokens.

Tested: Windows CPU, WSL CPU/CUDA, Linux ARM CPU, macOS CI CPU/Metal. Some operations use CPU fallback.
Unverified: ROCm, XPU, GB10, Strix Halo. NPU support: none. Binary releases: none.

## Build

Requirements: CMake >=3.24, C/C++ compiler, Git, Python >=3.9, NASM (x86).
For video and WebP, put `ffmpeg` and `ffprobe` on `PATH`.

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

Windows: use MSYS2 MinGW64, `python`, `-G Ninja`, and `.exe`. For CPU audio, set `EI_CPU_AUDIO_F16_2=1` (+270 MiB of weights).
For text only, omit `--mmproj`. [CUDA, Metal, and build options](CONTRIBUTING.md).

## Use

```sh
python examples/embed.py --text "task: search result | query: what powers the cell"
python examples/embed.py --image picture.jpg --audio recording.wav --video clip.mp4 --dimensions 256
```

Server: `--url http://HOST:PORT`. API: `/api/embed` or `/v1/embeddings`. Vectors have unit length. [Options](examples/embed.py).

## Performance versus llama.cpp

Baseline: llama.cpp [de7fa0a](https://github.com/ggml-org/llama.cpp/commit/de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b). Same weights, kernels, threads, and inputs; Q8_0, 768 dimensions, caches off. Both orders after warmup. Cosine >=0.999. Ratio >1 means this server is faster.

- [CPU run](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/37892946034): EPYC 9V74, Ubuntu 24.04, two threads. Other CPU load: 0.8–0.9%.
- [Metal run](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38020441663): virtual M1, Paravirtual GPU, three threads. Vision attention uses CPU. Other CPU load: 1.46–97.42%.

| Backend | Input | Clients | This server, emb/s | llama.cpp, emb/s | Ratio |
|---|---|---:|---:|---:|---:|
| CPU | Text, 32 tokens | 1 | 16.75 | 16.81 | 0.996x |
| CPU | Text, 32 tokens | 4 | 18.31 | 18.16 | 1.008x |
| CPU | Text, 256 tokens | 1 | 2.26 | 2.25 | 1.006x |
| CPU | Text, 256 tokens | 4 | 2.26 | 2.24 | 1.008x |
| CPU | Text, 1024 tokens | 1 | 0.44 | 0.44 | 1.008x |
| CPU | Text, 1024 tokens | 4 | 0.41 | 0.41 | 0.994x |
| Metal | Image | 1 | 0.1302 | 0.1097 | 1.187x |
| Metal | Image | 4 | 0.1915 | 0.1752 | 1.093x |

Metal: `EI_METAL_MEDIA_FLASH_ATTN2=1`; four-client ratio by order: 0.996–1.225x. Default AUTO: [0.801x across all inputs](perf/optimization_status.md#complete-virtual-metal-comparison).
CUDA results: pending a host below the CPU load limit.

[Measurements](perf/optimization_status.md#embeddinggemma-2-development-results) · [Text benchmark](perf/compare_llamacpp2.py) · [Media benchmark](perf/compare_media_llamacpp2.py)

## License

[MIT](LICENSE). Based on [QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c) and [Windows host code](https://github.com/jethac/embeddinggemma.c). Model weights have a separate license.
This software is based in part on the work of the Independent JPEG Group.

[Scope](GOAL.md) · [Release process](RELEASE.md) · [Upstream README](UPSTREAM_README.md)
