# embeddinggemma2-jetha.c

EmbeddingGemma 2 server in C. Fork of [embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c).

Inputs: text, image, audio, video, mixed. Dimensions: 128/256/512/768. Limit: 8192 tokens.

Tested: Windows CPU, WSL CPU/CUDA, Linux ARM CPU, macOS CI CPU/Metal. CPU fallback can occur. ROCm, XPU, GB10, and Strix Halo are unverified. No NPU support or binary releases.

## Build

Install CMake >=3.24, a C/C++ compiler, Git, Python >=3.9, and NASM on x86. For video and WebP, add `ffmpeg` and `ffprobe` to `PATH`.

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

- Windows: use [MSYS2 MinGW64](CONTRIBUTING.md), `python`, `-G Ninja`, and `.exe`.
- CUDA: install the toolkit and driver. Add `-DGGML_CUDA=ON`. Use `--backend cuda`.
- Metal: use macOS. Replace `-DGGML_METAL=OFF` with `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`. Use `--backend metal`.

Use separate build directories. For text only, omit `--mmproj`.

## Use

```sh
python examples/embed.py --text "task: search result | query: what powers the cell"
python examples/embed.py --image picture.jpg --audio recording.wav --video clip.mp4 --dimensions 256
```

Server: `--url http://HOST:PORT`. API: `/api/embed` or `/v1/embeddings`. Vectors have unit length. [Options](examples/embed.py).

## Performance versus llama.cpp

Baseline: llama.cpp [de7fa0a](https://github.com/ggml-org/llama.cpp/commit/de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b). Same Q8_0 weights, kernels, threads, inputs, and 768 dimensions. Caches off. Both orders after warmup. Cosine >=0.999. Ratio >1 means faster.

- [CPU run](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/37892946034): EPYC 9V74, Ubuntu 24.04, two threads. Other CPU load: 0.8–0.9%. Mean: 1.003x.
- [Metal run](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38020441663): virtual M1, Paravirtual GPU, three threads. Vision attention uses CPU. Other CPU load: 1.46–97.42%.

| Backend | Input | Clients | This server, emb/s | llama.cpp, emb/s | Ratio | Range by order |
|---|---|---:|---:|---:|---:|---:|
| CPU | Text, 32 tokens | 1 | 16.75 | 16.81 | 0.996x | 0.973–1.019x |
| CPU | Text, 32 tokens | 4 | 18.31 | 18.16 | 1.008x | 1.007–1.009x |
| CPU | Text, 256 tokens | 1 | 2.26 | 2.25 | 1.006x | 1.004–1.007x |
| CPU | Text, 256 tokens | 4 | 2.26 | 2.24 | 1.008x | 1.005–1.010x |
| CPU | Text, 1024 tokens | 1 | 0.44 | 0.44 | 1.008x | 1.003–1.014x |
| CPU | Text, 1024 tokens | 4 | 0.41 | 0.41 | 0.994x | 0.991–0.998x |
| Metal | Image | 1 | 0.1302 | 0.1097 | 1.187x | 1.072–1.317x |
| Metal | Image | 4 | 0.1915 | 0.1752 | 1.093x | 0.996–1.225x |

Metal rows use `EI_METAL_MEDIA_FLASH_ATTN2=1`. Default AUTO: [0.801x geometric mean across all inputs](perf/optimization_status.md#complete-virtual-metal-comparison).
CUDA: other CPU load exceeded the benchmark limit.

[Measurements](perf/optimization_status.md#embeddinggemma-2-development-results) · [Text benchmark](perf/compare_llamacpp2.py) · [Media benchmark](perf/compare_media_llamacpp2.py)

## License

[MIT](LICENSE). Includes QuixiAI code and [Windows host code](https://github.com/jethac/embeddinggemma.c). Model weights use a separate license.
This software is based in part on the work of the Independent JPEG Group.

[Scope](GOAL.md) · [Release process](RELEASE.md) · [Upstream README](UPSTREAM_README.md)
