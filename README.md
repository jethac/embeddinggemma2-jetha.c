# embeddinggemma2-jetha.c

EmbeddingGemma 2 server in C. Based on [embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c).

Inputs: text, image, audio, video, mixed. Dimensions: 128, 256, 512, 768.
Input limit: 8192 tokens.

Tested: Windows CPU, WSL CPU/CUDA, Linux ARM CPU, macOS CI CPU/Metal.
GPU operations can use the CPU. ROCm, XPU, GB10, and Strix Halo are unverified.
No NPU support. No binary releases.

## Build

Requirements: CMake >=3.24, C/C++ compiler, Git, Python >=3.9, NASM on x86.
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

Windows: use MSYS2 MinGW64. Install GCC, CMake, Ninja, Python, FFmpeg, and NASM.
Use `python` and `-G Ninja`. Run the `.exe`.

| Backend | CMake options | Server option |
|---|---|---|
| CUDA | `-DGGML_CUDA=ON` | `--backend cuda` |
| Metal | `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON` | `--backend metal` |

CUDA requires the CUDA toolkit and a compatible NVIDIA driver. Metal requires macOS.
Use a separate build directory for each backend. Remove `-DGGML_METAL=OFF` for Metal.
For text only, omit `--mmproj`. [Build options](CONTRIBUTING.md).

## Use

```sh
python examples/embed.py --text "task: search result | query: what powers the cell"
python examples/embed.py --image picture.jpg --audio recording.wav --video clip.mp4 --dimensions 256
```

Set the server with `--url http://HOST:PORT`.
API: `/api/embed` or `/v1/embeddings`. Output vectors have unit length.
[Requests](examples/embed.py) · [All input types](examples/journey2.py)

## Performance versus llama.cpp

llama.cpp: [de7fa0a](https://github.com/ggml-org/llama.cpp/commit/de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b).
Same Q8_0 weights, kernels, threads, inputs, tokens, and 768 dimensions.
Caches off. Warmup, then both engine orders. Cosine >=0.999.
Ratio >1 means this server is faster.

[CPU](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/37892946034): EPYC 9V74, Ubuntu 24.04, two threads.
Other CPU load: 0.8–0.9%. Mean ratio: **1.003x** (geometric).
[Metal](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38020441663): virtual M1, Paravirtual GPU, three threads.
`EI_METAL_MEDIA_FLASH_ATTN2=1`. Vision attention used the CPU. Other CPU load: 1.46–97.42%.

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

Default Metal AUTO: [0.801x across all inputs](perf/optimization_status.md#complete-virtual-metal-comparison) (geometric).
The Metal table shows image results with the option enabled.
CUDA media runs failed the CPU load limit.

[Measurements and options](perf/optimization_status.md#embeddinggemma-2-development-results) ·
[Text benchmark](perf/compare_llamacpp2.py) · [Media benchmark](perf/compare_media_llamacpp2.py)

## License

[MIT](LICENSE). Includes QuixiAI code and [Windows host code](https://github.com/jethac/embeddinggemma.c).
Weights use a separate license. Packages include dependency notices.
This software is based in part on the work of the Independent JPEG Group.

[Scope](GOAL.md) · [Release process](RELEASE.md) · [Upstream README](UPSTREAM_README.md)
