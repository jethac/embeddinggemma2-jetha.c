# embeddinggemma2-jetha.c

EmbeddingGemma 2 server in C. Based on [embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c).

Inputs: text, image, audio, video, mixed. Output dimensions: 128, 256, 512, 768.
Maximum input: 8192 tokens.

Tested: Windows CPU, WSL CPU/CUDA, Linux ARM CPU, macOS CI CPU/Metal.
GPU operations can use CPU fallback. ROCm, XPU, GB10, and Strix Halo are unverified.
No NPU support. No binary releases.

## Build

Requires CMake >=3.24, a C/C++ compiler, Git, Python >=3.9, and NASM on x86.
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

On Windows, use MSYS2 MinGW64 with GCC, CMake, Ninja, Python, FFmpeg, and NASM.
Use `python`, add `-G Ninja`, and run the `.exe`.

| Backend | CMake options | Server option |
|---|---|---|
| CUDA | `-DGGML_CUDA=ON` | `--backend cuda` |
| Metal | `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON` | `--backend metal` |

CUDA requires the CUDA toolkit and a compatible NVIDIA driver. Metal requires macOS.
Use a separate build directory for each backend. For Metal, remove `-DGGML_METAL=OFF`.
Move the complete installation directory. x86-64 builds select the CPU instruction set at runtime.
For text input only, omit `--mmproj`.

## Use

```sh
python examples/embed.py --text "task: search result | query: what powers the cell"
python examples/embed.py --image picture.jpg --audio recording.wav --video clip.mp4 --dimensions 256
```

Use `--url http://HOST:PORT` to select the server.
API: `/api/embed` or `/v1/embeddings`. Outputs are normalized. A full queue returns HTTP 503.
[Request examples](examples/embed.py) · [All five input types](examples/journey2.py)

## Performance versus llama.cpp

llama.cpp: `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`.
Same Q8_0 weights, GGML kernels, threads, inputs, tokens, and output dimensions (768).
Caches off. Both engine orders after warmup. Cosine >=0.999. Ratio >1 means this server is faster.

[CPU](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/37892946034): EPYC 9V74, Ubuntu 24.04, two threads.
Other CPU load: 0.8–0.9%. Geometric mean: **1.003x**.
[Metal](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38020441663): virtual M1, Paravirtual GPU, three threads.
`EI_METAL_MEDIA_FLASH_ATTN2=1`. Vision attention used CPU fallback. Other CPU load: 1.46–97.42%.

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

The default Metal option is AUTO. Its [all-input mean ratio](perf/optimization_status.md#complete-virtual-metal-comparison)
was 0.801x (geometric). The option above was timed for images only.
CUDA media runs exceeded the limit for other CPU load.

[Measurements and options](perf/optimization_status.md#embeddinggemma-2-development-results) ·
[Text benchmark](perf/compare_llamacpp2.py) · [Media benchmark](perf/compare_media_llamacpp2.py)

## License

[MIT](LICENSE). Based on QuixiAI code and [Windows host code](https://github.com/jethac/embeddinggemma.c).
Weights have a separate license. Packages include dependency notices.
This software is based in part on the work of the Independent JPEG Group.

[Scope](GOAL.md) · [Release process](RELEASE.md) · [Upstream README](UPSTREAM_README.md)
