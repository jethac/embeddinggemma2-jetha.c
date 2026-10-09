# embeddinggemma2-jetha.c

An experimental native EmbeddingGemma 2 port, intended to produce useful code
and evidence for the maintainers of [embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c).
The distinct name leaves the natural successor name available to upstream.

The scope covers text, images, audio, video, and mixed inputs, with CPU SIMD
(including AVX2 and AVX-512), accelerator execution, batching, and caching.
See [GOAL.md](GOAL.md) for the full scope, implementation direction,
performance requirements, acceptance criteria, and relationship to upstream.
The release process will follow upstream's conventions where applicable, with
this project's own names and an extended platform matrix.

**Current status:** the model-specific C backbone and native media encoders run
text, image, audio, video, and mixed requests on Windows CPU and Linux CUDA
development services. This is experimental: full reference parity, accelerator measurements,
broader CPU/platform coverage, media resource handling, and release integration remain
unfinished. There are no binary releases or verified GPU/NPU performance claims.

## Build and run the development server

CMake fetches a pinned MIT-licensed llama.cpp/GGML/libmtmd dependency and applies
the small media preprocessing fixes in `deps/`. Python is used for model
download and the sample client; inference runs inside the native executable.

```sh
python scripts/download-model2.py
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake --target embeddinggemma2-jetha -j 6
build-cmake/bin/embeddinggemma2-jetha --bind 127.0.0.1 --port 42667 \
  --model model/embeddinggemma-2-Q8_0.gguf \
  --mmproj model/mmproj-embeddinggemma-2-Q8_0.gguf
```

These commands require a C/C++ compiler and Git. The currently exercised build
uses MinGW64 on Windows; choose that toolchain explicitly and put its `bin`
directory on `PATH` for compilation and runtime DLLs. Video requires `ffmpeg`
and `ffprobe` on `PATH`. Omit `--mmproj` for text-only execution. With `--mmproj`,
use `--media-encoders vision` for image/video workloads or `--media-encoders audio`
for audio workloads to skip the unused encoder's weights. The default `all`
loads both. Requests for an unloaded modality are rejected before decoding.

On x86-64, the default build selects a CPU kernel plugin at runtime, checking
both CPU features and OS register-state support. It includes a baseline fallback,
AVX, AVX2, and AVX-512 variants. The Windows Xeon service selects Skylake-X;
all five tested modality outputs are bit-identical to the previous static build.
Keep the generated `bin` directory intact when running or copying this dev build:
its shared libraries and CPU plugins are required. Release packaging remains
unfinished. For a local static experiment, use `-DEI_CPU_DISPATCH=OFF` and
`-DGGML_AVX512=ON` on compatible hardware; that build has no runtime ISA fallback.
CUDA is exercised on an RTX 5060 Ti through Ubuntu under WSL with CUDA 13.0.
Build with `-DGGML_CUDA=ON` and start with `--backend cuda`; this requires the
CUDA toolkit at build time and a compatible NVIDIA driver at runtime. Both
media encoders use CUDA as well as the backbone. Other accelerator backends
remain unverified. The Linux CPU service build and image regression
pass in CI; macOS builds remain unverified for this implementation.

```sh
python examples/embed.py --text "task: search result | query: what powers the cell"
python examples/embed.py --image picture.jpg --dimensions 256
python examples/embed.py --audio recording.wav
python examples/embed.py --video clip.mp4
python examples/embed.py --text "A description of this scene" --image picture.jpg --audio recording.wav
```

## API

Text uses the inherited `/api/embed` and `/v1/embeddings` contracts. Multimodal
inputs use an object containing ordered `content` parts, or an array of these
objects. Media bytes are base64 encoded; base64 data URLs are also accepted.

```json
{"input":{"content":[{"type":"text","text":"A red square"},{"type":"image","data":"BASE64_BYTES"}]},"dimensions":256}
```

Supported part types are `text`, `image`, `audio`, and `video`. A video part can
set `fps` (default 1); the current implementation accepts at most 32 sampled
video frames across an input. Images and video frames are limited to 16 megapixels.
Audio is capped at 5,242,880 mono samples after resampling to 16 kHz (327.68 seconds).
Decoded images, PCM and resident video frame buffers share a 128 MiB input budget;
decoder copies, preprocessing and inference workspaces require additional memory.
Complete inputs must fit 8192 tokens.
Images use the reference's 280-token patch budget; video frames use 140. Returned
vectors are normalized after truncation to 128, 256, 512, or 768 dimensions.
`encoding_format` accepts `float` or `base64` (float32 bytes).

Native media responses contain `embeddings` and token `usage`; the OpenAI-shaped
route returns an embedding `data` list and `usage`. Media processing currently
runs serially and needs further queue, batching, and decoded-allocation work.
Treat this as a development service while those limits are being completed.

## Foundation and attribution

Based on [QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c),
with Windows host-layer changes from
[jethac/embeddinggemma.c](https://github.com/jethac/embeddinggemma.c).
The project remains MIT licensed, including new work; see [LICENSE](LICENSE).
The original QuixiAI copyright notice is preserved.

[UPSTREAM_README.md](UPSTREAM_README.md) preserves the original documentation
and benchmark claims for the original 300M model. They do not establish
EmbeddingGemma 2 results for this project.

The inherited Makefile, release scripts, and [CONTRIBUTING.md](CONTRIBUTING.md)
still describe the 300M implementation. Use the CMake commands above for this
port; adaptation of those workflows is unfinished. CI has a new Linux service
job; the retained platform matrix still exercises the legacy model. Model weights, local
reference checkouts, and generated artifacts are excluded from Git.
