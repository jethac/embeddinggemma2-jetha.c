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
unfinished. There are no binary releases or verified NPU performance claims.

## Performance versus llama.cpp

Development measurements on a shared Xeon W-2135 / RTX 5060 Ti 16 GB host,
Ubuntu under WSL, GCC 13.3 and CUDA 13.0 (2026-10-09). Both engines use the same
Q8_0 backbone, six CPU threads, the same compiled AVX-512/CUDA GGML kernels,
identical exact-token inputs, mean pooling and normalized 768-dimensional float
output. The llama.cpp baseline is pinned to
`de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`. Neither engine loads media encoders
for these text measurements. Exact-result, response and prompt caches are off.

Each cell starts fresh servers, checks concurrent embedding quality, warms both
engines and measures both orders for at least eight seconds per engine/order.
Before each recorded pass, the five-sample mean non-benchmark WSL CPU load was
below 150% (100% represents one logical core). **These results are provisional:**
that guard did not observe Windows host CPU contention. The harness now also
checks Windows CPU time under WSL; these cells need rerunning with that guard.
The host remains shared: the order
ranges below expose substantial variability in some cells and limit general
performance claims. Throughput is the mean of the two passes; the range spans
the two order-paired ratios. Every completed cell is retained, including losses.

| Backend | Tokens | Concurrent clients | Ours embeddings/s | llama.cpp embeddings/s | Throughput ratio | Ratio by order |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU AVX-512 | 32 | 1 | 16.00 | 16.37 | 0.98x | 0.94–1.01x |
| CPU AVX-512 | 32 | 4 | 17.53 | 17.88 | 0.98x | 0.96–1.00x |
| CPU AVX-512 | 256 | 1 | 2.11 | 2.18 | 0.97x | 0.95–0.99x |
| CPU AVX-512 | 256 | 4 | 2.14 | 2.13 | 1.01x | 0.94–1.08x |
| CPU AVX-512 | 1024 | 1 | 0.45 | 0.41 | 1.09x | 1.02–1.18x |
| CPU AVX-512 | 1024 | 4 | 0.46 | 0.45 | 1.03x | 1.03–1.03x |
| CUDA | 32 | 1 | 100.29 | 94.69 | 1.06x | 0.96–1.17x |
| CUDA | 32 | 4 | 70.06 | 89.01 | 0.79x | 0.77–0.80x |
| CUDA | 256 | 1 | 120.55 | 84.45 | 1.43x | 1.43–1.43x |
| CUDA | 256 | 4 | 87.24 | 83.11 | 1.05x | 1.03–1.07x |
| CUDA | 1024 | 1 | 43.82 | 28.35 | 1.55x | 1.51–1.59x |
| CUDA | 1024 | 4 | 32.61 | 25.53 | 1.28x | 1.07–1.48x |

CPU's geometric mean across its six cells is **1.01x**, with small losses on
short/medium single-client workloads. Minimum CPU cosine similarity is
**0.99997**. CUDA's geometric mean across its six cells is **1.16x**. Minimum CUDA
cosine similarity against llama.cpp was **0.99991**, above the 0.999 gate.
CUDA wins five cells by mean throughput, but short concurrent text loses in both
orders. Single-client medium/long text wins hold in both orders. Short single-client
and long concurrent rates still vary; larger concurrency and steadier hardware
measurements remain necessary. These figures cover
text; image, audio, video, mixed inputs and other hardware require separate
comparisons. Cache and singleflight savings are separate serving measurements.

For an additional Linux CI CPU measurement, run
`gh workflow run ci.yml --repo jethac/embeddinggemma2-jetha.c -f benchmark_cpu=true`.
The opt-in steps build pinned llama.cpp against the service's installed shared
GGML kernels, then run the same six text cells with two threads, caches disabled
and both engine orders. Hardware and measurements appear in the job log. CI
results describe that runner; they do not replace the local host's measurements.

Reproduce with [perf/compare_llamacpp2.py](perf/compare_llamacpp2.py). Build the
native server using the build instructions below and build `llama-server` from its pinned dependency
checkout (`build-cmake/_deps/llama-src`) in a separate build directory. Match
backend, compiler, CPU ISA and GGML options between builds. The measured builds
use `GGML_NATIVE=OFF`, `GGML_AVX512=ON`, `GGML_OPENMP=OFF`, `GGML_CUDA=ON`,
`CMAKE_CUDA_ARCHITECTURES=120a` and static libraries; the native build also uses
`EI_CPU_DISPATCH=OFF`. This static AVX-512 configuration requires a compatible
CPU. The llama.cpp build enables `LLAMA_BUILD_SERVER` and `LLAMA_BUILD_TOOLS`.

```sh
python3 perf/compare_llamacpp2.py \
  --model model/embeddinggemma-2-Q8_0.gguf \
  --embeddinggemma-bin build-cmake/bin/embeddinggemma2-jetha \
  --llama-server build-llama/bin/llama-server \
  --backend cuda --threads 6 --token-counts 32,256,1024 \
  --concurrency 1,4 --target-seconds 8
```

Use `--backend cpu` for the CPU comparison. The harness prints both server
commands, per-pass latency/throughput, host-load samples and cosine results.
CPU comparisons explicitly disable llama.cpp GPU devices, host-operation
offload and KV offload; zero GPU weight layers alone still permits host-operation
offload in this pinned version.
It uses ports 42674/42675 by default and rejects occupied ports.
Under WSL, Windows `python.exe` must be on PATH for host CPU sampling. An idle
guest cannot qualify a saturated Windows host. Set `EI_PROFILE_BACKBONE2=1` on
the native server to log graph construction, input preparation, synchronized
execution and output processing times; leave it unset for throughput measurements.

`EI_GRAPH_CACHE2=1` enables an experimental CUDA cache of two text graph shapes,
each with at most 256 aggregate batch tokens. Other requests use the normal
workspace. In a 120-request, 32-token/four-client trace, graph rebuilds fell from
64 to 2 and actual CUDA graph launches rose from 20 to 330. The two retained
32/96-token workspaces used 15.18 MiB combined. Changed inputs, unequal sequence
lengths, eviction, long text and all five modalities retained exactly matching
outputs. This flag defaults off: these are graph-reuse measurements, and a
quiet-host throughput comparison is still needed before claiming a speedup.

## Build and run the development server

CMake fetches a pinned MIT-licensed llama.cpp/GGML/libmtmd dependency and applies
the small media preprocessing fixes in `deps/`. Python 3.9 or newer is required
for dependency patching, model download and the sample client; inference runs
inside the native executable.

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
directory on `PATH` for compilation and running directly from the build tree. Video requires `ffmpeg`
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

Install the built service with `cmake --install build-cmake --prefix local-install`.
Run `local-install/bin/embeddinggemma2-jetha` (`.exe` on Windows) with the same
model and encoder arguments. Keep the whole install directory when moving it:
libraries, CPU plugins and license notices are installed alongside the service.
The MinGW install includes its runtime DLLs, so the installed executable does
not need the compiler's `bin` directory on PATH. Models remain separate; video
still requires external `ffmpeg` and `ffprobe`.

CUDA is exercised on an RTX 5060 Ti through Ubuntu under WSL with CUDA 13.0.
Build with `-DGGML_CUDA=ON` and start with `--backend cuda`; this requires the
CUDA toolkit at build time and a compatible NVIDIA driver at runtime. Both
media encoders use CUDA as well as the backbone. ROCm and XPU remain unverified.
The Linux CPU service build and image regression pass in CI.
The macOS ARM CPU build also runs all five modality journeys in CI. A Metal-enabled
service passed all five on the runner's Apple Paravirtual GPU, with minimum
cosine 0.99950580 against CPU. GGML can fall back to CPU for unsupported operations;
this does not establish physical Apple Silicon performance or an all-GPU path.
To try Metal on macOS, configure with `-DGGML_METAL=ON` and start with
`--backend metal`. Unsupported BF16 accelerator weights are widened exactly to
FP32 at load time. Broader hardware and reference quality remain unverified.

For the exercised macOS CPU build, configure with:

```sh
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_METAL=OFF -DGGML_OPENMP=OFF
```

Then use the build/run commands above, adding `--backend cpu` when starting the
server. With FFmpeg on `PATH`, `python examples/journey2.py` sends synthetic
text, image, audio, video and mixed requests and prints their latency and size.
Add `--url URL --compare-url OTHER_URL` to compare all five outputs against
another running service; the journey requires cosine above 0.999 for each.

On WSL, put weights on the Linux filesystem rather than a Windows-mounted
drive. For example, download with `--directory ~/embeddinggemma2-models` and
pass the files in that directory to `--model` and `--mmproj`. The local CUDA
service's measured fingerprinting time fell from 13.1 seconds to 2.4 seconds
after this move. Startup logs separate backbone loading, media encoder loading,
and cache fingerprinting so storage delays can be distinguished from inference.

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
Video probing uses stream duration when available, with container duration as a
fallback for formats such as WebM/Matroska. Buffered MP4 inputs with metadata at
the end are probed through a seekable cache wrapper.
Each external video probe has a 10-second deadline. Each video/WebP decoder has
a 30-second deadline starting at its first frame read and covering subsequent
reads. Expired subprocesses are terminated and reaped; a timeout rejects the
input even after complete frames were decoded. These are subprocess limits,
not a deadline for the whole request or model inference.
Media decoding and preprocessing use a separate lock, allowing text inference
to progress while an external probe or decoder waits. Media processing remains
serialized; encoder and backbone execution share the inference lock. In the
uncached stalled-probe regression, CUDA text latency fell from 10.60 s to
0.40 s, finishing before the probe's 10-second deadline. Windows CPU text
completed in 0.47 s under the same check. These are serving-stall measurements,
separate from the llama.cpp throughput comparison above.
Audio is capped at 5,242,880 mono samples after resampling to 16 kHz (327.68 seconds).
Decoded images, PCM and resident video frame buffers share a 128 MiB input budget;
decoder copies, preprocessing and inference workspaces require additional memory.
Complete inputs must fit 8192 tokens.
Images use the reference's 280-token patch budget; video frames use 140. Returned
vectors are normalized after truncation to 128, 256, 512, or 768 dimensions.
`encoding_format` accepts `float` or `base64` (float32 bytes).

Native media responses contain `embeddings` and token `usage`; the OpenAI-shaped
route returns an embedding `data` list and `usage`. Concurrent media requests with
identical HTTP bodies on the same route share one inference, even when caching
is disabled. Unique pending work is capped at 64 requests and 128 MiB of encoded
request keys plus entry bookkeeping; excess work returns HTTP 503. Decoded inputs,
responses and inference workspaces consume additional memory. Unique media
inferences still run serially; cross-request batching and resource handling need
further work.
Treat this as a development service while those limits are being completed.

To reuse exact media results after a restart, add `--persistent-cache-path cache.bin`
and keep `--response-cache-mb` nonzero (default 64). A bounded HTTP response snapshot
is saved to `cache.bin.responses` on graceful shutdown, alongside the text embedding
cache. Stop with SIGTERM/Ctrl-C on Unix or Ctrl-C/Ctrl-Break on Windows; forced
termination does not save the snapshot. The files include request bodies and media
bytes. Matching full model/mmproj contents, backend, encoder selection, and client
batch limit are required for reuse. Changed request bodies or API routes have
separate keys. Damaged or oversized records are ignored. Persistence reads the
complete weight files at startup; previous text-cache identities are invalidated
once by this change. With `--response-cache-mb 0`, HTTP response persistence is
disabled.

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
