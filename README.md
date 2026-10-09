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

A completed [CI comparison](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/37892946034)
on 2026-10-09 used Ubuntu 24.04, GCC 13.3, an AMD EPYC 9V74 runner with four
logical CPUs, two inference threads, and the same installed shared GGML library
and runtime CPU plugins for both engines. All six cells used the same Q8 weights,
uncached requests and both orders described above; observed non-benchmark CPU
load was 0.8–0.9%. These results describe that runner separately from the WSL host.

| Tokens | Clients | Ours embeddings/s | llama.cpp embeddings/s | Ratio | Ratio by order |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 16.75 | 16.81 | 0.996x | 0.973–1.019x |
| 32 | 4 | 18.31 | 18.16 | 1.008x | 1.007–1.009x |
| 256 | 1 | 2.26 | 2.25 | 1.006x | 1.004–1.007x |
| 256 | 4 | 2.26 | 2.24 | 1.008x | 1.005–1.010x |
| 1024 | 1 | 0.44 | 0.44 | 1.008x | 1.003–1.014x |
| 1024 | 4 | 0.41 | 0.41 | 0.994x | 0.991–0.998x |

The geometric mean is **1.003x**, effectively parity, with minimum cosine
**0.999970**. The small losses remain visible; this is not an established CPU
speed advantage.

A separate [packed-QKV CI comparison](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/37899814743)
used `EI_QKV2=1` on an AMD EPYC 7763 runner (four logical CPUs, two inference
threads), with the same Ubuntu/GCC, installed shared kernels, Q8 weights, cache
policy and both engine orders. Observed non-benchmark CPU load was 1.6–2.3%.

| Tokens | Clients | Packed ours embeddings/s | llama.cpp embeddings/s | Ratio | Ratio by order |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 11.73 | 11.82 | 0.993x | 0.949–1.036x |
| 32 | 4 | 12.30 | 12.20 | 1.007x | 1.004–1.011x |
| 256 | 1 | 1.48 | 1.48 | 0.999x | 0.989–1.008x |
| 256 | 4 | 1.47 | 1.46 | 1.006x | 1.006–1.006x |
| 1024 | 1 | 0.30 | 0.28 | 1.040x | 1.039–1.041x |
| 1024 | 4 | 0.28 | 0.28 | 1.029x | 1.026–1.031x |

Its geometric mean was **1.012x**, with minimum cosine **0.999927**. Different
runner CPUs prevent attributing the difference from the earlier run to packing;
this compares packed inference with llama.cpp on the EPYC 7763 only. Packed QKV
remains opt-in.

Reproduce the CI comparison with
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

`EI_QKV2=1` enables experimental packed Q/K/V projections for Q8_0 weights on
CPU and CUDA. It uses 27.62 MiB of additional weights. In a fixed 106-request,
32-token CUDA capture, quantized matmul launches fell from 22,680 to 17,640,
combined matmul/fixup time fell from 155.5 to 130.1 ms, and total captured GPU
kernel time fell 7.2%. These are profiling results, not end-to-end throughput.
Ten Windows CPU checks, including all five modalities, remained bit-identical;
the matching CUDA checks passed the 0.999 cosine gate (minimum 0.999830).
Persistent caches distinguish packed and separate projections because CUDA
accumulation order can change their outputs. The flag defaults off; quiet-host
throughput and broader hardware validation remain necessary. To measure it on
the CI runner, add `-f packed_qkv=true` to the CPU comparison command above.

`EI_CUDA_GLOBAL_ATTN2=1` enables an experimental CUDA fallback for global
attention when GGML's flash kernel cannot handle the input shape. It uses
matrix multiplication and masked softmax for at most 2048 aggregate tokens,
bounding the four-head score matrix to 64 MiB. Larger inputs and shapes with
native CUDA flash support use streaming attention: unsupported larger shapes
pad K/V to the next 256-key boundary, mask every added key, and use native
CUDA flash when supported. Other devices retain the existing fallback. For a 32-token request,
actual scheduler placement changed from nine alternating GPU/CPU splits to
one GPU split. With text and HTTP caches disabled, graph caching enabled and
packed QKV disabled, paired warm HTTP medians on the contended RTX 5060 Ti host
fell from 11.6 to 5.0 ms and from 12.9 to 5.0 ms in the reverse order (100
measured requests per pass). These are latency observations under contention,
not a quiet throughput comparison against llama.cpp. For an unaligned
2049-token request, padded flash also produced one CUDA split, and paired warm
HTTP medians fell from 2094 to 152 ms and 2515 to 338 ms in the reverse order
(20 measured requests per pass, packed QKV and result caches off). Padding
regressed for 32-token inputs in both orders, so short requests retain explicit
attention. The 8191-token request passed cosine 0.999938, and an unequal
2049-token API batch passed 0.999917 with isolated outputs. Its two inputs
crossed the packing cutoff and ran as separate engine forwards.
All five modality samples
passed cosine 0.999 (minimum 0.999848); combined with packed QKV, the minimum
was 0.999824. Unequal batches and lengths around native flash alignment and
the fallback limit also passed. Persistent caches distinguish this numeric
mode and implementation version. The flag defaults off pending broader
hardware and quiet-host validation.

`EI_GEGLU2=1` fuses the backbone's FFN and per-layer-input GELU/multiply pairs.
On the RTX 5060 Ti, paired warm 8191-token HTTP medians improved from 195.4 to
182.2 ms and from 200.1 to 188.4 ms in reverse order (20 measured requests after
six warmups, six threads, QKV off, global/local attention, lower mask range,
input reuse and graph caching on, result caches/profiling off). These are
native latency measurements, not a matched llama.cpp throughput claim.
All five modality samples, 8191/8192-token text, changed 2049-token batches,
five-image and long mixed inputs remained bit-identical on CUDA. In complete
12-forward GPU traces, kernel count fell from 12,804 to 12,228 and total kernel
time from 2209 to 2130 ms. The flag defaults off and applies only to CUDA.

`EI_MEDIA_BATCH2=1` combines short inputs in a multimodal request array into
bounded backbone groups: at most 1024 aggregate tokens and 512 per input. Longer
inputs retain their original single-input forward and full 8192-token context.
Encoding remains serialized, with one decoded input at a time and at most 2 MiB
of pending raw rows. The flag defaults off and has its own persistent-cache domain.
On the shared RTX 5060 Ti host, eight one-second audio inputs (264 total tokens,
six CPU threads, other CUDA flags as above, response/result caches and profiling
off) improved from 141.5 to 125.0 ms and 139.1 to 120.9 ms in reverse order.
Windows CPU improved from 1426 to 1319 ms and 1376 to 1289 ms on that sample;
a single mixed-array CPU pass regressed slightly (15.36 to 15.70 seconds).
Mixed CUDA timings varied strongly under contention, so there is no general
throughput claim. These are comparisons with our sequential implementation,
not llama.cpp. Larger 4096-token groups were rejected after regressions.
Minimum batch-versus-single cosine was 0.999793 across the exercised CPU/CUDA
samples. Changed sequence boundaries, group splits, OpenAI ordering and persistent
cache separation passed; other accelerators remain unverified. Dev CUDA enables
this flag; the Windows primary keeps it off. Cross-request raw batching remains
unfinished.

Dependency debug logs are suppressed by default; normal diagnostics, warnings
and startup timings remain visible. Set `EI_DEBUG_LOG2=1` to include dependency
debug output. Large tensor inventories can delay startup when logs are written
through a slow filesystem mount.

`EI_REUSE_INPUTS2=1` keeps positions, attention masks and pooling weights in
dedicated backend buffers while a graph and its sequence boundaries remain
unchanged. Token IDs and media rows are uploaded on every forward; this does
not cache embeddings. Masks contain exact FP16 zero/negative infinity values.
The 8191-token CUDA graph uses 256.05 MiB of dedicated auxiliary storage.
On the contended RTX 5060 Ti, paired warm 8191-token HTTP medians fell from
2106 to 392 ms and from 1207 to 413 ms in the reverse order (20 measured
requests per pass, six CPU threads, profiling and result caches disabled,
global-attention fallback and graph caching enabled, packed QKV disabled).
These are latency observations, not a quiet llama.cpp throughput comparison.
Changed text, changed boundaries in one 2049-token/three-sequence engine batch,
graph rebuilds and short-shape eviction produced bit-identical outputs on
Windows CPU and CUDA. All five modality samples also matched exactly on both.
The flag defaults off; other accelerator backends remain unverified.

`EI_CUDA_LOCAL_ATTN2=1` pads local-attention keys to a 256-key stride for
unaligned inputs of at least 1024 aggregate tokens. This enables GGML's grouped
query and mask-scanning paths; every added key is masked. At 8191 tokens,
paired warm HTTP medians fell from 514 to 402 ms and from 362 to 282 ms in the
reverse order (20 measured requests per pass, six threads, QKV off, global
attention and input reuse on, result caches/profiling off, contended RTX 5060 Ti).
In complete 12-forward GPU captures, local flash-kernel time fell from 2195 to
1145 ms, and total kernel time fell 23.5%. These are profiling and latency
results, not a quiet throughput comparison against llama.cpp.
Actual 2049-token/three-sequence batches, changed boundaries, five-image and
long mixed inputs passed cosine 0.999 (minimum 0.999878); aligned 8192-token
inputs and all five short modality samples stayed bit-identical. Persistent
caches distinguish this numeric mode. It defaults off and applies only to CUDA.

`EI_CUDA_LOCAL_RANGE2=1` also skips fully masked leading key tiles in local
attention from 1024 aggregate tokens, including the key padding above. With
local padding already enabled, warm 8191-token HTTP medians fell from 259 to
196 ms and from 264 to 195 ms in the reverse order (20 measured requests after
six warmups per pass, six threads, QKV off, global attention/input reuse/graph
caching on, result caches/profiling off, RTX 5060 Ti). This is about 25% lower
latency than the padded local-attention path on this host; it is not a matched
llama.cpp throughput claim. In complete 12-forward GPU captures, local flash
time fell from 1145 to 200 ms and total kernel time fell 30.3%, including the
additional mask-scan cost. All five modality samples, 8191/8192-token text,
actual 2049-token/three-sequence batches with changed boundaries, five-image
inputs and long mixed inputs stayed bit-identical. The flag defaults off,
applies only to CUDA and has a separate persistent-cache identity.

## Text responsiveness during media encoding

The media encoder and backbone now use separate locks during encoding, so short
text can finish while an image/audio encoder is busy. On the shared Windows
Xeon W-2135 host with six CPU threads, a fresh short text submitted 500 ms into
a four-image request waited 26.7 seconds before the change. The installed final
service took 0.87 seconds for text while that media request took 23.5 seconds;
concurrent and standalone embeddings were identical. This is a responsiveness
observation on a busy host, not a quiet throughput comparison against llama.cpp.
The media queue still admits only one decoded request at a time, and backbone
execution remains serialized. Reproduce on a CPU service with caches disabled:

```sh
python tests/media_text_progress2.py --url http://127.0.0.1:42667
```

## Build and run the development server

CMake fetches pinned llama.cpp/GGML/libmtmd and simdjson dependencies, retaining
their MIT notices, and applies the small media preprocessing fixes in `deps/`.
Python 3.9 or newer is required
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

Reuse HTTP connections for repeated requests. On this Windows/WSL NAT host,
opening a connection cost about 23 ms. Large requests through WSL's localhost
relay also incurred a roughly 40 ms delayed-ACK stall on reused connections.
The Linux server now requests immediate acknowledgements while reading an
incomplete body. On the deployed CUDA endpoint, one-second audio requests on
reused connections measured 56.6 -> 19.4 ms and 55.7 -> 14.4 ms in reverse
order. Eight-client waves measured 104 -> 90 ms and 103 -> 87 ms; another pass
during builds was effectively flat. These are transport observations on a
shared host, not a kernel or general throughput claim. Each request had a unique
cache key; the primary retained its response cache, while the original comparison
service had caching disabled. All five modality outputs remained identical.

The focused regression uses invalid 43 KB and 2 MiB requests to exclude model
time. Run it from Windows against the Linux endpoint:
`python tests/http_body_progress2.py --url http://127.0.0.1:42669`.
Native Linux and Windows checks cover reusable connections and error framing;
reproducing the relay stall requires the Windows-to-WSL path.

Media base64 decoding now uses a byte lookup table instead of searching the
alphabet for every character. On the same shared host, an uncached 1800×1800
PPM image (12,960,131-byte JSON request) measured 780 -> 611 ms and 645 -> 542 ms
in reverse order, about 16–22% lower HTTP latency. Native Linux clients used
fresh connections, matching 64 MiB response caches and unique keys; JSON body
construction was outside the timer. These paired observations cover that upload,
not general inference throughput. All five modalities remained byte-identical
to the accepted service, including malformed-base64 rejection behavior.
The deployed CUDA primary completed the same fresh upload at a 454 ms median
in a later pass; that separate pass does not establish an additional speedup.
The installed Windows CPU primary also includes this decoder update. Its first
shared-host large-upload pass was effectively flat (33.785 -> 33.899 seconds),
with encoder and backbone work dominating; no Windows inference speedup is
established. All five Windows modality vectors remained identical after updating.

A subsequent decoder change borrows the parsed base64 string and decodes full
quartets directly, retaining padding checks on the final quartet. The same
uncached upload measured 595 -> 548 ms and 436 -> 416 ms in reverse order,
about 5–8% lower HTTP latency, with identical vectors and matching cache settings.
The CUDA primary completed a later pass at 364 ms; this separate pass is not an
additional speedup comparison. Both development primaries include the change;
these observations establish no Windows CPU inference speedup.

Multimodal JSON parsing uses pinned simdjson 5.0.3 with runtime CPU dispatch
and scalar fallback. Parsing and converting a 13 MB request measured 6–7 ms
versus 101 ms with the previous parser. Actual uncached uploads measured
397 -> 305 ms and 405 -> 299 ms in reverse order (23–26% lower HTTP latency),
with identical embeddings and the same cache/client policy above. Shared-host
latency varies; this is not a general throughput or Windows inference claim.
The original parser remains a fallback for parser limits and unusual numbers,
and malformed JSON remains rejected. simdjson's MIT notice ships with runtime
packages; the project's license remains MIT.

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

`install.sh` and Windows `install.ps1` target this repository's future releases.
They verify both the raw executable and its matching runtime archive against
`SHA256SUMS`, then install the libraries, CPU plugins and license notices together.
No binary releases exist yet; use the CMake source installation above.
Unix exposes `embeddinggemma2-jetha` through a relative symlink; Windows exposes
`embeddinggemma2-jetha.cmd` (default directory:
`$env:LOCALAPPDATA/Programs/embeddinggemma2-jetha`). Updates preserve the previous
application directory and validate the new executable before switching the launcher.
Model weights remain separate; video/WebP need FFmpeg on PATH.
See [RELEASE.md](RELEASE.md) for native staging and the complete release matrix.

The inherited Makefile, `scripts/stage-release.sh`, and [CONTRIBUTING.md](CONTRIBUTING.md)
still describe the 300M implementation. Use the CMake commands above for this
port; adaptation of those workflows is unfinished. Separate Gemma 2 CI jobs
build and exercise Linux, macOS and Windows services; the retained legacy matrix
still exercises the original model. Model weights, local
reference checkouts, and generated artifacts are excluded from Git.
