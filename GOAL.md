# embeddinggemma2-jetha.c

Build a fast, native, model-specialized EmbeddingGemma 2 inference server,
starting from QuixiAI's MIT-licensed embeddinggemma.c and the native Windows
host-layer contribution. Public home: `jethac/embeddinggemma2-jetha.c`.
The distinct name avoids taking the natural upstream successor name.

## License

The project remains MIT licensed, including new implementation work. Preserve
the existing `LICENSE` and QuixiAI copyright notice. Retain required attribution
and notices for dependencies; model weights remain governed by their own license.

## How to spend effort

These rules govern development and supersede process-heavy interpretations of
the inherited guides.

1. Done means a person can complete the journey on the deployed dev service.
   Deploy and try the journey early, then after each meaningful change. Passing
   tests, reports, and proofs do not substitute for a working journey.
2. Measure before fixing. For slowness or flakiness, add timing or logging,
   identify the biggest cost, fix it, and measure again. Run the system directly
   when that answers the question; do not build machinery around the question.
3. Verification must change a decision. Before adding a check, test, or report,
   state what failure would make you do differently. If nothing, skip it.
4. Tests protect safety invariants and bugs that actually happened. Add one
   regression test per real bug and show it failing before the fix. Do not build
   test matrices for unused code.
5. No process artifacts for things that never shipped: evidence bundles,
   qualification reports, input hash manifests, provenance layers, handoff
   documents, or recovery tooling. Delete or bypass unfinished process machinery
   that blocks a journey, while preserving safety invariants. Release checksums
   and required license notices apply to actual distribution.
6. Inspect what the system actually does: the real model input and prompt,
   request on the wire, and log output. Documentation is not runtime evidence.
7. Report a failure in one plain line and keep moving. Avoid long explanations
   of why work is unfinished.
8. Check in after each implementation round with the user-visible improvement
   on the deployed service and measured numbers. If nothing a user would notice
   improved, change course. Do not present documentation or test counts as
   service improvements.

## Relationship to upstream

This project is an experimental port intended to get ahead of upstream's
EmbeddingGemma 2 work and produce useful code and evidence they can adopt.
It should make that work easier, not create a competing claim to the upstream
project name or impose maintenance work on its maintainers.

- Preserve history, licenses, attribution, and a clear upstream remote.
- Keep architecture, host portability, API, and optimization changes separable
  and reviewable so upstream can cherry-pick what helps.
- Leave concise code comments and reproduction commands when they help upstream
  understand or adopt a change. Do not build separate evidence or handoff systems.
- Keep this project's experimental branding, release ownership, and version
  history distinct. Reuse upstream's release process wherever it fits.
- Do not open upstream issues, submit pull requests, or contact maintainers
  without the user's instruction. Local implementation can proceed independently
  of the inherited contributing guide's upstream issue-first policy.

## Scope

Everything is in scope: text (including code), images, audio, video, and mixed
modality inputs in the model's shared embedding space. A text-only port is not
completion of this goal. Intermediate implementation steps do not narrow the
scope or become a substitute for the finished product.

- Implement the actual EmbeddingGemma 2 architecture, including projection-only
  per-layer inputs, local/global attention dimensions, Q/K/V normalization,
  layer output scales, and the final 512-to-768 projection.
- Implement vision and audio encoder integration, the required media
  preprocessing, video frame sampling and ordering, modality boundary tokens,
  and mixed input assembly. Pool and normalize according to the reference.
- Support selectively loading the encoders needed for a workload.
- Return normalized 768-, 512-, 256-, or 128-dimensional embeddings. Preserve
  the upstream native and OpenAI-compatible text API contracts, and specify
  and document an explicit multimodal API.
- Preserve native Windows support and support Linux and macOS. Keep model
  weights separate from source and executable artifacts.
- Cover CPU and applicable accelerator backends: CUDA, Metal, ROCm, and Intel
  XPU/SYCL. Do not silently advertise an unimplemented or unverified backend.

## Performance is a requirement

Make the entire serving path as performant as practical while preserving
embedding quality. Optimize from measured bottlenecks, not instruction-set
names or theoretical throughput alone.

- CPU optimization explicitly includes scalar fallback, AVX, AVX2/FMA/F16C,
  AVX-512, and applicable AVX-512 extensions; ARM SIMD applies to ARM targets.
  Detect CPU and OS support at runtime for portable distributions. Never
  require an unsupported instruction set merely to start the executable.
- Benchmark AVX2 against AVX-512 on matching hardware, accounting for frequency
  changes, bandwidth limits, core count, thread count, and sequence length.
  The local Xeon W-2135 provides hardware for AVX-512 validation; newer
  extensions such as VNNI/BF16 need hardware that actually supports them.
- Publish measured performance comparisons against pinned llama.cpp in the
  README, using the same weights, hardware, thread count, input tokens, output
  dimensions and cache policy. Include losses and optimize from them. Measure
  both engine orders; distinguish uncached inference from caching/singleflight
  gains, and identify contention or missing modality/backend coverage.
- Use efficient quantized kernels, persistent workspaces, fused operations
  where beneficial, and efficient matrix/attention execution on accelerators.
- Use the user's tailnet GB10 ThinkStation PGX and Strix Halo NUC as native
  optimization targets once access is available. Measure CUDA on GB10 and
  CPU versus applicable GPU backends on Strix Halo with matching requests.
- Investigate Strix Halo XDNA 2 NPU execution for the complete model, including
  its vision and audio encoders. Start with a BF16 ONNX compiler probe, inspect
  actual NPU/CPU operator placement, and compare embeddings and end-to-end
  latency against the working implementation before adding quantization.
  Test the backbone and encoders independently to locate compiler limitations;
  partial acceleration is an intermediate experiment, not completion of the
  full multimodal scope. Keep successful sessions resident, and consider
  bounded sequence-length buckets if required by the compiler. Measure energy
  per request where available as well as latency and throughput. Label mixed
  CPU/NPU execution honestly; successful runtime fallback is not proof that
  the model runs entirely on the NPU. NPU support is currently unimplemented.
  The 740M parameters imply about 1.48 GB of BF16 weights before activations and
  runtime overhead; do not confuse system-memory fit with on-chip SRAM residency.
  Ryzen AI 1.8 lists Strix Halo, BF16 NLP, and the original EmbeddingGemma 300M,
  but not EmbeddingGemma 2. Use its ONNX/Vitis AI path as the initial experiment,
  inspecting placement because unsupported subgraphs automatically run on CPU.
  See AMD's [release notes](https://ryzenai.docs.amd.com/en/latest/relnotes.html)
  and [deployment documentation](https://ryzenai.docs.amd.com/en/latest/modelrun.html).
- Use FluidInference's [EmbeddingGemma 2 Core ML conversion](https://huggingface.co/FluidInference/embeddinggemma-2-coreml)
  as an accelerator reference. It ships text, audio and vision assets: text uses
  fixed 32–512-token ANE functions with CPU token-table lookup; audio and vision
  use the GPU in the recommended serving path. Its card reports audio attention
  falling back to CPU under ANE, and vision running on ANE but slower than GPU.
  Fixed buckets, packed independent sequences and rescaled FP16 RMSNorm are
  useful compiler/performance experiments for our NPU work. Preserve the full
  8192-token context and multimodal scope here; its 512-token limit is not our
  completion target. Reproduce placement, quality and timings on target hardware
  before adopting a path. Published assets and FluidUse are Apache-2.0; retain
  their required notices if reused, while our own implementation remains MIT.
- Preserve and extend dynamic batching, bounded queues, duplicate singleflight,
  exact-result caching, and useful concurrency behavior to multimodal requests.
- Measure the real request end to end. Add timings for tokenization, media
  preprocessing, encoders, the text backbone, batching, or serialization when
  they can identify the bottleneck and change the next optimization decision.

## Implementation direction

The current direction is a model-specific C graph using a pinned GGML kernel
dependency and Gemma 4 media encoder support through libmtmd. This is an
implementation choice to validate, not a completed port or a proven speedup.
Retain upstream attribution and pin dependencies with required license notices.
Prefer specialization and reuse of validated optimized kernels over a Python
inference subprocess. Revise the approach when measurement justifies it.

## Release process

Follow upstream's [RELEASE.md](RELEASE.md) conventions where applicable, adapting
the repository, project/asset names, model, dependencies, and platform matrix.
Do not invent a separate distribution workflow without a concrete need.

- Build the complete supported release matrix from the same clean commit.
- Build and run the real journey on native platforms and matching accelerator
  hardware for the targets being released, using the final staged executables.
- Publish raw executables and matching runtime archives with stable asset names
  and `SHA256SUMS`. Keep model
  weights and intermediate files out of releases. Preserve checksum-verified
  installation and appropriate backend detection/fallback.
- Strip binaries, sign and verify Darwin artifacts, and retain portable OS/ABI
  baselines. CPU release binaries must dispatch safely across supported ISAs;
  host-specific build flags are for local experiments.
- Carry over useful accelerator code-object/runtime checks and include native
  Windows in the release matrix. Include required dependency license notices.
- Use this repository's own versions, tags, release destination, asset prefix,
  and cache/install identity. Adapt inherited release scripts before using them
  for publication; they currently target the original project.
- Obtain approval for an exact release version before changing release versions,
  tagging, or publishing, as required by the inherited release runbook. Creating
  this public development repository does not publish a binary release.

## Validation and acceptance

- Try real text, image, audio, video, and mixed-input journeys on the deployed
  dev service. Exercise model acquisition, startup, requests, and useful results,
  not merely an internal inference call. Repeat after meaningful changes.
- Compare real outputs against a pinned reference to decide whether an inference
  path is usable or needs correction. Inspect intermediate model inputs and
  values only where a discrepancy or optimization question calls for it.
- Protect safety invariants: CPU/OS instruction support, batch isolation, cache
  identity, finite normalized outputs, context limits, bounded queues and media
  allocations, and graceful malformed-input handling. Keep checks focused.
- Add regression tests for observed bugs, failing before the fix. Avoid speculative
  combinations and unused-backend matrices. State untested targets plainly.
- Compare performance with the same model/quantization, exact inputs and token
  counts, dimensions, cache settings, warmup, concurrency, and hardware when the
  comparison can change implementation choices. Report measured latency and
  throughput improvements; make no unsupported speed claims.
- Provide concise build/run instructions and API examples that reproduce the
  working journeys. Follow the release process when distributing executables.

This goal is complete when a person can complete the full multimodal journeys on
the deployed dev service, with correct useful results and measured performance.
Tests and documentation support that outcome; they do not define completion.

## Current state (2026-10-09)

- Upstream base: QuixiAI/embeddinggemma.c at
  `55964e25b199ddf7a9707814ec490324f749248f`.
- Windows foundation: jethac/embeddinggemma.c at
  `2cba339e028dede34c3219dac895ef039c880b64`, incorporated locally.
- A model-specific C GGML backbone, Gemma 4 media encoders, exact Gemma 4 BPE
  tokenizer, CMake build, model downloader, and multimodal client are implemented.
  Text, image, audio, video, and mixed journeys run on the Windows CPU dev service.
  One text comparison against the same Q8 GGUF in llama.cpp yielded cosine
  0.9999705; broader reference coverage remains unfinished.
  Against Hugging Face revision `914f7f89142e33e77833254d9c9b90c3cef7303b`
  in FP32, single samples yielded cosine 0.999693 (text), 0.999892 (image),
  0.999564 (audio), 0.999649 (video), and 0.999809 (text/image/audio mixed).
  The earlier audio discrepancy came from forcing eager attention in the HF
  reference: its additive mask was interpreted as boolean by the audio encoder,
  reversing attention eligibility. Correct SDPA mode uses a boolean mask; the
  440 Hz regression now passes at cosine 0.999310 and guards the mask type.
  The reference's additive log-mel floor is applied in native preprocessing.
- The observed image sizing bug has a service regression test: its square-image
  input now uses 260 tokens, matching the reference processor, instead of 85.
  Video uses a separate 140-token frame budget and excludes the generic helper's
  prose prefix. A two-frame sample produces 248 tokens. These checks establish
  input sizing; the output comparisons above cover individual samples only.
- Selective encoder loading is implemented: `--media-encoders vision|audio|all`
  skips unused weights and rejects unloaded modalities before decoding. On the
  Windows Xeon CPU service with the same Q8 files, startup private memory was
  1067.8 MiB (all), 582.7 MiB (vision), and 682.4 MiB (audio). Tested image,
  video, and audio outputs were identical to all-encoder mode.
  An encoder batching experiment was rejected on this CPU: the cache-disabled
  two-frame video warm median rose from 4050.7 to 4280.9 ms despite identical
  output. Revisit batching on accelerators using actual measurements.
- Complete media resource handling and cross-request multimodal batching are
  unfinished. Remote hardware access remains unresolved; NPU support is absent.
  Concurrent identical media HTTP bodies now share one in-flight inference,
  independently of completed-response caching, with native/OpenAI isolation.
  Unique pending work is limited to 64 entries and 128 MiB of encoded keys plus
  bookkeeping; excess work returns HTTP 503. This is not a total-memory bound.
  With caches disabled, the observed four-request CPU duplication regression
  ran four inferences before the fix (slowest caller 21.8 s), versus one after
  the fix (all callers 5.66 s; a later busy-host round took 8.24 s).
  The deployed default endpoint completed the same wave in 7.91 s. Warm CUDA
  callers improved from a slowest 1087 ms with four inferences to 460 ms with
  one; the first cold CUDA wave took 21.45 s including encoder warmup. All five
  matching Windows and CUDA modality outputs were bit-identical before/after.
  Admission limits, error propagation and API isolation pass on Windows/Linux.
  A CUDA compatible-frame encoder batching experiment was also rejected: paired
  busy-host medians worsened from 356 to 414 ms (two-frame video), 1128 to 1250 ms
  (eight-frame video), 376 to 478 ms (two images), 845 to 928 ms (four images),
  and 585 to 1035 ms (mixed shapes/audio). Both tested sequential/batched samples
  retained FP32 reference agreement above 0.999; the slower prototype was removed.
- `--persistent-cache-path` now saves bounded HTTP responses alongside the text
  cache in a `.responses` snapshot on graceful shutdown. The observed media
  restart regression failed before implementation with two recomputations; native
  Windows and Linux CUDA now reload both native/OpenAI responses with zero
  inference calls. Matching full backbone/mmproj bytes, backend, encoder selection
  and client batch limit are required. Record checksums and length checks reject
  damaged or oversized entries. Windows Ctrl-C/Ctrl-Break and Unix SIGTERM stop
  paths were exercised. Native image requests improved from 6474/7393 ms to
  32.1/32.5 ms after restart; CUDA improved from 653/201 ms to 0.85/0.79 ms.
  The files retain raw request/media bytes; persistence requires a nonzero
  response-cache budget. Weight hashing adds startup I/O and invalidates the
  previous partial-header text-cache identity once. Cross-request batching and
  semantic reuse across differently formatted bodies remain unfinished.
  The default Windows dev endpoint now uses a 64 MiB response cache and
  `.reference/dev-cache2` persistence. Its real restart loaded all five modality
  responses without inference; image/video/mixed requests fell from 5757/5539/
  6246 ms before restart to 27.1/27.2/36.1 ms, with identical 768-dimensional
  outputs. These are cache-hit improvements, not faster inference kernels.
- Decoded media input now has bounds checked before image/PCM allocation:
  16 megapixels per image/frame, 5,242,880 audio samples, and 128 MiB of retained
  decoded input/frame buffers across parts. Video frames are capped at 32 across
  the input, including the actual lazy decoder callbacks. Decoder copies and
  preprocessing/inference workspaces consume additional bounded memory.
  A 4100-square compressed PNG was previously accepted and processed; its
  regression failed on the old service and passes after the fix. Controlled
  warm rejection takes about 24 ms. Excess audio and aggregate decoded input
  are also rejected; tested valid image/audio/video outputs remain identical,
  and the mixed text/image/audio journey still uses 294 tokens.
- Valid WebM/Matroska inputs were rejected because their duration is often stored
  on the container rather than the video stream. The regression failed with
  HTTP 400 before the fix. Probing now falls back to container duration while
  retaining known stream duration; tested two-frame outputs match MP4 at cosine
  0.9999999, with 248 tokens. A second observed probe failure on a larger MP4
  with metadata at EOF is fixed by using the decoder's seekable buffered-input
  wrapper for ffprobe as well. A two-second video with 37-second audio remains
  a two-frame visual input, while a 33-frame WebM is rejected before encoding.
  Native Windows and Linux CUDA checks pass, and both default dev endpoints
  serve the new build. On the deployed services, the formerly rejected WebM
  and Matroska journeys complete with 248 tokens; cache-disabled first requests
  took 7.22/7.75 s on the busy Windows CPU and 443/498 ms on CUDA. Existing
  modality samples and a lossless WebP/PPM pair remain bit-identical. The previous
  persistent-cache and container CI runs passed. Broader resource handling and
  unknown-duration inputs remain unfinished.
- External video/WebP subprocesses now have deadlines: 10 seconds for each
  probe and 30 seconds for each decoder, starting at its first frame read.
  The observed stalled-probe regression exceeded its 15-second HTTP timeout
  before the fix. With caching disabled, it now returns HTTP 400 in 10.18 s
  on native Windows and 10.07 s on Linux CUDA; uncached text waiting behind
  it completes in 11.42/11.48 s, respectively. The timed-out child is reaped.
  Model-free checks on both platforms reject a decoder timeout both before
  its first frame and after one complete frame; partial video cannot silently
  produce an embedding. Normal EOF still succeeds. All five matching native
  Windows/CUDA modality outputs remain identical. These are subprocess limits,
  not an overall request or inference deadline. Patch application now compares
  expected source prefixes, allowing overlapping dependency patches to upgrade
  and reconfigure without rewriting already-patched files or losing local edits.
  Both default Windows/CUDA endpoints now serve the deadline build, reload
  their existing persistent responses, and return unchanged uncached video
  embeddings. The latter requests took 82.5 s on the busy Windows host and
  5.78 s on CUDA; these cold requests do not establish a kernel speed change.
  CUDA startup exceeded the helper's 300-second observation window while the
  same process remained alive in WSL file I/O; it subsequently became ready
  without restart. The decoder change now passes the Linux model CI job and
  all four inherited platform jobs.
- Startup logs now separate backbone loading, media encoder loading, and both
  full-file cache fingerprints. Two Windows-mounted-path startups took
  28.7/80.2 s in total, including 13.1/48.5 s for fingerprinting. Matching
  native-WSL-storage startups took 23.0/6.2 s, including 2.4/1.1 s for
  fingerprinting. Order was mounted/native/native/mounted; host load varied,
  so these are observations rather than a quiet-host or cold-start guarantee.
  Both native copies were validated against all original bytes. The default
  CUDA endpoint now uses `/home/jetha/embeddinggemma2-models`, retains its
  existing persistent cache, and serves the startup-timing build. All five
  uncached modality outputs on the native-storage trial were unchanged; the
  deployed primary reloaded all five responses and returned an unchanged
  uncached video in 1.79 s. This is a storage/startup improvement, not a GPU
  inference kernel optimization. Windows server compilation with the actual
  build defines passes; its existing primary remains on the deadline build.
- README describes the current dev commands. A new Linux CPU CI job builds
  the CMake implementation and runs the image regression on its service;
  its first remote run passed. The inherited CI matrix validates
  the legacy 300M foundation only. Makefile and release adaptation is unfinished.
- A native macOS ARM CPU job now builds the complete EmbeddingGemma 2 service
  with AppleClang 17 and Metal/OpenMP disabled. Its running service completes
  text, image, audio, video and mixed synthetic journeys with finite normalized
  768-dimensional outputs. First-request timings on the three-core hosted runner
  were 0.80/7.12/0.46/3.96/5.98 s, respectively; these are not performance
  comparisons against another backend. Image/audio/video/mixed token counts
  were 260/29/248/294. Container handling, decoded input bounds, decoder deadline
  classification, admission/cache safety and persistent restart journeys also
  passed. The observed image restart requests went from 5.75/5.38 s to
  0.88/0.68 ms with zero media inference calls. A follow-up actual text request
  matches the pinned native Q8 reference at cosine 0.99986903. Its first-request
  modality timings were 1.31/10.13/2.11/8.66/8.48 s on a separate runner; this
  variability reinforces that these are observations, not a performance claim.
  Full multimodal reference coverage and Metal
  execution on Apple GPU hardware remain unverified. `examples/journey2.py`
  reproduces the actual five requests without requiring a user's sample files.
- OpenAI model-field validation now precedes text/media routing. Its observed
  media-form bypass has a regression that failed before the fix; missing,
  empty, and numeric model fields now return HTTP 400 with parameter `model`.
  All five valid OpenAI modality journeys were exercised at 128 dimensions.
  Served docs and OpenAPI now expose media inputs and this port's model/default
  port (42667). The legacy model retains its existing default port.
- Local WSL exposes the RTX 5060 Ti and CUDA 13.0. The Linux CUDA executable
  builds and serves all five modality journeys on localhost:42669; both media
  encoders select CUDA. With cache disabled, exploratory warm medians included
  320.7 ms for the 260-token image and 1521.4 ms for the 248-token video.
  These timings were taken on a busy host and do not establish a CPU speedup.
  Matching CPU/CUDA samples yielded cosine 0.999789 (text), 0.999948 (image),
  0.999679 (audio), 0.999794 (video), and 0.999820 (mixed). These single-sample
  comparisons do not establish full reference parity. The Windows CPU
  development service remains available on localhost:42667. Linux configuration of a patched
  Windows dependency checkout now tolerates CRLF/LF context differences.
- x86-64 CMake builds now use runtime CPU plugins by default, covering baseline,
  AVX, AVX2, AVX-512, and newer variants. A separate dependency patch checks
  XSAVE/OSXSAVE and XCR0 before admitting vector kernels; CPUID alone is
  insufficient. Native selector checks cover missing OS register state, and
  the actual AVX-512 selector rejects QEMU's CPU without AVX. A baseline-only
  Windows service returned a finite normalized 768-dimensional embedding.
  The default Windows service on localhost:42667 now selects Skylake-X; all five
  matching modality samples were bit-identical to the previous static build.
  A mixed latency outlier did not reproduce: subsequent alternating calls were
  5.20/5.21 s (dispatch) versus 5.03/5.09 s (static) on the busy host. This is
  not a quiet-host performance comparison. Linux dispatch and native/emulated
  baseline service journeys now pass, including CI. All 14 CPU plugins load
  under QEMU without AVX; only the baseline is admitted. An alternating AVX2/
  AVX-512 experiment on the busy Xeon favored AVX-512 for long text (1721/1477 ms)
  and mixed input (7088/5970 ms); short text/audio were noisy. Both text variants
  exceed cosine 0.999 against FP32. These results retain the current selection,
  without establishing quiet-host or measured-frequency performance. ARM dispatch,
  physical older/newer CPU coverage and release packaging
  remain unfinished. The dev build requires its shared libraries and plugins.
  Source installation now includes the matching media library, CPU plugins,
  MinGW runtime DLLs and dependency license notices; adapting the inherited
  binary release flow to distribute these dependencies remains unfinished.
- Initial fixed-shape ONNX exports of the backbone, vision, and audio encoders
  run on CPU ONNX Runtime. Their assembled 294-token text/image/audio sample
  matches the FP32 reference at cosine 0.99999994 (maximum element error
  1.31e-7). Export avoids the reference's untraceable mask construction and
  audio KV-window unfold while preserving its exact boolean attention mask.
  These are FP32 graphs, totaling about 2.98 GB; BF16 NPU compilation, operator
  placement, and execution remain untested because Strix Halo access is unresolved.
  Successful CPU execution is not evidence of NPU compatibility.
- The README now contains matched, uncached text comparisons against pinned
  llama.cpp on the local Xeon / RTX 5060 Ti WSL host. The six-cell geometric
  means are 1.01x on CPU AVX-512 and 1.16x on CUDA, with minimum cosine
  0.99997 / 0.99991. The harness starts fresh servers per cell, checks concurrent
  outputs, measures both orders for at least eight seconds and retains losses.
  CPU baseline configuration explicitly excludes GPU devices and operation/KV
  offload: zero GPU weight layers alone did not establish a CPU-only comparison.
  Shared-host runs varied widely; the README shows the last full matched run
  and per-order ranges. These results are provisional: the original quiet-host
  guard observed only the WSL guest, missing Windows host CPU contention.
  The guard now samples Windows GetSystemTimes as well and requires a working
  Windows Python on PATH under WSL. Its observed-bug regression failed before
  the fix and passes on Windows/Linux. Rerun the cells with the corrected guard.
  Short concurrent CUDA text measured a
  deficit (0.79x in that run). Measure its graph rebuilding, scheduling and
  compute costs before choosing the next optimization. Media, other hardware,
  larger concurrency and wider token ranges still need comparisons.
- `EI_PROFILE_BACKBONE2=1` logs graph construction, input preparation,
  synchronized execution and output processing for each native forward. In a
  busy-host short-concurrent trace, shapes alternated between 32 tokens / one
  request and 96 tokens / three requests. Input preparation was about 0.01–0.03
  ms; most time was inside execution, including CUDA graph capture and GGML
  debug logging. Those absolute times and a debug-log filtering trial are not
  performance evidence: Windows host CPU was 1200% while the WSL guard passed.
  No logging-filter optimization shipped. Measure batch collection and graph
  reuse on a quiet host before adding graph-cache or kernel machinery.
  The CUDA development service was redeployed with profiling disabled; all five
  fresh-input modality journeys returned finite normalized 768-dimensional
  vectors, and the five persisted results matched their pre-restart vectors.
  This round establishes no user-visible inference speedup.
- Local reference sources, tool environments, models, and build outputs remain
  ignored and must not be committed.
- Oversized text requests reported the inherited 2048-token limit even though
  EmbeddingGemma 2 accepts 8192. Both native and OpenAI errors now derive their
  limit from the model configuration, preserving the legacy model's 2048 limit.
  The HTTP regression failed before the fix and passes on the redeployed
  Windows CPU and Linux CUDA services. All five modality journeys passed on
  both services. Exact 8192-token requests returned finite normalized vectors:
  CUDA versus pinned Q8 llama.cpp cosine was 0.99998452, and Windows AVX-512
  CPU versus CUDA cosine was 0.99992663. This fixes misleading client guidance;
  it does not change inference speed or increase the existing context limit.
- Media probing, decoding and preprocessing no longer hold the backbone mutex.
  A separate media mutex preserves one media request's decoded/preprocessed
  buffers at a time; encoding and backbone execution still use the shared
  inference lock. The stalled-probe regression now requires uncached text to
  complete while the probe is alive. It failed before the fix at 10,599 ms;
  after the fix text completed in 397 ms on CUDA and 471 ms on Windows CPU,
  while the probe timed out and was reaped at about ten seconds. Both development
  services were redeployed, and all five fresh-input journeys passed. Fresh
  CUDA outputs matched the pre-change vectors exactly; duplicate singleflight
  and decoded-media budget checks passed on both services. This fixes decoder
  head-of-line blocking; it does not establish a llama.cpp throughput speedup.

- `cmake --install` previously omitted `libmtmd`: the staged Windows executable
  failed with loader status 0xc0000135 and Linux failed with missing
  `libmtmd.so.SOVERSION`. Installation now includes the media library with the
  pinned llama library's actual version, CPU plugins, MinGW runtime DLLs and
  dependency license notices. The startup regression failed before the fix and
  passes on Windows/Linux from unrelated working directories without build-tree
  library paths (Windows uses only System32 on PATH). Both complete install
  directories were relocated; all five uncached text/image/audio/video/mixed
  journeys passed, with finite normalized 768-dimensional outputs and media
  counts 260/29/248/294. Logs confirm the AVX-512 plugin loads from each relocated
  install. The Windows development endpoint now runs the installed executable.
  CI serves its Linux/macOS journeys from installed payloads as well. This
  changes source installation from a startup failure to five working input
  types; busy-host timings establish no inference speedup. Binary release
  packaging and the other unverified accelerators remain unfinished.

- The macOS job now probes `MTLCreateSystemDefaultDevice` before configuring
  Metal. When a device exists, it runs the installed Metal service and compares
  all five uncached modality outputs with the installed CPU service, requiring
  cosine above 0.999 and checking that the backbone and media logs select Metal.
  This is an execution attempt, not qualification until the job proves it.
  `examples/journey2.py --compare-url` exposes the same comparison for running
  services. Its local Windows CPU versus Linux CUDA journey passed for all five
  inputs; minimum cosine was 0.99949802 (audio). These busy-host requests establish
  no throughput improvement. The first Metal attempt found an Apple Paravirtual
  device and built successfully, but startup failed: our selector expected the
  registry name `Metal`, while the pinned dependency uses `MTL`. The selector
  and runtime log checks now use the actual registry/backend names. The existing
  explicit-Metal CI journey reproduced the bug before the fix; its rerun must
  pass before claiming Metal execution. A virtual GPU is not evidence of physical
  Apple Silicon performance.
  The corrected selector starts the Metal service and both media encoders as
  `MTL0`, but its first text request aborts the process. The CI failure trap now
  prints the actual service stderr so the next run can identify the assertion
  or unsupported operation; startup alone does not qualify Metal execution.
  The captured abort identifies `per_layer_model_proj.weight`: it is BF16 even
  in the Q8 model, and this virtual Metal device reports no BF16 support. The
  loader now widens unsupported BF16 accelerator weights to FP32 with an exact
  conversion before allocating/uploading them. CPU and BF16-capable accelerators
  retain their original weight representation. This projection grows from 12
  to 24 MiB on the affected device; the temporary FP32 upload buffer is freed
  after loading. The Metal journey reproduced the abort before this change and
  must pass after it before qualification.
  The rerun (`0356838`, CI 37890404735) passed all six jobs. The installed Metal
  service completed all five uncached journeys on the Apple Paravirtual device:
  text/image/audio/video/mixed cosine versus CPU was
  0.99976394/0.99995557/0.99950580/0.99987662/0.99986856, above the 0.999 gate.
  Logs select `MTL0` for the backbone and both encoders. This changes Metal from
  aborting on the first request to five working input types. GGML's CPU fallback
  remains available; this virtual device does not qualify an all-GPU path or
  physical Apple Silicon throughput. The rebuilt Linux CUDA service was
  redeployed on 42669, and all five fresh outputs exactly matched its pre-change
  vectors; CUDA retained BF16 weights. No inference speedup is established.
- The relocated Linux source install also completed the uncached text journey
  under `qemu-x86_64 -cpu qemu64` with AVX unavailable and all CPU plugins present.
  The baseline plugin was selected, and the response contained 768 finite
  normalized dimensions. The existing CI fallback journeys now use the installed
  executable, covering deployment dependencies as well as build-tree dispatch.

- The opt-in CI CPU comparison builds pinned llama.cpp against the installed
  shared GGML library and runs the existing six text cells with two threads,
  caches off, both orders and the quiet-host guard. Its first run failed before
  measurement because `GGML_BACKEND_PATH` expects a library file, not a directory.
  The baseline now has symlinks to the exact installed CPU plugins beside its
  executable, with no override. The local rebuilt baseline starts and returns
  a finite normalized 768-dimensional vector; rerun CI before publishing rates.
- Nsight Systems 2025.3.2 captured API calls but no GPU kernel timings with the
  installed CUDA 13.4 driver. A separately extracted NVIDIA CLI 2026.5.1 collects
  kernel data. In one 120-request, 32-token/four-client trace, quantized matmuls
  consumed 66.1% of GPU kernel time (stream-k fixup alone 27.9%), versus 6.9% for
  attention. Graphs rebuilt on 64 of 68 forwards, mostly alternating one and
  three requests. A 2 ms batch-wait trial collected more four-request batches
  but still rebuilt most graphs. Matched, periodically flushed captures had
  108.9/129.0 ms total kernel time for 200/2000 us waits. Instrumentation and host
  contention prevent a throughput claim; no wait/default/kernel change shipped.
  Measure stable graph reuse and matmul costs before attempting an attention
  rewrite. The existing Windows CPU and Linux CUDA development services remain
  available; profiling services shut down after their actual request journeys.

- An opt-in CUDA short-text graph cache (`EI_GRAPH_CACHE2=1`, default off)
  retains two shapes of at most 256 aggregate tokens; media and longer text use
  the normal workspace. The same 120-request/four-client trace now rebuilds 2
  graphs instead of 64, with 330 actual CUDA graph launches instead of 20.
  The retained 32/96-token workspaces total 15.18 MiB. Thirty-six changed-input,
  unequal-length, eviction and long-text requests exactly matched the proven
  cache-off service; all five fresh modality outputs also matched exactly.
  The graph topology and numerical kernels are unchanged. Quiet-host matched
  throughput is still required before a speed claim or default promotion.
  A follow-up Nsight capture with the cache enabled needed explicit graph-node
  tracing; the default capture had no kernel timings and was discarded. The
  successful trace attributes about 62.6% of kernel time to quantized matmuls,
  including 26.1% to stream-k fixup, versus 7.0% to attention. This still favors
  a focused matmul scheduling experiment before an attention rewrite. Captured
  kernel totals include profiling overhead and do not establish service speed.
  A private, default-off Q8 tiling trial removed fixup for short dense matmuls,
  but was rejected: 106 fixed 32-token requests increased combined matmul kernel
  time from 155.5 to 170.1 ms; 106 fixed three-sequence/96-token requests increased
  it from 272.0 to 285.6 ms. Both controls retained exact outputs; tiling's minimum
  cosine was 0.999920/0.999907. This was not a throughput comparison. The source
  trial was reverted; no new kernel flag or default is shipped. Skipping fixup
  alone sacrifices enough occupancy to lose on these shapes. Future matmul
  scheduling work must preserve useful occupancy, with actual measurements.

- The opt-in CPU comparison completed on CI 37892946034 (`1e72b77`): Ubuntu
  24.04/GCC 13.3, AMD EPYC 9V74 with four logical CPUs and two inference threads,
  matching installed shared GGML kernels/plugins, caches off and both orders.
  The six ratios were 0.996/1.008/1.006/1.008/1.008/0.994 for 32, 256, and 1024
  tokens at one/four clients. Geometric mean 1.003x is effectively parity;
  minimum cosine was 0.999970. README now includes every cell and order range,
  including losses, separately from the provisional WSL figures. No CPU speed
  advantage is established. The default-off graph-cache commit `c15270c` also
  passed all six CI jobs, including Linux and macOS deployed journeys.

- The inherited installer actually requested QuixiAI's 300M release checksums.
  It now targets this repository, its own asset/executable prefix, and
  `EMBEDDINGGEMMA2_*` overrides. A mocked-download regression reproduced the
  wrong URL before the fix and now verifies the correct URL and no installation
  after download failure. No binary releases exist; CMake source installation
  remains the working route. Shared-library staging and release publication are
  unfinished; this identity fix establishes no inference speedup.

- Packed Q/K/V projections are retained as an opt-in experiment (`EI_QKV2=1`,
  default off, Q8_0 CPU/CUDA). They add 27.62 MiB of weights. Ten Windows CPU
  checks, including 1024-token text and all five modalities, matched exactly;
  CUDA's matching checks passed cosine 0.999, with minimum 0.999830 (mixed).
  A fixed 106-request/32-token capture reduced quantized matmul launches from
  22,680 to 17,640, combined matmul/fixup time from 155.5 to 130.1 ms, and total
  captured kernel time by 7.2%. The 96-token capture also had lower raw totals,
  but missing events prevent a clean comparison. These are GPU profiling
  results, not deployed throughput gains; quiet-host comparisons remain needed.
  An actual CUDA restart reused a separate-projection response while packed
  fresh inference differed. Cache identity now includes the effective packed
  mode for native and HTTP persistence, preserving the original identity when
  off. That regression failed before the fix and passed on CUDA and native
  Windows CPU after it. The CI CPU comparison can request packed QKV separately
  and verifies its actual startup log before measuring. Other accelerators and
  broader CPU hardware remain unqualified for this experimental path.

- The packed-QKV CPU comparison completed in CI run 37899814743 on an EPYC
  7763 runner (four logical CPUs, two inference threads, Ubuntu 24.04/GCC 13.3,
  same installed shared kernels and Q8 weights, caches off, both orders).
  Geometric mean against llama.cpp was 1.012x; minimum cosine was 0.999927.
  Ratios for 32/256/1024 tokens with one/four clients were 0.993/1.007,
  0.999/1.006 and 1.040/1.029. Non-benchmark CPU load was 1.6–2.3%. README
  includes every cell and order range. This runner differs from the earlier
  EPYC 9V74, so these results do not isolate the effect of packing.

- An opt-in CUDA global-attention fallback (`EI_CUDA_GLOBAL_ATTN2=1`, default
  off) removes observed CPU attention splits when the pinned CUDA flash kernel
  rejects an input shape. Explicit masked attention is limited to 2048 aggregate
  tokens (64 MiB maximum four-head scores); supported flash shapes retain the
  original path. Larger unsupported shapes now pad K/V to a 256-key boundary
  and mask added keys, selecting native CUDA flash when supported and retaining
  the existing fallback on devices without it. Actual 32-token placement changed from nine
  GPU/CPU splits to one CUDA split. In both orders, 100-request warm HTTP medians
  on the contended local RTX 5060 Ti improved from 11.6 to 5.0 ms and 12.9 to
  5.0 ms, with graph caching on, packed QKV off and both result caches off.
  These are end-to-end latency observations, not quiet llama.cpp throughput.
  All five modalities passed cosine 0.999 (minimum 0.999848); enabling packed
  QKV too passed at minimum 0.999824. Unequal-batch isolation and lengths
  255/256/257, 1023, 2047/2048/2049 passed in the initial bounded experiment;
  supported aligned lengths matched exactly with packed QKV off. The existing persistent
  numeric-mode regression also passes for this mode: cached results match fresh
  inference after a mode change. Quiet-host and broader hardware comparisons
  remain necessary before enabling this path by default.
  Current CUDA and Windows CPU builds with both numeric flags off retained
  exactly matching outputs for all five modalities. Both persistent numeric-mode
  regressions pass on CUDA after the cache-fingerprint refactor.

  The larger padded path was selected after actual CUDA placement and paired
  HTTP measurements: 2049-token median latency fell from 2094 to 152 ms and
  from 2515 to 338 ms in the reverse order (20 measured warm requests per pass,
  packed QKV off, result caches off, contended local host). The final 2049-token
  graph has one CUDA split. Padding lost against explicit attention at 32 tokens
  in both orders, so it is used only above the explicit-score memory limit.
  Final 8191-token and unequal 2049-token API requests passed cosine
  0.999938 and 0.999917, with batch isolation passing. OpenAI usage confirmed
  their token counts; the unequal inputs (1023/1026) crossed the packing cutoff
  and ran as separate forwards, so this was not a combined-engine batch check.
  The 2049-token persistent-mode regression passed;
  cache identity now uses the global-attention v2 domain so older variant
  responses cannot survive this numeric change. All five modalities passed on
  the final opt-in CUDA service; flags off retained exactly matching outputs
  on current Windows CPU and CUDA builds.
  The 2049-token request also passed with packed QKV enabled (cosine 0.999922),
  retaining one CUDA split.
  A fresh matched CUDA comparison with packed QKV, graph caching and global
  attention enabled reused the service's exact current static GGML archives in
  pinned llama.cpp. Its quiet-host guard rejected the run before any measured
  cell: Windows CPU stayed around 1179–1200% for five minutes. No new llama.cpp
  throughput result was published, and the experimental flags remain off by
  default. Comparison metadata now records the graph-cache flag and checks its
  actual startup marker alongside the other numeric modes.

- CI 37902823700 hit its 120-second Metal readiness deadline while the service
  was still initializing the embedded library. The workflow now observes the
  same live processes for a bounded 300 seconds and logs readiness elapsed time.
  The following run 37904005129 completed all six jobs; Metal startup took
  32 seconds and all five CPU/Metal comparisons passed (minimum 0.999506).
  This does not establish physical Apple GPU throughput or a faster startup.
  Run 37906879977 then reached the 300-second deadline because its explicit CPU
  service stalled inside Metal library initialization; the separate Metal
  service reached readiness. GGML's registry eagerly creates Metal devices even
  for CPU selection. The server now disables Metal device enumeration before
  initializing an explicitly selected CPU engine on macOS. The native journey
  checks reject any Metal shader initialization in that CPU process. Run
  37912238090 passed all six jobs: macOS CPU became ready in 2 seconds and
  Metal in 22 seconds; all five CPU/Metal comparisons passed (minimum cosine
  0.999506). This removes the unnecessary shader dependency from explicit CPU
  startup; it does not establish physical Apple GPU throughput.

- Static auxiliary input reuse (`EI_REUSE_INPUTS2=1`, default off) addresses
  measured 8191-token preparation costs of 388–524 ms per warm forward. Positions,
  full/local masks and pooling weights now have dedicated primary-backend
  storage. FP16 mask values are exact zero/negative infinity, eliminating casts.
  Layout reuse requires identical sequence offsets in the same live graph;
  changed boundaries, graph rebuild/eviction and compute failure invalidate it.
  Token IDs and raw media rows are always refreshed. The 8191-token CUDA graph
  owns 256.05 MiB of auxiliary storage. The profiled warm preparation fell to
  0.004–0.005 ms. With profiling disabled, paired warm HTTP medians fell from
  2106 to 392 ms and from 1207 to 413 ms in the reverse order (20 measured
  requests per pass, six threads, QKV off, global attention/graph caching on,
  result caches off, contended RTX 5060 Ti host). These are not quiet throughput
  comparisons against llama.cpp.
  Changed inputs/boundaries, graph rebuilds and short-shape eviction matched
  exactly against the flag-off path on CUDA and Windows CPU. The safety test
  confirms one actual 2049-token/three-sequence engine forward (682/683/684),
  correcting the earlier API-batch coverage gap. All five modality samples
  matched exactly on both backends; two changed 8191-token CUDA inputs also
  matched. No numeric cache identity change is needed for these identical
  outputs. Comparison metadata checks the requested input-reuse startup mode.
  Other accelerator backends remain unverified. Linux installed-service CI
  now exercises the focused sequence-boundary and graph-lifetime safety check.
  The CUDA dev service on port 42669 now runs this path with global attention
  and short-text graph caching enabled, QKV off. Its deployed text, image,
  audio, video and mixed journeys passed against the existing Windows CPU
  service (minimum cosine 0.999498). HTTP response caching remains available
  on that dev service; the latency measurements above disabled it.
  CI runs 37913096620 and 37913466268 passed all six jobs, including the
  installed Linux sequence-boundary and graph-lifetime safety check.

- CUDA local key padding (`EI_CUDA_LOCAL_ATTN2=1`, default off) follows a complete
  12-forward 8191-token GPU capture: local attention consumed 53.0% of kernel
  time, global attention 16.3%; all 240 local and 48 global calls were captured.
  Local keys now pad to a 256-key stride from 1024 aggregate tokens, with added
  keys masked. This enables the pinned kernel's grouped-query and mask-scan
  paths. The complete trial capture retained all expected calls: local flash
  kernels fell from 2195 to 1145 ms, total kernel time from 4141.74 to 3169.92 ms
  (23.5%). Paired warm HTTP medians fell from 514 to 402 ms and from 362 to
  282 ms in reverse order (20 measured requests per pass, QKV off, global
  attention/input reuse on, result caches/profiling off, contended RTX 5060 Ti).
  No quiet-host llama.cpp throughput claim follows from these measurements.
  Actual combined 2049-token batches and changed boundaries passed cosine
  0.999878 or better. Five images (1292 raw tokens) and a long mixed input
  (1326 raw tokens) also passed; all five short modality samples and aligned
  8192-token inputs remained bit-identical. The input-reuse lifetime/boundary
  safety check passed with local padding enabled. The numeric cache domain
  distinguishes this mode; its 2049-token persistent-cache check passed with
  cached/fresh responses identical and original/variant outputs different.
  The flag applies only to CUDA. All five Windows CPU modality samples stayed
  bit-identical with the flag set (the CPU backend ignores it).
  The deployed CUDA service on port 42669 now enables local padding alongside
  global attention, input reuse and short-text graph caching, QKV off. All five
  deployed modality journeys passed against the existing Windows CPU service
  (minimum cosine 0.999498). It also returned finite normalized outputs for
  8191-token text and five images (1292 tokens). Its warm 8191-token median was
  360 ms for 20 measured requests under current contention; each HTTP body was
  unique to bypass the dev service's response cache, and the text cache is off.
  CI run 37916077030 passed all six jobs for this deployed change.

- CUDA lower mask-range scanning (`EI_CUDA_LOCAL_RANGE2=1`, default off) skips
  fully masked leading key tiles from 1024 aggregate tokens. Both enabled and
  disabled paths were bit-identical against the previous deployed CUDA build
  across all five modalities, 8191/8192-token inputs, actual three-sequence
  2049-token batches with changed boundaries, five images and long mixed inputs.
  The observed reverse-order pool cleanup assertion is fixed: the existing
  public `tests/qkv_cache2.py --backend cuda --mode cuda-local-range --tokens
  1024` failed before the fix and passes afterward. Its 2049-token variant also
  passes, with cached/fresh responses identical and distinct cache identities.
  The reusable-input boundary/rebuild/eviction safety check passes with this
  mode enabled, exercising both F16 and F32 attention masks.
  Paired warm 8191-token HTTP medians improved from 258.525 to 195.758 ms and
  from 264.155 to 194.904 ms in reverse order (20 measured requests after six
  warmups per pass, six threads, QKV off, global/local attention, input reuse
  and graph caching on, profiling/result caches off, RTX 5060 Ti). That is
  approximately 25% lower latency than the already padded local-attention path;
  this is not a matched llama.cpp throughput comparison.
  A complete 12-forward capture contains 240 local and 48 global flash calls.
  Local flash time fell from 1145.116 to 199.956 ms; its mask scan increased
  from 37.15 to 113.272 ms. Total kernel time fell from 3169.918 to 2209.150 ms
  (30.3%); global flash time was 658.797 ms. The CUDA dev service on port 42669
  now enables this mode alongside its previous flags. All five deployed
  modalities passed against Windows CPU (minimum cosine 0.999715), and
  8191-token text plus five images (1292 tokens) returned normalized outputs.
  Its warm 8191-token median was 221.990 ms for 20 unique HTTP requests after
  six warmups, text cache off. The response cache remains available to users.
  CI run 37927324328 passed all six jobs. The matched CUDA comparison against
  llama.cpp reached its five-minute quiet-host deadline without measuring a
  cell; no new llama.cpp speed claim is supported.
  The existing complete trace also identifies a global-attention resource
  constraint: 252 registers/thread, 256 threads/block and 84,224 bytes of shared
  memory/block on the RTX 5060 Ti (65,536 registers and 102,400 shared bytes/SM).
  Global attention already groups all four query heads and runs one block/SM;
  smaller query tiles are the next measured candidate, rather than adding GQA
  grouping already present. Local attention uses 255 registers/thread,
  128 threads/block and 35,328 shared bytes/block, with two blocks/SM.

- The 32-column global-attention tile trial was rejected. Both modes built;
  the disabled CUDA journey and all five Windows CPU modalities were
  bit-identical. Enabled CUDA outputs passed cosine 0.999913 or better, but
  paired 8191-token medians worsened from 224.808 to 241.772 ms and from
  240.561 to 270.355 ms in reverse order. A complete 12-forward trace retained
  all 240 local and 48 global calls: global flash time increased from 658.797
  to 962.025 ms, total kernel time from 2209.150 to 2512.615 ms. The actual
  32-column kernel still uses 256 threads, 196 registers/thread and 67,584
  shared bytes/block, allowing one block/SM. Its combine buffer keeps shared
  memory above the two-block limit.
  The subsequent 8-column trial was also rejected. Its enabled journey passed
  cosine 0.999913 or better, and its disabled journey stayed bit-identical, but
  paired 8191-token medians worsened from 226.778 to 319.615 ms and from
  203.144 to 305.322 ms in reverse order. A complete 12-forward trace retained
  all expected calls: global flash time rose to 1690.279 ms, total kernel time
  to 3231.480 ms, while local flash stayed near 197 ms. This variant actually
  allowed two blocks/SM (128 threads, 230 registers/thread, 41,376 shared
  bytes/block), so more resident blocks alone did not improve this workload.
  Both unshipped overrides and their cache/test plumbing are removed. The
  deployed service retains the original 64-column global tile alongside the
  validated lower-mask-range path; its working binary is unchanged.

- Dependency debug logging caused an actual full-model startup timeout with a
  Windows-mounted log sink (180-second readiness deadline). Changing only the
  sink to a Linux file let the original binary start in 7.667 seconds, writing
  195,551 bytes before the first request. Default logging now filters dependency
  DEBUG messages, preserving normal diagnostics, warnings and startup timings;
  `EI_DEBUG_LOG2=1` restores full diagnostics. The same Windows-log journey
  completed startup and all 18 requests in 9.3 seconds after the fix, with
  backbone/media loading totaling 1.944 seconds and every response bit-identical
  against the previous deployed CUDA service. Its log was 17,909 bytes after
  those requests. Coverage includes all five modalities, 8191/8192-token text,
  actual 2049-token batches with changed boundaries, five images and long mixed
  inputs. The focused log regression fails against the original captured log
  and passes against the fixed log; it also checks that warnings remain visible.
  Windows CPU and CUDA builds pass. All five Windows CPU modality responses
  remain bit-identical. The deployed CUDA service on port 42669 retains global
  attention, local padding/range, input reuse and graph caching, QKV off and
  profiling unset. Its five modality journeys passed against Windows CPU
  (minimum cosine 0.999715); 8191-token text and five images returned finite
  normalized embeddings. Increasing host contention made its subsequent
  startup slower (35.458 seconds for backbone/media loading), so the early
  controlled startup observation is not a guaranteed service startup time.
  No inference throughput claim or numeric cache identity change follows.

- CUDA GeGLU (`EI_GEGLU2=1`, default off) fuses the two GELU/multiply pairs per
  backbone layer using GGML's existing split operation, including the strided
  per-layer-input view. Both enabled and disabled CUDA paths were bit-identical
  against the deployed unfused build in all 18 cases: all five modalities,
  8191/8192-token text, actual changed-boundary 2049-token batches, five images
  and long mixed inputs. Reusable-input graph rebuild/eviction safety checks
  and the 2049-token persistent-cache isolation check pass. The cache domain
  distinguishes this mode even though observed outputs remain identical.
  Paired warm 8191-token HTTP medians fell from 195.396 to 182.241 ms and from
  200.125 to 188.415 ms in reverse order (20 measured requests after six warmups,
  six threads, QKV off, global/local attention, lower mask range, input reuse
  and graph caching on, result caches/profiling off, RTX 5060 Ti). This is
  approximately 6-7% lower latency, not a matched llama.cpp throughput claim.
  A complete 12-forward capture retained all 240 local and 48 global attention
  calls. Kernel count fell from 12,804 to 12,228, total kernel time from
  2209.150 to 2130.040 ms (3.6%). Its 576 fused calls cost 165.713 ms; local/global
  attention times stayed near 199/661 ms. The short-request trial showed no
  regression in either order but too much host variation for a speedup claim.
  CPU outputs also remained identical in the 1024-token timing trial, but
  changing host load reversed the measured outcome: unfused/fused medians
  2777/3640 ms in one order, 5033/3564 ms in reverse. No CPU benefit is
  established, so the retained flag applies only to CUDA. With the flag set,
  all five Windows CPU modality samples stayed bit-identical and the fused
  startup marker was absent. The deployed CUDA service on port 42669 now
  enables GeGLU alongside global/local attention, lower mask range, input reuse
  and graph caching, QKV off and profiling unset. All five deployed modality
  journeys passed against Windows CPU (minimum cosine 0.999715). Deployed
  8191-token text and five images (1292 tokens) returned finite normalized
  embeddings; its warm median was 284.023 ms for 20 unique requests after six
  warmups under increased contention. This is not comparable to the preceding
  deployment's 221.990 ms observation and is not a deployed throughput win.
  CI run 37935509482 passed all six jobs for the preceding logging change;
  the existing Linux/macOS full-encoder jobs now also invoke its focused log
  regression. These results do not yet validate this GeGLU commit.

- CI run 37937705376 passed all six jobs for the deployed GeGLU change.
  A keep-alive receive-buffer release/heap-trimming experiment is discarded.
  Four completed 12 MiB requests left a 96 MiB RSS increase in the original
  probe; releasing receive capacity reduced that observation to 72 MiB.
  Adding glibc heap trimming still produced 35-83 MiB increases across revised
  CUDA/CPU probes. Runtime logging confirmed that 16 MiB receive capacities
  shrank to 43 bytes and that trimming executed, but those facts did not prove
  a repeatable total-memory benefit or qualify allocator-wide trimming costs.
  The unshipped code and unreliable RSS regression are removed. Both binaries
  are rebuilt from the retained implementation; the deployed services were
  unchanged throughout this experiment. Total resource handling remains open.

- Packed QKV with GeGLU was measured at 8191 tokens before changing deployment.
  Warm HTTP medians were 220.777/202.257 ms (unpacked/packed) in one order and
  213.802/213.411 ms in reverse (20 measured requests after six warmups, the
  existing CUDA flags on, result caches/profiling off). Cosine was 0.999954.
  A complete 12-forward capture retained all 240 local and 48 global attention
  calls plus 576 GeGLU calls. Kernel count fell from 12,228 to 10,500, but total
  kernel time only fell from 2130.040 to 2115.030 ms (0.7%). Matrix time stayed
  near 425 ms despite fewer calls; activation quantization fell from 139.352 to
  112.914 ms, while normalization/rope grew slightly. The reverse HTTP order
  was effectively flat, so the dev service keeps QKV off. Fewer launches alone
  do not establish a useful improvement on this captured CUDA workload.

- A contiguous GeGLU indexing specialization was also rejected. All 18 CUDA
  cases remained bit-identical, but paired 8191-token HTTP medians changed from
  235.928 to 222.825 ms in one order and from 201.857 to 208.483 ms in reverse.
  The complete capture confirmed 288 contiguous and 288 strided GeGLU calls,
  all 240 local/48 global attention calls and 12,228 kernels overall. GeGLU
  time was 164.356 ms versus 165.713 ms; total kernel time was 2128.923 versus
  2130.040 ms, effectively flat. Removing row division did not expose a useful
  speedup. Its source patch and build plumbing are removed, and the retained
  CUDA binary is rebuilt. The existing trace also confirms 11 CUDA graph
  launches after the initial forward; long requests already reuse CUDA graphs.
  No deployed user-visible improvement resulted from these rejected trials.
  The next direction is the unfinished release packaging/fresh-install journey.

Keep this file current as implementation decisions and verified evidence change.

- Release packaging now retains raw executables with matching runtime archives.
  A raw Windows executable alone failed with 0xC0000135; the real CPU payload
  contains 22 DLLs, including 14 portable CPU variants, plus dependency notices.
  Its relocated service completed all five uncached modalities with cosine
  1.00000000 against the accepted CPU service. The primary Windows endpoint now
  runs the new installer's versioned application directory; fresh requests for
  all five modalities again matched the packaged comparison service exactly.
  All loaded model/compiler runtime DLLs came from that installed directory.
  This fixes distribution startup, without establishing an inference speedup.
- Unix and Windows installers verify the executable and runtime checksums before
  switching their launcher and preserve the existing application after a corrupt
  download. Real package tests passed with installation paths containing spaces.
  The Unix test first caught a global shell variable replacing the executable
  filename with the archive filename; Windows PowerShell 5 first rejected valid
  native stderr from --help. Both bugs were fixed and the same journeys passed.
  Linux packaging checks used the existing relocated cb14b3e CPU installation;
  its installed application completed all five uncached modalities with finite,
  normalized 768-dimensional outputs and all model libraries loaded from the
  application prefix. A fresh current portable Linux build is still rebuilding.
  These are local
  download fixtures, not published-GitHub installer acceptance.
- The port release runbook and asset verifier now describe nine native
  executable/runtime pairs (18 checksum entries), including Windows CPU and
  Linux ARM64 CUDA for GB10. Linux CI exercises actual package installation.
  No version/tag/release was created. Complete same-commit native qualification,
  accelerator hardware access and final GitHub installation remain unfinished.

- Windows now has a separate Gemma 2 native CI job using MinGW64, rather than
  relying on the legacy 300M job. It stages actual executable/runtime assets,
  exercises the installer under Windows PowerShell 5, then runs all five
  modalities and existing media/API safety checks from the extracted package.
  The job is added for real package coverage; its first run is pending and does
  not establish new hardware support or a service performance improvement.

- 2026-10-10: Short CPU text was blocked behind the media encoder's unrelated
  graph: a four-image request took 32.880 seconds and contended text 32.491
  seconds; the focused regression then failed at 26.692 / 27.094 seconds.
  The backbone mutex now covers only backbone compute, leaving the encoder's
  separate backend/scheduler under the existing media mutex. Immutable token
  table reads and media cleanup also run outside the backbone lock. Admission
  still allows only one decoded media request, with the existing byte/context
  limits. No new numeric path or cache domain is introduced.
  The installed final Windows primary passed at 0.872 seconds for contended text
  during a 23.536-second four-image request, with concurrent and standalone
  text/media vectors identical. Busy-host media totals establish no throughput
  improvement. The regression is in native Linux and Windows service CI.
- CUDA concurrent text and 16 images (4158 tokens in that sample) exactly matched
  the accepted original CUDA primary; text completed before the media response.
  All five sequential modalities also matched exactly. Both development primaries
  now run the narrowed-lock implementation, keeping their cache settings and
  CUDA numeric flags. Fresh final requests for all five modalities passed on
  both primaries, with CUDA versus CPU cosine at least 0.99978864. The current
  installed Linux package also passed. Native Metal/ROCm/XPU concurrency is
  unverified.
- The fresh portable Linux CPU rebuild and runtime package are now complete.
  Its actual locally downloaded installer passed checksum-failure preservation,
  and the final installed payload ran all five modalities with finite normalized
  768-dimensional outputs. Loaded model libraries stayed within its own prefix.
  Packaging commit 791546a passed all six CI jobs (37947457519). The first native
  Windows Gemma 2 job for 3595152 passed build/staging, PowerShell 5 installation
  and all five packaged modalities; all seven CI jobs passed (37947821308).

- 2026-10-10: Media arrays performed one backbone forward per input. Eight
  audio inputs measured 246.2 ms end to end, with 57.8 ms in encoders and 144.5 ms
  in the backbone. An opt-in EI_MEDIA_BATCH2 path now prepares inputs serially,
  combines short raw rows with per-sequence offsets, and computes one backbone
  forward per bounded group. It preserves output order, per-input pooling,
  usage totals and the complete 8192-token single-input path. Pending rows are
  allocated under the existing media mutex, so one decoded input and at most
  2 MiB pending rows coexist with the current input (at most 16 MiB raw rows).
  This implements request-array batching; cross-request raw batching is still
  unfinished. Its numeric cache domain is embeddinggemma2-media-batch-v1.
- Initial 4096-token groups regressed four 1024-token inputs by 12-15% and eight
  512-token inputs by 5-6% in both orders. Those groups are rejected. The retained
  path groups at most 1024 tokens, with individual inputs capped at 512 for
  grouping; longer inputs bypass batching. Eight-audio warm paired CUDA medians
  were 141.519 -> 125.049 ms and 139.134 -> 120.905 ms in reverse order, caches and
  profiling off. CPU medians were 1426.413 -> 1319.185 and 1376.291 -> 1289.362 ms.
  A single CPU mixed-array pass slightly regressed (15361.56 -> 15701.72 ms).
  CUDA mixed-array medians were 5403.321 -> 2149.443 and 2092.105 -> 1934.734 ms;
  concurrent CPU work heavily distorted the first order, so these do not support
  a general speedup claim. CPU primary remains off; dev CUDA enables the flag.
- Default-off outputs matched the accepted CUDA service exactly for all five
  modalities and the audio array. Enabled mixed arrays, unequal inputs, changed
  equal-total sequence boundaries, group splits, and a full 8192-token input
  between short inputs passed; longer bypassed inputs remained exact. Minimum
  exercised CPU/CUDA cosine was 0.999793. Raw batch=3/8 graph traces confirmed
  actual combined forwards. Windows CPU, native Linux CPU and CUDA passed the
  focused isolation/splitting/OpenAI-ordering check. Windows/CUDA cached and
  fresh batched responses matched, with a distinct persistent cache identity;
  original and batched vectors genuinely differed. Linux/Windows CI runs these
  safety checks. Other accelerators' batching remains unverified.
- b55577b passed all seven CI jobs (37949612556). The narrowed-lock improvement
  and native Windows packaging checks are now terminal successes. The new
  multimodal batching implementation still needs its own complete CI run.

- Dev CUDA is running EI_MEDIA_BATCH2=1 with the accepted attention/GeGLU/input
  reuse flags and existing persistent response cache settings. Its eight-audio
  array exactly matched the qualified uncached candidate. Fresh text, image,
  audio, video and mixed requests completed on both primaries; minimum CUDA
  versus CPU cosine was 0.99983622. Windows primary keeps the accepted b55577b
  installed binary with batching off; the new Windows build and opt-in path were
  exercised separately. No binary release or version/tag change was made.

- 2026-10-10: Concurrent audio profiling separated encoder/backbone time from
  Windows-to-WSL transport: native Linux requests took about 11 ms versus 35 ms
  from Windows, with about 23 ms spent connecting. Moving logs between Windows
  and Linux storage did not materially change this. Reused connections exposed
  a separate roughly 40 ms request-body stall in WSL's localhost TCP relay.
  Linux now requests TCP_QUICKACK before each remaining body read, preserving
  the existing byte limits, framing and numeric/cache identity. A single initial
  ACK fixed 43 KB bodies but left a 2 MiB tail stalled; the retained loop fixes
  both. The focused real-bug regression failed on the original service at
  42.7 ms versus 0.5 ms for a small request and passed on the final candidate.
- The final CUDA primary is deployed with its existing flags and persistent
  response cache. Fresh text, image, audio, video and mixed vectors exactly
  matched the original uncached CUDA service; OpenAI model validation passed.
  Windows-to-WSL reused-connection audio medians were 56.571 -> 19.391 ms and
  55.737 -> 14.429 ms in reverse order; eight-client waves were 104.274 ->
  90.054 ms and 103.216 -> 87.429 ms. Every call used a unique response key;
  the comparison service had caching disabled and the primary retained its
  cache. A pass during builds was flat at eight clients, so no general throughput
  claim is established. Native Linux and Windows builds passed; the relay
  regression passed from Windows and Linux, and on the Windows CPU primary.
  Cross-request raw batching remains unfinished, and accelerator/hardware/release
  gaps remain. Request-array batching commit 05c4e53 passed all seven CI jobs
  (37953271489); this transport fix still needs its own CI run.

- 2026-10-10: Transport commit 6cddc87 passed all seven CI jobs (37956101787).
  A cross-request prototype merged compatible queued JSON requests, with bounded
  eight-request/8 MiB groups, error isolation and a separate numeric cache domain.
  Its first eight-client audio waves regressed from 73-74 to about 101 ms and
  32-client waves from 285-286 to 319-322 ms. Raw graph caching, slots by batch
  width, and a demand-only 0.5 ms collection window still produced inconsistent
  gains and regressions. The entire prototype was removed; no scheduler flag,
  cache domain, extra graph slots or admission change is retained or deployed.
  Cross-request batching remains required. The next approach must avoid repeated
  JSON materialization and transient graph-shape overhead, rather than ship the
  regressing prototype.
- Changed course to media parsing. The actual decoder's alphabet search cost
  273-310 ms for 8 MiB in an isolated paired run; a byte lookup cost 45-52 ms,
  with identical bytes across the full alphabet, padding and data-URL samples.
  The retained lookup preserves the original bounds and error handling and
  introduces no numeric path or cache identity change. Real malformed padding,
  invalid alphabet and non-ASCII input were still rejected with HTTP 400.
- Matching 64 MiB response caches, unique keys and native Linux fresh connections
  measured a 12,960,131-byte 1800x1800 PPM upload at 780.027 -> 610.578 ms and
  645.479 -> 541.843 ms in reverse order (six measured warm calls per pass).
  Body construction was outside the HTTP timer. Earlier passes during builds
  were 951 -> 754 and 977 -> 686 ms; these shared-host observations establish
  no general throughput claim. All large-upload vectors were identical.
  The final CUDA primary is deployed with its existing flags and persistence;
  all five fresh modalities exactly matched the qualified candidate, and the
  same fresh large upload completed at a later 454.432 ms median. Native Windows,
  Linux CPU and CUDA builds passed. Windows primary still runs its accepted
  installed binary; the lookup implementation needs its own CI run. Hardware,
  cross-request batching and complete release-matrix gaps remain unfinished.

- 2026-10-10: Decoder commit 303392f passed all seven CI jobs (37959862276).
  Its portable Windows executable/runtime pair was staged and installed through
  the actual PowerShell 5 installer with local download fixtures. Installation
  with spaces and checksum-failure preservation passed. The installed candidate
  and final Windows primary completed all five uncached modalities with vectors
  exactly matching their accepted comparison services. Model/compiler runtime
  DLLs load from the installed prefix, selecting ggml-cpu-skylakex; its MIT
  license matches the source. Both development primaries now include the lookup.
  Windows retains six threads, batching off and its existing persistent cache.
- The first valid 12,960,131-byte Windows image-upload order was effectively flat
  at 33784.817 -> 33898.775 ms. Encoder and backbone time dominated. Host CPU was
  78% busy in a later snapshot; the native CPU library code sections were
  unchanged despite different package hashes. The remaining timing pass was
  intentionally stopped because it could not qualify a CPU speedup. No such
  claim is made. The candidate was stopped after final deployed journeys; no
  binary release, tag or version change was created. Hardware, cross-request
  batching, quiet comparisons and the full release matrix remain unfinished.
- CPU source inspection found the backbone leaves its threadpool unset; the
  actual GGML path creates and frees a disposable pool per forward. Measure
  that cost and steady requests before deciding whether to keep a resident pool.

- 2026-10-10: Actual installed six-thread pool creation/free measured 0.473 ms
  median (50 warm samples); deployed fresh short text took 62–129 ms. No resident
  pool change was made: this cost offers little improvement on that workload.
  Changed course to decoder copies and per-byte work. The retained decoder now
  borrows the parsed string/data-URL suffix and writes full quartets directly,
  leaving the final quartet's padding handling intact. An isolated comparison
  exercised 100,000 valid/malformed boundary samples with identical acceptance
  and decoded bytes; component timings were variable, so no component speedup
  is claimed. Actual malformed HTTP requests still return 400.
- The same fresh 12,960,131-byte PPM upload measured 595.498 -> 548.148 ms and
  436.080 -> 415.943 ms in reverse order (six measured warm calls per pass),
  with matching 64 MiB caches, unique request keys, native Linux clients and
  JSON construction outside the timer. All vectors were identical. The CUDA
  primary is deployed with its existing flags/persistence; all five fresh
  modalities exactly matched the candidate, and a later upload pass measured
  363.593 ms, not a further paired speedup. Native Windows and CUDA builds passed.
  The Windows package was staged into a new disposable prefix, installed using
  PowerShell 5 local download fixtures and deployed with its existing settings.
  All five fresh modality vectors matched the accepted candidate exactly;
  selected runtime DLLs load from its installed prefix. No Windows inference
  speedup is claimed. OpenAI model validation passed on both primaries.
  This implementation still needs its own CI run; hardware, cross-request
  batching, quiet llama.cpp comparisons and the full release matrix remain open.

- 2026-10-10: Decoder commit 72a784e passed all seven CI jobs (37964136764).
  Measuring the actual JSON parse/assembly path found about 101 ms
  parsing and 3.3 ms copying/wrapping a 13 MB input. A pinned simdjson 5.0.3
  adapter, converting into the existing JSON representation and borrowing the
  input collection, measured 6–7 ms parsing/materialization. It statically links
  the parser with runtime CPU dispatch and scalar fallback, retaining the MIT
  dependency notice in installed packages. Original parsing remains the fallback
  for limits/unusual numbers, preserving acceptance and error behavior. No
  inference kernels or numeric cache identities change. All 30,019 isolated
  boundary samples matched the original parser's acceptance and materialized
  JSON under actual `haswell` and forced `fallback` implementations.
- Paired fresh native Linux uploads with matching caches and unique keys measured
  397.122 -> 305.099 ms and 405.208 -> 299.434 ms in reverse order (six warm calls
  per pass), with identical vectors. The CUDA primary is deployed with its
  existing flags and persistence. All five fresh modalities exactly matched
  the candidate, and arrays, Unicode, escaped base64, dimensions and usage were
  identical to the old primary. Later shared-host timings varied substantially;
  a comparison of the two updated services was 326/319 and 405/351 ms. No general
  throughput or extra deployed-pass speedup is claimed.
- Native Windows and CUDA builds passed. The Windows executable/runtime was
  staged and installed through PowerShell 5 local download fixtures, including
  the new parser notice. It loads outside the build tree. The installed candidate
  matched the old primary exactly across all five fresh modalities, despite a
  100%-busy host during part of that run. The Windows primary is now updated
  with its existing six-thread, batching-off and persistence settings; all five
  fresh deployed modalities, JSON safety and OpenAI model checks passed. Runtime
  DLLs load from the installed prefix and its parser notice matches the source. The
  focused JSON safety check protects Unicode, duplicate fields, large integers,
  dimensions/order and malformed-input rejection, and is included in native CI.
  This parser implementation needs its own CI run. Hardware, cross-request
  batching, quiet llama.cpp comparisons and the full release matrix remain open.

- 2026-10-10: Actual full-body hashing of a 12,960,131-byte request measured
  12.45/12.50 ms per FNV pass versus 1.37/1.38 ms with the already-vendored
  xxHash. Fresh media requests perform three indexing passes. The retained
  EmbeddingGemma 2 path uses xxHash for admission and response-cache lookup,
  preserving full-key/API equality, bounded admission and the original FNV
  snapshot checksum. Legacy 300M lookup retains its original hash; its response
  cache check passed without the new header dependency. No inference kernels,
  numeric cache identities or persistence format change.
- Matching cache settings, unique keys and native Linux fresh connections
  measured the same upload at 292.196 -> 260.758 ms and 292.827 -> 253.065 ms
  in reverse order (six warm calls per pass), with identical embeddings. Cached
  upload comparisons were inconsistent (58.3/47.2 then 27.5/48.3 ms), so no
  cache-hit gain is claimed. Native Windows and CUDA builds, admission safety,
  persistence corruption/budget/identity checks and JSON safety passed. All five
  fresh candidate vectors exactly matched the old services.
- Both primaries are deployed with their existing settings. Each loaded a
  snapshot written by the old service and returned an exact known response
  without a new inference log entry, exercising checksum compatibility and
  rebuilding the in-memory index. Fresh deployed CUDA vectors exactly matched
  its candidate; the installed Windows primary completed all five modalities
  and JSON/OpenAI checks. Windows runtime DLLs load from the installed prefix.
  The Windows executable/runtime was staged and installed with PowerShell 5
  local download fixtures and loads outside the build tree. No Windows inference
  speedup or general throughput gain is claimed. This change needs its own CI;
  parser commit 91b5fc9 still has two native jobs running as of the last check.
  Hardware, cross-request batching, quiet comparisons and the complete release
  matrix remain unfinished.
