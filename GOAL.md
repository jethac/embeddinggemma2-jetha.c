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
- Publish raw executables with stable asset names and `SHA256SUMS`. Keep model
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
  remain unfinished. The dev build requires its shared libraries and plugins;
  adapting the inherited executable installation/release flow to acquire these
  matching dependencies remains unfinished.
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

Keep this file current as implementation decisions and verified evidence change.
