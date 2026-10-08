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
- Portable CPU ISA dispatch,
  complete media resource handling, and multimodal queue/batching/singleflight are
  unfinished. Remote hardware access remains unresolved; NPU support is absent.
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
- README describes the current dev commands. A new Linux CPU CI job builds
  the CMake implementation and runs the image regression on its service;
  its first remote run passed. The inherited CI matrix validates
  the legacy 300M foundation only. Makefile and release adaptation is unfinished.
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
- Initial fixed-shape ONNX exports of the backbone, vision, and audio encoders
  run on CPU ONNX Runtime. Their assembled 294-token text/image/audio sample
  matches the FP32 reference at cosine 0.99999994 (maximum element error
  1.31e-7). Export avoids the reference's untraceable mask construction and
  audio KV-window unfold while preserving its exact boolean attention mask.
  These are FP32 graphs, totaling about 2.98 GB; BF16 NPU compilation, operator
  placement, and execution remain untested because Strix Halo access is unresolved.
  Successful CPU execution is not evidence of NPU compatibility.
- Local reference sources, tool environments, models, and build outputs remain
  ignored and must not be committed.

Keep this file current as implementation decisions and verified evidence change.
