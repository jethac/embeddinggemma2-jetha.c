# embeddinggemma2-jetha.c

Build a fast, native, model-specialized EmbeddingGemma 2 inference server,
starting from QuixiAI's MIT-licensed embeddinggemma.c and the native Windows
host-layer contribution. Public home: `jethac/embeddinggemma2-jetha.c`.
The distinct name avoids taking the natural upstream successor name.

## Relationship to upstream

This project is an experimental port intended to get ahead of upstream's
EmbeddingGemma 2 work and produce useful code and evidence they can adopt.
It should make that work easier, not create a competing claim to the upstream
project name or impose maintenance work on its maintainers.

- Preserve history, licenses, attribution, and a clear upstream remote.
- Keep architecture, host portability, API, and optimization changes separable
  and reviewable so upstream can cherry-pick what helps.
- Record reference provenance, correctness results, benchmark methodology, and
  rejected experiments so maintainers do not have to repeat the investigation.
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
  extensions such as VNNI/BF16 require separately qualified hardware.
- Use efficient quantized kernels, persistent workspaces, fused operations
  where beneficial, and efficient matrix/attention execution on accelerators.
- Preserve and extend dynamic batching, bounded queues, duplicate singleflight,
  exact-result caching, and useful concurrency behavior to multimodal requests.
- Measure tokenization, media decoding/preprocessing, each encoder, the text
  backbone, pooling/projection, batching, and HTTP serialization separately as
  well as end to end. Record latency, throughput, memory use, and correctness.

## Implementation direction

The current direction is a model-specific C graph using a pinned GGML kernel
dependency and Gemma 4 media encoder support through libmtmd. This is an
implementation choice to validate, not a completed port or a proven speedup.
Retain upstream attribution and record dependency revisions and licenses.
Prefer specialization and reuse of validated optimized kernels over a Python
inference subprocess. Revise the approach when measurement justifies it.

## Release process

Follow upstream's [RELEASE.md](RELEASE.md) conventions where applicable, adapting
the repository, project/asset names, model, dependencies, and platform matrix.
Do not invent a separate distribution workflow without a concrete need.

- Build the complete supported release matrix from the same clean commit.
- Build and validate on native platforms and matching accelerator hardware;
  qualify the final staged executables as well as development builds.
- Publish raw executables with stable asset names and `SHA256SUMS`. Keep model
  weights and intermediate files out of releases. Preserve checksum-verified
  installation and appropriate backend detection/fallback.
- Strip binaries, sign and verify Darwin artifacts, and retain portable OS/ABI
  baselines. CPU release binaries must dispatch safely across supported ISAs;
  host-specific build flags are for local experiments.
- Carry over accelerator code-object/runtime checks and include native Windows
  in the qualified release matrix. Include dependency provenance and licenses
  if the new encoder/kernel dependencies require additional notices or files.
- Use this repository's own versions, tags, release destination, asset prefix,
  and cache/install identity. Adapt inherited release scripts before using them
  for publication; they currently target the original project.
- Obtain approval for an exact release version before changing release versions,
  tagging, or publishing, as required by the inherited release runbook. Creating
  this public development repository does not publish a binary release.

## Validation and acceptance

- Pin model files and reference implementation revisions. Compare actual
  outputs against the reference for every modality and meaningful mixtures.
- Validate tokenizer IDs, prompt prefixes, input scaling, media preprocessing,
  attention masks and positions, output projection, pooling, and normalization.
- Include short and long contexts, local-attention boundaries, batch isolation,
  dimension truncation, malformed inputs, non-finite outputs, and context limits.
- Verify batch versus individual parity and cache identity for media content,
  preprocessing settings, prompts, model revision, and output dimensions.
- Test public HTTP behavior, health/readiness, lifecycle, queue limits, and
  graceful errors. Bound media sizes and decoded allocation sizes.
- Run checks on supported native platforms and applicable hardware. Mark
  untested combinations explicitly; compilation alone is not numeric validation.
- Compare performance with the same model/quantization, exact inputs and token
  counts, dimensions, cache settings, warmup, concurrency, and hardware. Retain
  raw measurements and commands. Make no speed claim without evidence.
- Provide reproducible build/run instructions, model acquisition, API examples
  for every modality, and verified executable artifacts when ready.

This goal is complete only when full multimodal serving works, correctness is
verified, and the supported performance paths have reproducible evidence.

## Starting state (2026-10-09)

- Upstream base: QuixiAI/embeddinggemma.c at
  `55964e25b199ddf7a9707814ec490324f749248f`.
- Windows foundation: jethac/embeddinggemma.c at
  `2cba339e028dede34c3219dac895ef039c880b64`, incorporated locally.
- EmbeddingGemma 2 reference architecture and the existing media encoder APIs
  have been inspected. No EmbeddingGemma 2 inference implementation or
  performance result has been validated yet.
- The inherited README and other upstream documentation describe the original
  300M model until rewritten; they are not claims of EmbeddingGemma 2 support.
- Local reference sources, tool environments, models, and build outputs remain
  ignored and must not be committed.

Keep this file current as implementation decisions and verified evidence change.
