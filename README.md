# embeddinggemma2-jetha.c

An experimental native EmbeddingGemma 2 port, intended to produce useful code
and evidence for the maintainers of [embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c).
The distinct name leaves the natural successor name available to upstream.

The scope covers text, images, audio, video, and mixed inputs, with CPU SIMD
(including AVX2 and AVX-512), accelerator execution, batching, and caching.
See [GOAL.md](GOAL.md) for the full scope, implementation direction,
performance requirements, acceptance criteria, and relationship to upstream.

**Current status:** the original EmbeddingGemma 300M server and native Windows
CPU foundation are present. EmbeddingGemma 2 inference has not been implemented
or validated yet. There are no EmbeddingGemma 2 benchmark claims or releases.

## Foundation and attribution

Based on [QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c),
with Windows host-layer changes from
[jethac/embeddinggemma.c](https://github.com/jethac/embeddinggemma.c).
The inherited source is MIT licensed; see [LICENSE](LICENSE).

[UPSTREAM_README.md](UPSTREAM_README.md) preserves the original documentation
and benchmark claims for the original 300M model. They do not establish
EmbeddingGemma 2 results for this project.

The inherited build and test instructions in [CONTRIBUTING.md](CONTRIBUTING.md)
currently describe the 300M implementation. Model weights, local reference
checkouts, and generated artifacts are excluded from Git.
