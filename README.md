# embeddinggemma2-jetha.c

EmbeddingGemma 2 server in C.

- Inputs: text, image, audio, video, mixed.
- Output: 128, 256, 512, or 768 dimensions. Limit: 8192 tokens.
- Tested: Windows CPU, WSL CPU/CUDA, Linux ARM CPU, macOS CI CPU/Metal.
- Unverified: ROCm, XPU, GB10, Strix Halo. No NPU support or binary releases.
- Some accelerator operations use CPU fallback.

## Use

[Build and start the server](CONTRIBUTING.md#development-setup).

```sh
python examples/embed.py --text "task: search result | query: what powers the cell"
python examples/embed.py --image picture.jpg --audio recording.wav --video clip.mp4 --dimensions 256
```

Set the server address with `--url http://HOST:PORT`. API: `/api/embed` or `/v1/embeddings`. [Options](examples/embed.py).

## Performance versus llama.cpp

Baseline: llama.cpp [de7fa0a](https://github.com/ggml-org/llama.cpp/commit/de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b). Same weights, GGML revision, threads, and inputs. Q8_0; 768 dimensions; caches off; both orders; cosine >=0.999. Ratio >1 means faster.

- [CPU](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38035288012): EPYC 9V74; two threads; other CPU load 0.5–1.6%. Both engines: `EI_CPU_AUDIO_F16_2=1`.
- [ARM CPU](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38036295848): Linux ARM64 CI; two threads; other CPU load 0%. Both engines: `EI_CPU_AUDIO_F16_2=1`, `EI_ARM_FP16_ACC_F32=1`.
- [Metal](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38020441663): virtual M1; three threads; CPU vision attention; other CPU load 1.46–97.42%.

| Backend | Input | Clients | This server, emb/s | llama.cpp, emb/s | Ratio |
|---|---|---:|---:|---:|---:|
| CPU | Text, 18 tokens | 1 | 23.737 | 24.719 | 0.960x |
| CPU | Text, 18 tokens | 4 | 28.309 | 28.745 | 0.985x |
| CPU | Image | 1 | 0.1491 | 0.1484 | 1.005x |
| CPU | Image | 4 | 0.1456 | 0.1450 | 1.004x |
| CPU | Audio, 1 second | 1 | 5.207 | 5.136 | 1.014x |
| CPU | Audio, 1 second | 4 | 5.093 | 5.509 | 0.924x |
| CPU | Video, 2 seconds | 1 | 0.1908 | 0.1859 | 1.026x |
| CPU | Video, 2 seconds | 4 | 0.1899 | 0.1871 | 1.015x |
| CPU | Mixed | 1 | 0.1525 | 0.1522 | 1.002x |
| CPU | Mixed | 4 | 0.1498 | 0.1497 | 1.001x |
| ARM CPU | Text, 18 tokens | 1 | 7.025 | 8.956 | 0.784x |
| ARM CPU | Text, 18 tokens | 4 | 7.062 | 9.254 | 0.763x |
| ARM CPU | Image | 1 | 0.05963 | 0.06177 | 0.965x |
| ARM CPU | Image | 4 | 0.05964 | 0.06146 | 0.970x |
| ARM CPU | Audio, 1 second | 1 | 1.543 | 1.669 | 0.925x |
| ARM CPU | Audio, 1 second | 4 | 1.530 | 1.679 | 0.911x |
| ARM CPU | Video, 2 seconds | 1 | 0.07375 | 0.07630 | 0.967x |
| ARM CPU | Video, 2 seconds | 4 | 0.07343 | 0.07632 | 0.962x |
| ARM CPU | Mixed | 1 | 0.05790 | 0.05974 | 0.969x |
| ARM CPU | Mixed | 4 | 0.05794 | 0.05983 | 0.968x |
| Metal | Image | 1 | 0.1302 | 0.1097 | 1.187x |
| Metal | Image | 4 | 0.1915 | 0.1752 | 1.093x |

x86 audio C1 ratio by order: 0.964–1.065x.
ARM warm four-client audio: [comparison failed at cosine 0.998954](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38037459764/job/114170787198). This server matched serial output; llama.cpp changed.
Metal uses `EI_METAL_MEDIA_FLASH_ATTN2=1`. Four-client ratio by order: 0.996–1.225x. Default AUTO: [0.801x across all inputs](perf/optimization_status.md#complete-virtual-metal-comparison).
CUDA: no result below the host CPU load limit.

[Measurements](perf/optimization_status.md#embeddinggemma-2-development-results) · [Text benchmark](perf/compare_llamacpp2.py) · [Media benchmark](perf/compare_media_llamacpp2.py)

## License

[MIT](LICENSE). Based on [QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c) and [Windows host code](https://github.com/jethac/embeddinggemma.c). Model weights have a separate license.
This software is based in part on the work of the Independent JPEG Group.

[Scope](GOAL.md) · [Release process](RELEASE.md) · [Upstream README](UPSTREAM_README.md)
