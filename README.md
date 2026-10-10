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

Baseline: llama.cpp [de7fa0a](https://github.com/ggml-org/llama.cpp/commit/de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b). Same weights, kernels, threads, and inputs. Q8_0; 768 dimensions; caches off; both orders; cosine >=0.999. Ratio >1 means faster.

- [CPU](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/37892946034): EPYC 9V74; two threads; other CPU load 0.8–0.9%.
- [ARM CPU](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38036295848/job/114167311248): Linux ARM64 CI; two threads; other CPU load 0%. Both engines: `EI_CPU_AUDIO_F16_2=1`, `EI_ARM_FP16_ACC_F32=1`.
- [Metal](https://github.com/jethac/embeddinggemma2-jetha.c/actions/runs/38020441663): virtual M1; three threads; CPU vision attention; other CPU load 1.46–97.42%.

| Backend | Input | Clients | This server, emb/s | llama.cpp, emb/s | Ratio |
|---|---|---:|---:|---:|---:|
| CPU | Text, 32 tokens | 1 | 16.75 | 16.81 | 0.996x |
| CPU | Text, 32 tokens | 4 | 18.31 | 18.16 | 1.008x |
| CPU | Text, 256 tokens | 1 | 2.26 | 2.25 | 1.006x |
| CPU | Text, 256 tokens | 4 | 2.26 | 2.24 | 1.008x |
| CPU | Text, 1024 tokens | 1 | 0.44 | 0.44 | 1.008x |
| CPU | Text, 1024 tokens | 4 | 0.41 | 0.41 | 0.994x |
| ARM CPU | Text, 18 tokens | 1 | 7.025 | 8.956 | 0.784x |
| ARM CPU | Text, 18 tokens | 4 | 7.062 | 9.254 | 0.763x |
| ARM CPU | Audio, 1 second | 1 | 1.543 | 1.669 | 0.925x |
| ARM CPU | Audio, 1 second | 4 | 1.530 | 1.679 | 0.911x |
| Metal | Image | 1 | 0.1302 | 0.1097 | 1.187x |
| Metal | Image | 4 | 0.1915 | 0.1752 | 1.093x |

Metal uses `EI_METAL_MEDIA_FLASH_ATTN2=1`. Four-client ratio by order: 0.996–1.225x. Default AUTO: [0.801x across all inputs](perf/optimization_status.md#complete-virtual-metal-comparison).
CUDA: no result below the host CPU load limit.

[Measurements](perf/optimization_status.md#embeddinggemma-2-development-results) · [Text benchmark](perf/compare_llamacpp2.py) · [Media benchmark](perf/compare_media_llamacpp2.py)

## License

[MIT](LICENSE). Based on [QuixiAI/embeddinggemma.c](https://github.com/QuixiAI/embeddinggemma.c) and [Windows host code](https://github.com/jethac/embeddinggemma.c). Model weights have a separate license.
This software is based in part on the work of the Independent JPEG Group.

[Scope](GOAL.md) · [Release process](RELEASE.md) · [Upstream README](UPSTREAM_README.md)
