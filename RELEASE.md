# Release Runbook

This runbook distributes EmbeddingGemma 2 from
`jethac/embeddinggemma2-jetha.c`. No binary release has been published yet.
The inherited Makefile `release-*` targets and `scripts/stage-release.sh`
build the legacy 300M server; use the CMake workflow below for this port.

## Release contract

- Obtain explicit approval for the exact version before editing `VERSION`,
  creating a tag, or publishing a release.
- Build the entire matrix from the same clean release commit. Use native build
  hosts and verify the final installed payload on the corresponding hardware.
- Retain MIT and dependency notices. Keep model weights, GGUFs, dependency source,
  headers, static archives, debug files and build intermediates out of assets.
- Keep upstream's raw executable naming and `SHA256SUMS` convention with this
  project's prefix. Each executable also needs a matching runtime archive:
  portable CPU dispatch, media libraries and Windows compiler runtime DLLs
  cannot be served by the raw executable alone.
- Strip executables and runtime libraries. Ad-hoc sign and verify Darwin files;
  embed the Metal shader library. Review loader dependencies and minimum OS/ABI
  requirements on the actual intended hosts before publishing.
- Publish all 18 binary/runtime assets and `SHA256SUMS`; do not substitute
  partial, mixed-commit or unqualified accelerator builds.

## Asset matrix

| Target | Backends | Runtime suffix |
|---|---|---|
| Darwin ARM64 | cpu, metal | `.runtime.tar.gz` |
| Linux ARM64 | cpu, cuda (GB10) | `.runtime.tar.gz` |
| Linux x86_64 | cpu, cuda, rocm, xpu | `.runtime.tar.gz` |
| Windows x86_64 | cpu | `.runtime.zip` |

Raw executables are named `embeddinggemma2-jetha-TARGET-BACKEND`, with `.exe`
on Windows. Runtime archives have the same stem. For example:

```text
embeddinggemma2-jetha-linux-x86_64-cpu
embeddinggemma2-jetha-linux-x86_64-cpu.runtime.tar.gz
embeddinggemma2-jetha-windows-x86_64-cpu.exe
embeddinggemma2-jetha-windows-x86_64-cpu.runtime.zip
SHA256SUMS
```

Archives contain application-relative `bin/`, `lib/` and `share/licenses/`
files. They exclude the main executable, which remains a separate raw asset.
System accelerator drivers/toolkits remain host prerequisites; packaging does
not establish support for hardware that has not run the model.

## Build and stage

After exact version approval, update the version and relevant README examples,
commit and push. Create a detached worktree from that clean commit for each
native host. Never stage development output as a published release.

For Linux x86_64 CPU (use fresh build, install and distribution directories):

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DEI_CPU_DISPATCH=ON \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF -DGGML_METAL=OFF
cmake --build build-release --target embeddinggemma2-jetha -j 4
cmake --install build-release --prefix "$PWD/release-install" --strip
python3 scripts/stage-release2.py --build build-release \
  --prefix release-install --dist dist --target linux-x86_64 --backend cpu
```

Use `--target windows-x86_64`, `linux-arm64` or `darwin-arm64` on the matching
native host. Windows uses the project's MinGW CMake instructions in the README.
ARM64 uses `-DEI_CPU_DISPATCH=OFF -DGGML_NATIVE=OFF`. Select exactly the intended
accelerator with `GGML_CUDA`, `GGML_METAL`, `GGML_HIP` or `GGML_SYCL`; disable the
others. Metal also requires `-DGGML_METAL_EMBED_LIBRARY=ON`. CUDA/ROCm/SYCL
architecture and toolchain settings must cover the hardware advertised in the
release. GB10, Strix Halo, native Metal, ROCm and XPU qualification is still
unfinished; these commands do not make those release targets ready.

The stager strips the disposable installation prefix in place, validates the
executable platform and required notices, then writes the raw executable and
runtime archive. Never point it at a deployed application prefix.

## Try the installed journey

Before publication, run the installers against locally staged real assets
(the fixture replaces only network downloads):

```sh
python3 tests/installer_runtime2.py --assets dist
```

On Windows:

```powershell
./tests/installer_runtime2.ps1 -AssetsDir ./dist
```

These check loading from a path with spaces and preservation of a working
installation after checksum failure. They do not qualify inference or a GPU.
Extract each actual payload into a fresh prefix and run
`tests/installed_service2.py --prefix PREFIX` without build-tree library paths.
Start that executable with the backbone and both media encoders, disable caches,
and run `examples/journey2.py --url URL` for text, image, audio, video and mixed
requests. Compare against the reference and the accepted CPU implementation;
check context limits and numerical quality for that backend. Review the actual
loaded modules, not just the archive listing. Video/WebP require FFmpeg on PATH.
Run the relevant existing safety/regression checks and required CI checks.

## Assemble and publish

Collect all nine executable/runtime pairs from the same commit into one fresh
`dist` directory. Generate and verify the full distribution:

```sh
sh scripts/release-assets.sh checksums dist
sh scripts/release-assets.sh verify dist
```

The verifier rejects missing assets, wrong executable platforms, invalid
archives, duplicate checksum entries and checksum mismatches. It cannot prove
native hardware qualification or a common source commit; complete those above.
Do not use legacy `make release-ready` to qualify this port.

Only after exact version approval and all qualification is complete:

```sh
version=$(cat VERSION)
git tag -a "$version" -m "embeddinggemma2-jetha.c $version"
git push origin "$version"
gh release create "$version" dist/embeddinggemma2-jetha-* dist/SHA256SUMS \
  --repo jethac/embeddinggemma2-jetha.c \
  --title "embeddinggemma2-jetha.c $version" --notes-file release-notes.md
```

Release notes describe tested hardware, runtime requirements, measured
performance and limitations. Verify the published asset list, then try a pinned
GitHub installation in a fresh directory on every released platform:

```sh
./install.sh --version "$version" --variant cpu --install-dir /tmp/gemma2-install
/tmp/gemma2-install/embeddinggemma2-jetha --help
```

```powershell
./install.ps1 -Version APPROVED_TAG -InstallDir "$env:TEMP/gemma2-install"
& "$env:TEMP/gemma2-install/embeddinggemma2-jetha.cmd" --help
```

Repeat the full inference journey through the installed application. Exercise
explicit accelerator selection, automatic selection, and Linux CPU fallback
when accelerator libraries are unavailable. A failed/incomplete publication is
not ready to announce. Never move an existing release tag to another commit.
