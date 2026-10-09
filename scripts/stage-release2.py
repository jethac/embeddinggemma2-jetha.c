"""Stage a stripped CMake installation as a raw executable and runtime archive."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile


def stage(build, prefix, dist, target, backend):
    supported = {'windows-x86_64-cpu', 'darwin-arm64-cpu', 'darwin-arm64-metal',
                 'linux-x86_64-cpu', 'linux-x86_64-cuda', 'linux-x86_64-rocm',
                 'linux-x86_64-xpu', 'linux-arm64-cpu', 'linux-arm64-cuda'}
    if f'{target}-{backend}' not in supported:
        raise ValueError('unsupported release platform/backend combination')
    suffix = '.exe' if target.startswith('windows-') else ''
    asset = f'embeddinggemma2-jetha-{target}-{backend}'
    output = dist / (asset + suffix)
    archive = dist / (asset + ('.runtime.zip' if suffix else '.runtime.tar.gz'))
    if output.exists() or archive.exists():
        raise ValueError('release assets already exist; choose an empty staging directory')
    cache = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if '=' in line and ':' in line.split('=', 1)[0]:
            key, value = line.split('=', 1)
            cache[key.split(':', 1)[0]] = value
    if cache.get('CMAKE_BUILD_TYPE') != 'Release' or cache.get('GGML_NATIVE') != 'OFF':
        raise ValueError('release staging requires Release and GGML_NATIVE=OFF')
    if target.endswith('x86_64') and cache.get('EI_CPU_DISPATCH') != 'ON':
        raise ValueError('x86_64 releases require portable runtime CPU dispatch')
    enabled = {name for name, option in [('cuda', 'GGML_CUDA'), ('metal', 'GGML_METAL'),
               ('rocm', 'GGML_HIP'), ('xpu', 'GGML_SYCL')] if cache.get(option) == 'ON'}
    if enabled != (set() if backend == 'cpu' else {backend}):
        raise ValueError(f'{backend} asset disagrees with compiled accelerators: {enabled}')
    if backend == 'metal' and cache.get('GGML_METAL_EMBED_LIBRARY') != 'ON':
        raise ValueError('Metal releases require the embedded GGML shader library')
    binary = prefix / 'bin' / ('embeddinggemma2-jetha' + suffix)
    subprocess.run([sys.executable, str(Path(__file__).with_name('check-binary-platform.py')),
                    target, str(binary)], check=True)
    root = prefix / 'share/licenses/embeddinggemma2-jetha'
    for name in ('LICENSE', 'llama.cpp-LICENSE', 'LICENSE-nlohmann.txt', 'simdjson-LICENSE-MIT',
                 'hash/xxhash/LICENSE', 'hash/sha256/LICENSE', 'hash/rotate-bits/LICENSE.md'):
        if not (root / name).is_file():
            raise ValueError(f'missing distribution license: {name}')
    files = []
    for directory in ('bin', 'lib', 'share/licenses'):
        for path in (prefix / directory).rglob('*'):
            if not path.is_file() or path == binary:
                continue
            relative = path.relative_to(prefix)
            if directory != 'share/licenses' and not re.search(r'(\.dll|\.dylib|\.so(?:\.\d+)*)$', path.name):
                continue
            if not path.resolve().is_relative_to(prefix.resolve()):
                raise ValueError(f'runtime symlink escapes installation: {relative}')
            if suffix and path.is_symlink():
                raise ValueError(f'Windows runtime must contain regular files: {relative}')
            files.append(path)
    files.sort()
    if not files:
        raise ValueError('empty runtime payload')
    strip = cache.get('CMAKE_STRIP')
    if not strip or not Path(strip).is_file():
        raise ValueError('CMake strip tool is unavailable')
    native_files = [binary] + [p for p in files if 'share' not in p.relative_to(prefix).parts and not p.is_symlink()]
    for path in native_files:
        subprocess.run([strip, '-x' if target.startswith('darwin-') else '--strip-unneeded', str(path)], check=True)
        if target.startswith('darwin-'):
            subprocess.run(['codesign', '--force', '--sign', '-', str(path)], check=True)
            subprocess.run(['codesign', '--verify', '--verbose=2', str(path)], check=True)
    dist.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, output)
    if suffix:
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
            for path in files:
                package.write(path, path.relative_to(prefix).as_posix())
    else:
        with tarfile.open(archive, 'w:gz') as package:
            for path in files:
                package.add(path, arcname=path.relative_to(prefix).as_posix(), recursive=False)
    print(f'Staged {output.name} and {archive.name}: {len(files)} runtime/license files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True, help='CMake --install prefix')
    parser.add_argument('--dist', type=Path, required=True)
    parser.add_argument('--target', choices=['windows-x86_64', 'linux-x86_64', 'linux-arm64', 'darwin-arm64'], required=True)
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'metal', 'rocm', 'xpu'], required=True)
    args = parser.parse_args()
    try:
        stage(args.build.resolve(), args.prefix.resolve(), args.dist.resolve(), args.target, args.backend)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + '\n')
