"""One manual Ubuntu22 reproduction of the observed no-SYCL-GPU loader bug.

Private Intel extraction only; this does not qualify an Intel GPU or benchmark.
"""
import concurrent.futures
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'xpu-experiment'
SDK = WORK / 'sdk/opt/intel/oneapi'
BUILD = WORK / 'build'
PREFIX = WORK / 'installed'
LOGS = WORK / 'logs'
# Exact official package payloads used in the local Intel2026 compile proof.
PACKAGES = [
    ('pool/main/intel-oneapi-dpcpp-cpp-2026.1-2026.1.1-325_amd64.deb', 157991970, 'fa97aac3f6ef43b0966f736dc47ebd4c97bddc709c80cba0fec0e1a1f36f9662'),
    ('pool/main/intel-oneapi-libdpstd-devel-2022.13-2022.13.0-107_amd64.deb', 350052, 'ecf6f345e4fdef992d9127ccc9e7708f36f613398f95f075565e94bf53028835'),
    ('pool/main/intel-oneapi-common-vars-2026.0.0-235_all.deb', 12694, 'f04ac260f028dccb67c6d7a87e660b42247c7464fa4126ee8c07359ce527de80'),
    ('pool/main/intel-oneapi-common-licensing-2026.0-2026.0.0-235_all.deb', 30702, '6f85b5c530236545cce55fdf8eef38249c03a3e529252208d60c4627417fac85'),
    ('pool/main/intel-oneapi-common-oneapi-vars-2026.0-2026.0.0-235_all.deb', 10630, '0b4a4c82f3517004d7608a743c67689e71dee1c0706f73fb7f74c78aa6999013'),
    ('pool/main/intel-oneapi-compiler-dpcpp-cpp-common-2026.1-2026.1.1-325_all.deb', 1715704, 'd26bef9048fd2f9ff9cc8a0443f41766ef3ff0682d6847f3d60f496f3fdd4d26'),
    ('pool/main/intel-oneapi-compiler-dpcpp-cpp-runtime-2026.1-2026.1.1-325_amd64.deb', 84983254, 'df0413d57cff5a9c67bd02c1e958d7e0512f77878bbcf976c9e08af7c0b1069c'),
    ('pool/main/intel-oneapi-compiler-shared-2026.1-2026.1.1-325_amd64.deb', 14466662, '87039478855d1b7ce08c5a30456e360158cdf45a6fb6a791083651ec340c5e99'),
    ('pool/main/intel-oneapi-tbb-devel-2023.1-2023.1.0-151_amd64.deb', 1590760, 'f56474678ecf588da4a3ac5342bcffdfd001dfad78bb432a5eb19765f7f6e512'),
    ('pool/main/intel-oneapi-mkl-sycl-include-2026.1-2026.1.0-236_amd64.deb', 130664, '3dc8777afea889fe2e9fa09352c998771adf23a99ca9fbb3e7e1c93a00c5cf4d'),
    ('pool/main/intel-oneapi-mkl-core-devel-2026.1-2026.1.0-236_amd64.deb', 122373106, 'f748e21ebb3396670085da04375511ade4a372943838ddbf4f770ef65bcb7c43'),
    ('pool/main/intel-oneapi-mkl-classic-include-2026.1-2026.1.0-236_amd64.deb', 659080, '509a3737d8f38339dffa35c26ce8c01857787d0fcadb3daacb188efc2caa3483'),
    ('pool/main/intel-oneapi-compiler-shared-runtime-2026.1-2026.1.1-325_amd64.deb', 96698830, 'c0c77700243af0413f7fbefd26649943c4959171ead91b6ace36ef0c8dfb2630'),
    ('pool/main/intel-oneapi-tbb-2023.1-2023.1.0-151_amd64.deb', 1443780, 'f6e4ae469436a8c5ee2250203268344164c9ff455a2207219c75c463aab99a8b'),
    ('pool/main/intel-oneapi-umf-1.1-1.1.0-340_amd64.deb', 130996, '8705a997042c76f67228b4135184754e1cd3f8f68b6629bcb261c1d5b958ce78'),
    ('pool/main/intel-oneapi-compiler-shared-common-2026.1-2026.1.1-325_all.deb', 71241522, '283b675d295c9da38902edbc936980cde6b85d835480269f87559e5c59236591'),
    ('pool/main/intel-oneapi-mkl-sycl-blas-2026.1-2026.1.0-236_amd64.deb', 11639462, '9668d51a7cf68ae4b8944a8cdaa24f6d4ef72604c048f68fd8603b2b792bd783'),
    ('pool/main/intel-oneapi-mkl-core-2026.1-2026.1.0-236_amd64.deb', 126239806, '3b230c02fc0ac44f8f71062c20cd883b37c6c1e00940dc0a7f13b02d9c47bd93'),
    ('pool/main/intel-oneapi-openmp-2026.1-2026.1.1-325_amd64.deb', 69930442, '89fdf95c72613ae1a7fb0834c7b8ccd4c2460fc68e37643b2bbe44c37678bd02'),
    ('pool/main/intel-oneapi-tcm-1.5-1.5.0-489_amd64.deb', 677272, '7122ed7748d54342b24f0a2838a64f24b54e430ac4d06b2961a8eefee4ffb089'),
    ('pool/main/intel-oneapi-openmp-common-2026.1-2026.1.1-325_all.deb', 16958, '431b70bf1fb9ce58b392eb85110506c78c536f80f4d995c5080b1011e063b810'),
]


def run(command, **kwargs):
    print('RUN', ' '.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def extract(record):
    filename, size, sha256 = record
    target = WORK / 'downloads' / Path(filename).name
    with urllib.request.urlopen('https://apt.repos.intel.com/oneapi/' + filename,
                                timeout=120) as source, target.open('wb') as output:
        shutil.copyfileobj(source, output, 1024 * 1024)
    digest = hashlib.sha256()
    with target.open('rb') as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    assert target.stat().st_size == size and digest.hexdigest() == sha256, filename
    return target


def elf(path):
    with path.open('rb') as source:
        return source.read(4) == b'\x7fELF'


def sycl_plugin(prefix):
    paths = list(prefix.rglob('libggml-sycl.so'))
    assert len(paths) == 1, ('expected one installed SYCL plugin', paths)
    return paths[0]


def close_runtime():
    # The existing private staging recipe: copy only SDK runtime dependencies
    # plus MKL's dynamically opened ISA/VML siblings, retaining their notices.
    lib = PREFIX / 'lib'
    lib.mkdir(exist_ok=True)
    available = {}
    for path in SDK.rglob('*'):
        if path.is_file() and '.so' in path.name:
            available.setdefault(path.name, path)
    standard = {'libc.so.6', 'libm.so.6', 'libpthread.so.0', 'libdl.so.2',
                'librt.so.1', 'libstdc++.so.6', 'libgcc_s.so.1', 'ld-linux-x86-64.so.2'}
    zlib = Path('/lib/x86_64-linux-gnu/libz.so.1').resolve()
    shutil.copy2(zlib, lib / zlib.name)
    (lib / 'libz.so.1').symlink_to(zlib.name)
    pending = [path for folder in ('bin', 'lib') for path in (PREFIX / folder).rglob('*')
               if path.is_file() and elf(path)]
    extra = [p for p in (SDK / 'mkl/2026.1/lib').glob('*.so*')
             if not p.is_symlink() and p.is_file() and elf(p)]
    extra += [available[n] for n in ('libur_loader.so.0', 'libur_adapter_level_zero.so.0',
                                   'libur_adapter_opencl.so.0')]
    for source in extra:
        target = lib / source.name
        if not target.exists():
            shutil.copy2(source.resolve(), target)
        pending.append(target)
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        dynamic = subprocess.check_output(['readelf', '-d', str(path)], text=True)
        for name in re.findall(r'\(NEEDED\).*\[(.*?)\]', dynamic):
            if name in standard:
                continue
            target = lib / name
            if not target.exists():
                assert name in available, ('missing SDK dependency', name)
                shutil.copy2(available[name].resolve(), target)
            pending.append(target)
    for path in seen:
        dynamic = subprocess.check_output(['readelf', '-d', str(path)], text=True)
        needed = re.findall(r'\(NEEDED\).*\[(.*?)\]', dynamic)
        if path.name in available and '$ORIGIN' not in dynamic and (
                path.name.startswith('libmkl_') or any(n not in standard for n in needed)):
            run(['patchelf', '--set-rpath', '$ORIGIN', path])
        versions = subprocess.check_output(['readelf', '--version-info', str(path)], text=True)
        for kind, limit in (('GLIBC', (2, 34)), ('GLIBCXX', (3, 4, 29))):
            values = [tuple(map(int, v.split('.'))) for v in
                      re.findall(r'\b' + kind + r'_(\d+(?:\.\d+)+)', versions)]
            assert not values or max(values) <= limit, (path, kind, values)
    notices = PREFIX / 'share/licenses/embeddinggemma2-jetha/intel-oneapi'
    shutil.copytree(SDK / 'licensing/2026.0', notices)
    for component, version in (('compiler', '2026.1'), ('mkl', '2026.1'),
                               ('tbb', '2023.1'), ('umf', '1.1'), ('tcm', '1.5')):
        docs = SDK / component / version / 'share/doc' / component
        if (docs / 'licensing').is_dir():
            shutil.copytree(docs / 'licensing', notices / component)
        else:
            (notices / component).mkdir()
            shutil.copy2(docs / 'LICENSE.TXT', notices / component / 'LICENSE.TXT')
    znotice = PREFIX / 'share/licenses/embeddinggemma2-jetha/ubuntu22-zlib'
    znotice.mkdir()
    shutil.copy2('/usr/share/doc/zlib1g/copyright', znotice / 'copyright')


def installed_journey(package, clean_env):
    sys.path.insert(0, str(ROOT / 'perf'))
    from compare_llamacpp import Endpoint, ManagedServer
    from compare_media_llamacpp2 import fixtures
    reference = WORK / 'native-reference'
    shutil.copytree(package, reference, symlinks=True)
    # Same app/CPU/math; omit just the SYCL plugin from this reference process.
    sycl_plugin(reference).unlink()
    cases_dir = WORK / 'fixtures'
    cases_dir.mkdir()
    cases = fixtures(cases_dir, 1)
    clients = {}
    for kind, bodies in cases.items():
        value = bodies[0][0]['input']
        parts = [{'type': 'text', 'text': value}] if isinstance(value, str) else value['content']
        command = [sys.executable, str(ROOT / 'examples/embed.py')]
        for index, part in enumerate(parts):
            if part['type'] == 'text':
                command += ['--text', part['text']]
            else:
                path = cases_dir / f'{kind}-{index}.media'
                path.write_bytes(base64.b64decode(part['data'].split(',', 1)[1]))
                command += ['--' + part['type'], str(path)]
        clients[kind] = command
    results = []
    for name, prefix in (('native', reference), ('xpu-fallback', package)):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        endpoint = Endpoint('127.0.0.1', port, '/v1/embeddings', 'openai')
        log = LOGS / (name + '.log')
        command = [str(prefix / 'bin/embeddinggemma2-jetha'), '--backend', 'auto',
                   '--bind', '127.0.0.1', '--port', str(port), '--model',
                   str(ROOT / 'model/embeddinggemma-2-Q8_0.gguf'), '--mmproj',
                   str(ROOT / 'model/mmproj-embeddinggemma-2-Q8_0.gguf'),
                   '--cache-entries', '0', '--response-cache-mb', '0']
        with ManagedServer(command, endpoint, '/healthz', log, env=clean_env) as server:
            actual = {}
            for kind, client in clients.items():
                output = json.loads(subprocess.check_output(client +
                    ['--url', f'http://127.0.0.1:{port}'], env=clean_env, text=True))
                vector = output['embeddings'][0]
                assert len(output['embeddings']) == 1 and len(vector) == 768
                assert all(math.isfinite(v) for v in vector)
                assert abs(math.fsum(v*v for v in vector) - 1) < 1e-5
                usage = output.get('usage')
                if kind != 'text':
                    assert isinstance(usage, dict) and usage['prompt_tokens'] > 0
                    assert usage['total_tokens'] == usage['prompt_tokens']
                actual[kind] = vector, usage
            results.append(actual)
            maps = Path(f'/proc/{server.process.pid}/maps').read_text()
            for line in maps.splitlines():
                if any(n in line for n in ('libggml', 'libllama', 'libmtmd')):
                    assert str(prefix) in line, ('non-package model runtime', line)
            assert 'CPU audio F16 active: 132 Conformer matrices' in log.read_text()
            assert 'EmbeddingGemma 2: CPU,' in log.read_text(), 'AUTO did not select native CPU'
            if name == 'xpu-fallback':
                assert 'no visible SYCL GPU devices; skipping backend' in log.read_text()
                # Registry rejects nullptr and may unload the declined plugin;
                # the actual registration marker proves its enumeration ran.
                assert sycl_plugin(prefix).is_file()
    for kind in cases:
        assert results[0][kind] == results[1][kind], ('CPU fallback drift', kind)
        print('INSTALLED_CPU_FALLBACK_BYTEEXACT', kind, flush=True)


def main():
    assert sys.platform == 'linux' and 'VERSION_ID="22.04"' in Path('/etc/os-release').read_text()
    WORK.mkdir()
    LOGS.mkdir()
    (WORK / 'downloads').mkdir()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        packages = list(pool.map(extract, PACKAGES))
    for package in packages:
        run(['dpkg-deb', '-x', package, WORK / 'sdk'])
        package.unlink()
    for component in SDK.iterdir():
        versions = [p for p in component.iterdir() if p.is_dir() and
                    re.fullmatch(r'\d+(?:\.\d+)+', p.name)] if component.is_dir() else []
        if versions and not (component / 'latest').exists():
            (component / 'latest').symlink_to(max(versions, key=lambda p:
                tuple(map(int, p.name.split('.')))).name)
    for path in (WORK / 'sdk').rglob('*'):
        if path.is_symlink():
            assert path.resolve().is_relative_to((WORK / 'sdk').resolve()), path
    env = {k: v for k, v in os.environ.items() if not k.startswith('EI_')}
    env.update(LD_LIBRARY_PATH=':'.join(str(SDK / p) for p in
        ('compiler/2026.1/lib', 'mkl/2026.1/lib', 'tbb/2023.1/lib', 'umf/1.1/lib')))
    compiler = SDK / 'compiler/2026.1/bin/icpx'
    run([compiler, '--version'], env=env)
    cmake_path = ROOT / 'CMakeLists.txt'
    final_cmake = cmake_path.read_text()
    patch_count = len(re.findall(r'"\$\{CMAKE_CURRENT_SOURCE_DIR\}/deps/[^"\n]+\.patch"', final_cmake))
    assert patch_count == 24, patch_count
    last_patch = '    "${CMAKE_CURRENT_SOURCE_DIR}/deps/ggml-sycl-no-device.patch"\n'
    assert final_cmake.count(last_patch) == 1
    cmake_path.write_text(final_cmake.replace(last_patch, ''))
    configure = ['cmake', '-S', ROOT, '-B', BUILD, '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_C_COMPILER=/usr/bin/gcc-11', '-DCMAKE_CXX_COMPILER=' + str(compiler),
        '-DCMAKE_CXX_FLAGS_INIT=--gcc-toolchain=/usr', '-DEI_CPU_DISPATCH=ON',
        '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF', '-DGGML_CUDA=OFF', '-DGGML_HIP=OFF',
        '-DGGML_METAL=OFF', '-DGGML_SYCL=ON', '-DGGML_SYCL_TARGET=INTEL',
        '-DGGML_SYCL_DNN=OFF', '-DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=OFF',
        '-DMKL_DIR=' + str(SDK / 'mkl/2026.1/lib/cmake/mkl'),
        '-DTBB_DIR=' + str(SDK / 'tbb/2023.1/lib/cmake/tbb'),
        '-DMKL_LINK=dynamic', '-DMKL_SYCL_LINK=dynamic']
    run(configure, env=env)
    run(['cmake', '--build', BUILD, '--target', 'embeddinggemma2-jetha', '-j', '2'], env=env)
    run(['cmake', '--install', BUILD, '--prefix', PREFIX, '--strip'], env=env)
    close_runtime()
    run([sys.executable, ROOT / 'scripts/download-model2.py'])
    regression = [sys.executable, ROOT / 'tests/xpu_cpu_fallback2.py', '--binary',
                  PREFIX / 'bin/embeddinggemma2-jetha', '--model',
                  ROOT / 'model/embeddinggemma-2-Q8_0.gguf']
    original = subprocess.run(list(map(str, regression)), capture_output=True, text=True)
    (LOGS / 'original23.log').write_text(original.stdout + original.stderr)
    assert original.returncode != 0 and 'No device of requested type available' in original.stderr, original
    print('ORIGINAL23_SPECIFIC_NO_DEVICE_FAIL_REPRODUCED', flush=True)
    cmake_path.write_text(final_cmake)
    run(configure, env=env)
    run(['cmake', '--build', BUILD, '--target', 'embeddinggemma2-jetha', '-j', '2'], env=env)
    run(['cmake', '--install', BUILD, '--prefix', PREFIX, '--strip'], env=env)
    # The closure/notice files already copied are untouched by CMake install.
    sections = subprocess.check_output(['readelf', '-SW', str(sycl_plugin(PREFIX))], text=True)
    assert '__CLANG_OFFLOAD_BUNDLE__sycl-spir64' in sections
    run([sys.executable, ROOT / 'scripts/stage-release2.py', '--build', BUILD,
         '--prefix', PREFIX, '--dist', WORK / 'dist', '--target', 'linux-x86_64', '--backend', 'xpu'])
    package = WORK / 'package'
    package.mkdir()
    with tarfile.open(WORK / 'dist/embeddinggemma2-jetha-linux-x86_64-xpu.runtime.tar.gz') as archive:
        archive.extractall(package)
    (package / 'bin').mkdir(exist_ok=True)
    shutil.copy2(WORK / 'dist/embeddinggemma2-jetha-linux-x86_64-xpu', package / 'bin/embeddinggemma2-jetha')
    for directory in (WORK / 'sdk', BUILD, PREFIX):
        directory.rename(directory.with_name(directory.name + '-hidden'))
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith('EI_') and
                 k not in ('LD_LIBRARY_PATH', 'GGML_BACKEND_PATH', 'ONEAPI_ROOT', 'MKLROOT')}
    clean_env.update(EI_THREADS='2', EI_CPU_AUDIO_F16_2='1', ONEAPI_DEVICE_SELECTOR='!*:*')
    for path in package.rglob('*'):
        if path.is_file() and elf(path):
            versions = subprocess.check_output(['readelf', '--version-info', str(path)], text=True)
            for kind, limit in (('GLIBC', (2, 34)), ('GLIBCXX', (3, 4, 29))):
                values = [tuple(map(int, v.split('.'))) for v in
                          re.findall(r'\b' + kind + r'_(\d+(?:\.\d+)+)', versions)]
                assert not values or max(values) <= limit, (path, kind, values)
            output = subprocess.check_output(['ldd', str(path)], env=clean_env, text=True)
            assert 'not found' not in output and str(SDK) not in output and str(BUILD) not in output, output
    run([sys.executable, ROOT / 'tests/installed_service2.py', '--prefix', package], env=clean_env)
    regression[regression.index('--binary') + 1] = package / 'bin/embeddinggemma2-jetha'
    run(regression, env=clean_env)
    installed_journey(package, clean_env)
    run([sys.executable, ROOT / 'tests/audio_f16_reference2.py', '--binary',
         package / 'bin/embeddinggemma2-jetha', '--model',
         ROOT / 'model/embeddinggemma-2-Q8_0.gguf', '--mmproj',
         ROOT / 'model/mmproj-embeddinggemma-2-Q8_0.gguf', '--threads', '2',
         '--log-dir', LOGS], env=clean_env)
    print('XPU24_INSTALLED_AUTO_CPU_FALLBACK_ALL5_TERMINAL0 noIntelGPUqualification', flush=True)


if __name__ == '__main__':
    main()
