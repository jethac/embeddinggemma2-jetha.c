"""Build an MSVC CUDA plugin for this project's MinGW Windows application.

Use the patched llama.cpp source and a completed MinGW application build.
The CUDA DLL imports that build's C ABI; it never builds a second ggml-base.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess


def run(command, environment, directory=None):
    executable = shutil.which(str(command[0]), path=environment['PATH']) or str(command[0])
    subprocess.run([executable, *map(str, command[1:])], env=environment,
                   cwd=directory, check=True, creationflags=subprocess.CREATE_NO_WINDOW)


def main(args):
    source = args.source.resolve()
    cpu = args.cpu_runtime.resolve()
    output = args.output.resolve()
    if output == cpu or cpu.is_relative_to(output) or output.is_relative_to(cpu):
        raise RuntimeError('output must be separate from the existing MinGW runtime')
    if not (source / 'ggml/include/ggml-backend.h').is_file():
        raise RuntimeError('--source must be the patched llama.cpp source directory')
    if not (cpu / 'embeddinggemma2-jetha.exe').is_file() or not (cpu / 'ggml-base.dll').is_file():
        raise RuntimeError('--cpu-runtime must contain the MinGW application and ggml-base.dll')
    cuda = args.cuda_root.resolve()
    if not (cuda / 'bin/nvcc.exe').is_file():
        raise RuntimeError('--cuda-root must contain bin/nvcc.exe')
    if not re.fullmatch(r'[0-9]+a?', args.cuda_arch):
        raise RuntimeError('--cuda-arch must be a CUDA architecture, for example 120a')
    vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    installation = subprocess.check_output([
        str(vswhere), '-latest', '-products', '*', '-requires',
        'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property',
        'installationPath'], text=True).strip()
    if not installation:
        raise RuntimeError('Visual Studio with the x64 C++ tools is required')
    visual_studio = Path(installation)
    vcvars = visual_studio / 'VC/Auxiliary/Build/vcvars64.bat'
    output.mkdir(parents=True, exist_ok=True)
    environment_command = output / 'cuda-build-env.cmd'
    environment_command.write_text('@echo off\ncall "' + str(vcvars) + '" >nul\nif errorlevel 1 exit /b %errorlevel%\nset\n')
    captured = subprocess.check_output(['cmd', '/d', '/c', str(environment_command)], text=True)
    environment = {key.upper(): value for key, value in os.environ.items()}
    for line in captured.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            environment[key.upper()] = value
    cmake = args.cmake or visual_studio / 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
    ninja = args.ninja or shutil.which('ninja', path=environment['PATH'])
    if not ninja:
        raise RuntimeError('Ninja is required; use --ninja to select it')
    exports = subprocess.check_output([str(args.objdump), '-p', str(cpu / 'ggml-base.dll')], text=True)
    names = re.findall(r'^\s*\[\s*\d+\]\s+\+base\[\s*\d+\]\s+\w+\s+(ggml_\w+)$', exports, re.M)
    if len(names) < 200:
        raise RuntimeError('cannot read the MinGW ggml-base C exports')
    definition = output / 'ggml-base.def'
    definition.write_text('LIBRARY ggml-base.dll\nEXPORTS\n' + '\n'.join(names) + '\n')
    import_library = output / 'ggml-base.lib'
    run(['lib.exe', '/nologo', '/machine:x64', '/def:' + str(definition),
         '/out:' + str(import_library)], environment)
    wrapper = output / 'wrapper'
    wrapper.mkdir(exist_ok=True)
    (wrapper / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.31)
project(embeddinggemma2_windows_cuda_plugin C CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_subdirectory("''' + source.as_posix() + '''/ggml" ggml)
get_target_property(cuda_links ggml-cuda LINK_LIBRARIES)
list(REMOVE_ITEM cuda_links ggml-base)
list(APPEND cuda_links "''' + import_library.as_posix() + '''")
set_property(TARGET ggml-cuda PROPERTY LINK_LIBRARIES "${cuda_links}")
target_compile_definitions(ggml-cuda PRIVATE GGML_SHARED GGML_BACKEND_DL)
target_include_directories(ggml-cuda PRIVATE "''' + source.as_posix() + '''/ggml/include")
''')
    build = output / 'build'
    run([cmake, '-S', wrapper, '-B', build, '-G', 'Ninja',
         '-DCMAKE_MAKE_PROGRAM=' + str(ninja), '-DCMAKE_BUILD_TYPE=Release',
         '-DBUILD_SHARED_LIBS=ON', '-DGGML_BACKEND_DL=ON', '-DGGML_CPU=OFF',
         '-DGGML_CUDA=ON', '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF',
         '-DGGML_CUDA_GRAPHS=ON', '-DCMAKE_CUDA_ARCHITECTURES=' + args.cuda_arch + '-real',
         '-DCMAKE_CUDA_COMPILER=' + str(cuda / 'bin/nvcc.exe'),
         '-DCUDAToolkit_ROOT=' + str(cuda)], environment)
    run([cmake, '--build', build, '--target', 'ggml-cuda', '--parallel', args.parallel], environment)
    runtime = output / 'runtime'
    shutil.copytree(cpu, runtime, dirs_exist_ok=True)
    shutil.copy2(build / 'bin/ggml-cuda.dll', runtime / 'ggml-cuda.dll')
    for pattern in ('cudart64*.dll', 'cublas64*.dll', 'cublasLt64*.dll'):
        candidates = list((cuda / 'bin').rglob(pattern))
        if len(candidates) != 1:
            raise RuntimeError('expected one CUDA runtime DLL for ' + pattern)
        shutil.copy2(candidates[0], runtime / candidates[0].name)
    print('Native Windows CUDA runtime: ' + str(runtime))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cpu-runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda-root', type=Path, required=True)
    parser.add_argument('--cuda-arch', default='120a')
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--objdump', type=Path, default=Path('C:/msys64/mingw64/bin/objdump.exe'))
    parser.add_argument('--cmake', type=Path)
    parser.add_argument('--ninja', type=Path)
    arguments = parser.parse_args()
    if os.name != 'nt':
        parser.error('this build route requires native Windows')
    if arguments.parallel < 1:
        parser.error('--parallel must be positive')
    main(arguments)
