"""Regression: an installed static CUDA GGML package must link a consumer."""
import argparse
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--prefix', type=Path, required=True)
p.add_argument('--cmake', default='cmake')
a = p.parse_args()
with tempfile.TemporaryDirectory(prefix='ggml-cuda-consumer-') as directory:
    root = Path(directory)
    (root / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.18)
project(installed_ggml_cuda LANGUAGES C CXX)
find_package(ggml CONFIG REQUIRED)
if(NOT GGML_CUDA OR GGML_SHARED_LIB)
    message(FATAL_ERROR "This regression requires installed static CUDA GGML")
endif()
add_executable(consumer consumer.c)
target_link_libraries(consumer PRIVATE ggml::ggml)
set_property(TARGET consumer PROPERTY LINKER_LANGUAGE CXX)
''', encoding='utf-8')
    (root / 'consumer.c').write_text('''#include "ggml-backend.h"
int main(void) {
    ggml_backend_load_all();
    return ggml_backend_dev_count() == 0;
}
''', encoding='utf-8')
    build = root / 'build'
    subprocess.run([a.cmake, '-S', str(root), '-B', str(build),
                    f'-DCMAKE_PREFIX_PATH={a.prefix.resolve()}'], check=True)
    subprocess.run([a.cmake, '--build', str(build), '--config', 'Release'], check=True)
    binary = build / 'consumer'
    if not binary.exists():
        binary = build / 'Release' / 'consumer.exe'
    subprocess.run([str(binary)], check=True)
print('installed static CUDA GGML consumer links and starts: passed')
