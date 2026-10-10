cmake_minimum_required(VERSION 3.24)
project(embeddinggemma2_jetha LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
include(GNUInstallDirs)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
include(FetchContent)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS ON CACHE BOOL "" FORCE)
set(EI_DISPATCH_DEFAULT OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(EI_DISPATCH_DEFAULT ON)
endif()
option(EI_CPU_DISPATCH "Load the best CPU/OS-compatible kernel variant at runtime" ${EI_DISPATCH_DEFAULT})
if(EI_CPU_DISPATCH)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
else()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
endif()
set(GGML_NATIVE OFF CACHE BOOL "Portable CPU baseline")
set(GGML_AVX512 OFF CACHE BOOL "Local AVX-512 experiment")
FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b)
FetchContent_MakeAvailable(llama)
# Keep media fixes and CPU/OS dispatch guards as separate, adoptable patches
# against the pinned dependency.
find_package(Git REQUIRED)
find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/apply-dependency-patches2.py"
    --source "${llama_SOURCE_DIR}"
    --git "${GIT_EXECUTABLE}" --base de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-gemma2.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/ggml-cpu-dispatch.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-video-container.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-deadline.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/ggml-cuda-mask-range.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-media-id.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-resize-rows.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-video-probe.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-encoder-profile.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/ggml-cpu-q8-pair.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-audio-graph-cache.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-pixel-profile.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-pixel-reuse.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/mtmd-vision-clip-metadata.patch"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/ggml-cuda-static-package.patch"
    RESULT_VARIABLE EI_PATCH_RESULT ERROR_VARIABLE EI_PATCH_ERROR)
if(NOT EI_PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Cannot apply pinned dependency patches: ${EI_PATCH_ERROR}")
endif()
if(NOT TARGET mtmd)
    # The dependency's version variables are scoped to its directory. Copy the
    # actual library properties before configuring its media library here.
    get_target_property(LLAMA_VERSION_BASE llama VERSION)
    get_target_property(LLAMA_VERSION_MAJOR llama SOVERSION)
    add_subdirectory("${llama_SOURCE_DIR}/tools/mtmd" "${llama_BINARY_DIR}/tools/mtmd")
endif()
find_package(Threads REQUIRED)
option(EI_VIDEO_PROBE "Build the optional small libavformat video metadata probe" OFF)
if(EI_VIDEO_PROBE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(EI_AVFORMAT REQUIRED IMPORTED_TARGET libavformat>=59 libavutil>=57)
    add_executable(embeddinggemma2-video-probe src/video_probe2.c)
    target_link_libraries(embeddinggemma2-video-probe PRIVATE PkgConfig::EI_AVFORMAT)
    install(TARGETS embeddinggemma2-video-probe RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
include(ExternalProject)
FetchContent_Declare(jpeg2
    URL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.2.0/libjpeg-turbo-3.2.0.tar.gz
    URL_HASH SHA256=6f30092cef9fb839779646608f4ee14ae3cbac989c47fa05e841b0841f09878e
    SOURCE_SUBDIR src)
FetchContent_MakeAvailable(jpeg2)
# libjpeg-turbo requires an independent CMake project. Statically link its
# runtime-dispatched decoder; do not install its developer tools or headers.
set(EI_JPEG_BINARY "${CMAKE_BINARY_DIR}/jpeg2-build")
set(EI_JPEG_ARCHIVE "${EI_JPEG_BINARY}/${CMAKE_STATIC_LIBRARY_PREFIX}turbojpeg${CMAKE_STATIC_LIBRARY_SUFFIX}")
if(MSVC)
    set(EI_JPEG_ARCHIVE "${EI_JPEG_BINARY}/Release/turbojpeg-static.lib")
endif()
ExternalProject_Add(ei-jpeg2-build
    SOURCE_DIR "${jpeg2_SOURCE_DIR}" BINARY_DIR "${EI_JPEG_BINARY}"
    DOWNLOAD_COMMAND "" INSTALL_COMMAND ""
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_TOOLS=OFF
        -DWITH_TESTS=OFF -DWITH_SIMD=ON -DREQUIRE_SIMD=ON
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target turbojpeg-static
    BUILD_BYPRODUCTS "${EI_JPEG_ARCHIVE}")
add_library(ei-jpeg2 STATIC IMPORTED)
set_target_properties(ei-jpeg2 PROPERTIES IMPORTED_LOCATION "${EI_JPEG_ARCHIVE}"
    INTERFACE_INCLUDE_DIRECTORIES "${jpeg2_SOURCE_DIR}/src")
add_dependencies(ei-jpeg2 ei-jpeg2-build)
if(UNIX)
    set_property(TARGET ei-jpeg2 PROPERTY INTERFACE_LINK_LIBRARIES m)
endif()
FetchContent_Declare(simdjson2
    GIT_REPOSITORY https://github.com/simdjson/simdjson.git
    GIT_TAG bf89ddd00ef5fcb6a7f20611ea8853d258b67249 # v5.0.3
    SOURCE_SUBDIR include) # Populate sources without its developer targets.
FetchContent_MakeAvailable(simdjson2)
# Keep runtime CPU dispatch and scalar fallback; do not build with -march=native.
add_library(ei-simdjson2 STATIC "${simdjson2_SOURCE_DIR}/singleheader/simdjson.cpp")
target_include_directories(ei-simdjson2 PUBLIC "${simdjson2_SOURCE_DIR}/singleheader")
add_executable(embeddinggemma2-jetha
    src/server.c src/engine2.c src/model2.c src/gguf.c src/tokenizer2.c
    src/kernels.c src/quants.c src/parallel.c src/inference_service.c
    src/response_cache.c src/float_format.c src/media_request2.cpp src/media_service2.c src/jpeg_decode2.c)
target_compile_definitions(embeddinggemma2-jetha PRIVATE EI_GEMMA2)
target_include_directories(embeddinggemma2-jetha PRIVATE "${llama_SOURCE_DIR}/vendor")
target_link_libraries(embeddinggemma2-jetha PRIVATE mtmd ggml llama vendor::nlohmann ei-simdjson2 ei-jpeg2 Threads::Threads)
if(WIN32)
    target_compile_definitions(embeddinggemma2-jetha PRIVATE _CRT_RAND_S)
    target_link_libraries(embeddinggemma2-jetha PRIVATE ws2_32)
endif()
if(MINGW)
    target_link_options(embeddinggemma2-jetha PRIVATE -static-libgcc -static-libstdc++)
endif()
if(NOT MSVC)
    target_compile_options(embeddinggemma2-jetha PRIVATE -Wall -Wextra -Werror)
    target_link_libraries(embeddinggemma2-jetha PRIVATE m)
endif()
install(TARGETS embeddinggemma2-jetha RUNTIME DESTINATION bin)
set(EI_LICENSE_DIR "${CMAKE_INSTALL_DATADIR}/licenses/embeddinggemma2-jetha")
install(FILES LICENSE DESTINATION "${EI_LICENSE_DIR}" RENAME LICENSE)
install(FILES "${llama_SOURCE_DIR}/LICENSE" DESTINATION "${EI_LICENSE_DIR}" RENAME llama.cpp-LICENSE)
install(FILES deps/LICENSE-nlohmann.txt DESTINATION "${EI_LICENSE_DIR}")
install(FILES "${simdjson2_SOURCE_DIR}/LICENSE-MIT" DESTINATION "${EI_LICENSE_DIR}" RENAME simdjson-LICENSE-MIT)
install(FILES "${jpeg2_SOURCE_DIR}/LICENSE.md" "${jpeg2_SOURCE_DIR}/README.ijg"
    DESTINATION "${EI_LICENSE_DIR}/libjpeg-turbo")
install(FILES "${jpeg2_SOURCE_DIR}/src/spng/LICENSE" "${jpeg2_SOURCE_DIR}/src/spng/libpng-LICENSE.txt"
    DESTINATION "${EI_LICENSE_DIR}/libjpeg-turbo/spng")
install(FILES "${jpeg2_SOURCE_DIR}/src/spng/zlib/LICENSE"
    DESTINATION "${EI_LICENSE_DIR}/libjpeg-turbo/zlib")
foreach(vendor_license hash/xxhash/LICENSE hash/sha256/LICENSE hash/rotate-bits/LICENSE.md)
    get_filename_component(vendor_directory "${vendor_license}" DIRECTORY)
    install(FILES "${llama_SOURCE_DIR}/vendor/${vendor_license}"
        DESTINATION "${EI_LICENSE_DIR}/${vendor_directory}")
endforeach()
if(MINGW)
    # Shared kernels/media still import the compiler runtimes even though the
    # executable statically links its own C++ runtime. Keep them beside the DLLs.
    get_filename_component(EI_MINGW_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
    file(GLOB EI_MINGW_RUNTIME_DLLS
        "${EI_MINGW_BIN}/libgcc_s_*.dll" "${EI_MINGW_BIN}/libstdc++-*.dll"
        "${EI_MINGW_BIN}/libwinpthread-*.dll" "${EI_MINGW_BIN}/libgomp-*.dll")
    install(FILES ${EI_MINGW_RUNTIME_DLLS} DESTINATION "${CMAKE_INSTALL_BINDIR}")
    foreach(runtime_license gcc-libs libwinpthread)
        set(runtime_license_dir "${EI_MINGW_BIN}/../share/licenses/${runtime_license}")
        if(EXISTS "${runtime_license_dir}")
            install(DIRECTORY "${runtime_license_dir}/"
                DESTINATION "${EI_LICENSE_DIR}/${runtime_license}")
        endif()
    endforeach()
endif()
add_executable(decoder-stall2 EXCLUDE_FROM_ALL tests/decoder_stall2.c)
add_executable(decoder-deadline-check2 EXCLUDE_FROM_ALL tests/decoder_deadline2.cpp)
add_executable(media-id-check2 EXCLUDE_FROM_ALL tests/media_id2.cpp)
target_link_libraries(media-id-check2 PRIVATE mtmd)
add_executable(jpeg-decode-check2 EXCLUDE_FROM_ALL tests/jpeg_decode2.c src/jpeg_decode2.c)
target_include_directories(jpeg-decode-check2 PRIVATE src)
target_link_libraries(jpeg-decode-check2 PRIVATE ei-jpeg2)
target_link_libraries(decoder-deadline-check2 PRIVATE mtmd)
add_executable(media-admission-check2 EXCLUDE_FROM_ALL tests/media_admission2.c src/media_service2.c)
target_include_directories(media-admission-check2 PRIVATE src "${llama_SOURCE_DIR}/vendor")
target_compile_definitions(media-admission-check2 PRIVATE EI_GEMMA2)
target_link_libraries(media-admission-check2 PRIVATE Threads::Threads)
add_executable(response-persistence-check2 EXCLUDE_FROM_ALL tests/response_persistence2.c src/response_cache.c)
target_include_directories(response-persistence-check2 PRIVATE src "${llama_SOURCE_DIR}/vendor")
target_compile_definitions(response-persistence-check2 PRIVATE EI_GEMMA2)
target_link_libraries(response-persistence-check2 PRIVATE Threads::Threads)
if(WIN32)
    target_compile_definitions(response-persistence-check2 PRIVATE _CRT_RAND_S)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    foreach(EI_CHECK_ISA scalar avx avx512)
        add_executable(cpu-dispatch-${EI_CHECK_ISA}-check2 EXCLUDE_FROM_ALL tests/cpu_dispatch2.cpp)
        target_include_directories(cpu-dispatch-${EI_CHECK_ISA}-check2 PRIVATE
            "${llama_SOURCE_DIR}/ggml/src" "${llama_SOURCE_DIR}/ggml/include")
        target_compile_definitions(cpu-dispatch-${EI_CHECK_ISA}-check2 PRIVATE GGML_BACKEND_DL)
        if(NOT EI_CHECK_ISA STREQUAL "scalar")
            target_compile_definitions(cpu-dispatch-${EI_CHECK_ISA}-check2 PRIVATE GGML_AVX)
        endif()
        if(EI_CHECK_ISA STREQUAL "avx512")
            target_compile_definitions(cpu-dispatch-${EI_CHECK_ISA}-check2 PRIVATE GGML_AVX2 GGML_AVX512)
        endif()
    endforeach()
endif()
add_executable(reference2 EXCLUDE_FROM_ALL perf/reference2.c)
target_link_libraries(reference2 PRIVATE llama ggml)
if(NOT MSVC)
    target_link_libraries(reference2 PRIVATE m)
endif()
if(MINGW)
    target_link_options(reference2 PRIVATE -static-libgcc -static-libstdc++)
endif()
