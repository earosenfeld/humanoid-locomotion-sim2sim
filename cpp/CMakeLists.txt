# 1 kHz humanoid control loop: manifest -> ONNX policy -> PD -> plant (MuJoCo, or your robot).
# Dependencies are fetched by CMake (Linux x86_64 prebuilt MuJoCo + onnxruntime, header-only json).
cmake_minimum_required(VERSION 3.28)
project(hls_loop LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)  # no C++20 modules; keeps zig/older clang toolchains working
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz)
FetchContent_Declare(mujoco_bin URL https://github.com/google-deepmind/mujoco/releases/download/3.13.0/mujoco-3.13.0-linux-x86_64.tar.gz)
FetchContent_Declare(ort_bin URL https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-1.30.0.tgz)
FetchContent_MakeAvailable(json mujoco_bin ort_bin)

add_library(mujoco SHARED IMPORTED)
set_target_properties(mujoco PROPERTIES
  IMPORTED_LOCATION ${mujoco_bin_SOURCE_DIR}/lib/libmujoco.so.3.13.0
  INTERFACE_INCLUDE_DIRECTORIES ${mujoco_bin_SOURCE_DIR}/include)
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ort_bin_SOURCE_DIR}/lib/libonnxruntime.so.1.30.0
  INTERFACE_INCLUDE_DIRECTORIES ${ort_bin_SOURCE_DIR}/include)

add_executable(hls_loop
  src/manifest.cpp src/obs.cpp src/policy.cpp src/plant.cpp src/loop.cpp src/main.cpp)
target_include_directories(hls_loop PRIVATE include)
target_link_libraries(hls_loop PRIVATE nlohmann_json::nlohmann_json mujoco onnxruntime pthread)
# -ffp-contract=off: no fused multiply-add, so the loop is bit-exact with the numpy reference plant
target_compile_options(hls_loop PRIVATE -O2 -Wall -Wextra -ffp-contract=off)
set_target_properties(hls_loop PROPERTIES BUILD_RPATH
  "${mujoco_bin_SOURCE_DIR}/lib;${ort_bin_SOURCE_DIR}/lib")
