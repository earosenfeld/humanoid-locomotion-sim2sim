# Root-free toolchain: `zig c++` as the compiler (see scripts/setup-cpp-toolchain-wsl.sh).
find_program(ZIG_CC zig-cc HINTS $ENV{HOME}/.local/bin REQUIRED)
find_program(ZIG_CXX zig-c++ HINTS $ENV{HOME}/.local/bin REQUIRED)
set(CMAKE_C_COMPILER ${ZIG_CC})
set(CMAKE_CXX_COMPILER ${ZIG_CXX})
