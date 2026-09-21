#!/usr/bin/env sh
# C++ toolchain without root: zig (clang-based C/C++ compiler) + cmake/ninja wheels in .venv.
set -eu
mkdir -p "$HOME/.local/zig" "$HOME/.local/bin"
if ! command -v zig >/dev/null; then
  url=$(curl -fsSL https://ziglang.org/download/index.json | python3 -c \
    "import sys,json; d=json.load(sys.stdin); v=sorted((k for k in d if k!='master'), key=lambda s: tuple(map(int,s.split('.'))))[-1]; print(d[v]['x86_64-linux']['tarball'])")
  curl -fsSL "$url" | tar -xJ --strip-components=1 -C "$HOME/.local/zig"
  ln -sf "$HOME/.local/zig/zig" "$HOME/.local/bin/zig"
fi
printf '#!/bin/sh\nexec zig cc "$@"\n' > "$HOME/.local/bin/zig-cc"
printf '#!/bin/sh\nexec zig c++ "$@"\n' > "$HOME/.local/bin/zig-c++"
chmod +x "$HOME/.local/bin/zig-cc" "$HOME/.local/bin/zig-c++"
uv pip install -q -p .venv/bin/python cmake ninja
echo "configure with: .venv/bin/cmake -S cpp -B build -G Ninja --toolchain cpp/cmake/zig-toolchain.cmake"
