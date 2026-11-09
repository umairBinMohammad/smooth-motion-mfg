#!/bin/sh
# Cross-builds the tracer (build/version.dll) with llvm-mingw.
# Toolchain: https://github.com/mstorsjo/llvm-mingw (set TOOLCHAIN to its bin directory).
set -eu
TOOLCHAIN="${TOOLCHAIN:-}"
[ -n "$TOOLCHAIN" ] && PATH="$TOOLCHAIN:$PATH"
CC=x86_64-w64-mingw32-clang
CXX=x86_64-w64-mingw32-clang++
MH=third_party/minhook
mkdir -p build/obj
for f in $MH/src/buffer.c $MH/src/hook.c $MH/src/trampoline.c $MH/src/hde/hde64.c; do
    $CC -O2 -I$MH/include -c "$f" -o "build/obj/$(basename "$f" .c).o"
done
$CXX -O2 -std=c++17 -Wall -Wextra -fno-exceptions -fno-rtti -I$MH/include \
    -shared -static -o build/version.dll \
    src/tracer/tracer.cpp src/tracer/proxy.cpp build/obj/*.o src/tracer/version.def \
    -Wl,--no-insert-timestamp
echo "built build/version.dll"
