#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Find cmake
CMAKE=""
for p in \
  "/c/Program Files/CMake/bin/cmake" \
  "/c/Program Files (x86)/CMake/bin/cmake" \
  "/c/Strawberry/c/bin/cmake" \
  "/mingw64/bin/cmake" \
  "/usr/bin/cmake" \
  "$(which cmake 2>/dev/null)"
do
  if [ -x "$p" ] || [ -x "${p}.exe" ]; then
    CMAKE="$p"
    break
  fi
done

if [ -z "$CMAKE" ]; then
  echo "ERROR: cmake not found. Install cmake and add it to PATH."
  exit 1
fi

echo "Using cmake: $CMAKE"

echo "=== Cleaning previous build ==="
rm -rf build_ninja

echo "=== Building ==="
"$CMAKE" -G "Ninja" -S . -B build_ninja -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build build_ninja --config Release

echo "=== Training ==="
./build_ninja/train_main.exe --data data.csv --out out --pretrain-seconds 3600 --finetune-seconds 1800

echo "=== Starting chat server ==="
echo "Open http://localhost:8080 in your browser"
./build_ninja/chat_server_main.exe --ckpt out/finetune.ckpt --port 8080 --ui ui
