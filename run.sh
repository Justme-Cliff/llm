#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Add common Windows tool paths so cmake/ninja/gcc are found from bash
export PATH="/c/Program Files/CMake/bin:$PATH"
export PATH="/c/Strawberry/c/bin:$PATH"
export PATH="/c/Program Files/Git/usr/bin:$PATH"

echo "=== Building ==="
cmake -G "Ninja" -S . -B build_ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_ninja --config Release

echo "=== Training ==="
./build_ninja/train_main.exe \
  --data data.csv \
  --out out \
  --pretrain-seconds 3600 \
  --finetune-seconds 1800

echo "=== Starting chat server ==="
echo "Open http://localhost:8080 in your browser"
./build_ninja/chat_server_main.exe --ckpt out/finetune.ckpt --port 8080 --ui ui
