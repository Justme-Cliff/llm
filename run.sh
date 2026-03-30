#\!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CMAKE="/c/Program Files/CMake/bin/cmake"
NINJA="/c/Strawberry/c/bin/ninja"

echo "=== Cleaning previous build ==="
rm -rf build_ninja

echo "=== Building ==="
"$CMAKE" -G "Ninja" -DCMAKE_MAKE_PROGRAM="$NINJA" -S . -B build_ninja -DCMAKE_BUILD_TYPE=Release
"$CMAKE" --build build_ninja --config Release

echo "=== Training ==="
./build_ninja/train_main.exe \n  --data data.csv \n  --out out \n  --pretrain-seconds 3600 \n  --finetune-seconds 1800

echo "=== Starting chat server ==="
echo "Open http://localhost:8080 in your browser"
./build_ninja/chat_server_main.exe --ckpt out/finetune.ckpt --port 8080 --ui ui
