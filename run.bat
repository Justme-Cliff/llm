@echo off
cd /d "%~dp0"

echo === Cleaning previous build ===
if exist build_ninja rmdir /s /q build_ninja

echo === Building ===
cmake -G "Ninja" -S . -B build_ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_ninja --config Release

echo === Training ===
build_ninja\train_main.exe --data data.csv --out out --pretrain-seconds 3600 --finetune-seconds 1800

echo === Starting chat server ===
echo Open http://localhost:8080 in your browser
build_ninja\chat_server_main.exe --ckpt out\finetune.ckpt --port 8080 --ui ui
