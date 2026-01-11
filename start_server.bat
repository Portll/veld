@echo off
set SHODH_DEV_API_KEY=shodh-test-key
set ORT_DYLIB_PATH=C:\Users\Varun Sharma\OneDrive\Documents\Roshera\Vector-DB\vectora\kalki-v2\libs\onnxruntime.dll
cd /d "C:\Users\Varun Sharma\OneDrive\Documents\Roshera\Vector-DB\vectora\kalki-v2\shodh-memory"
target\x86_64-pc-windows-msvc\release\shodh-memory-server.exe
