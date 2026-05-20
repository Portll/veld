@echo off
REM Veld server launcher (Windows).
REM
REM Reads the API key, host, and port from %APPDATA%\veld\config.toml.
REM Run "veld.exe init" once first to generate that file.
REM Any extra arguments are passed through to "veld server".

cd /d "%~dp0"

set "VELD_BIN="
if exist "target\release\veld.exe" set "VELD_BIN=target\release\veld.exe"
if not defined VELD_BIN if exist "target\x86_64-pc-windows-msvc\release\veld.exe" set "VELD_BIN=target\x86_64-pc-windows-msvc\release\veld.exe"

if not defined VELD_BIN (
    echo veld.exe not found. Build it first with: cargo build --release
    exit /b 1
)

"%VELD_BIN%" server %*
