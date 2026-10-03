@echo off
setlocal
echo ========================================================
echo   Building ApexChess High-Performance C++ Core Engine
echo ========================================================

set PATH=C:\msys64\mingw64\bin;%PATH%
if not exist "bin" mkdir bin

echo [1/2] Compiling with g++ (AVX2, BMI2, -O3, C++20, Static)...
g++ -O3 -std=c++20 -mavx2 -mbmi2 -static -static-libgcc -static-libstdc++ -I src/cpp ^
    src/cpp/bitboard.cpp ^
    src/cpp/position.cpp ^
    src/cpp/movegen.cpp ^
    src/cpp/evaluate.cpp ^
    src/cpp/search.cpp ^
    src/cpp/uci.cpp ^
    src/cpp/main.cpp ^
    -o bin\apex_engine.exe

if %ERRORLEVEL% EQU 0 (
    echo [2/2] [OK] Successfully built bin\apex_engine.exe!
    bin\apex_engine.exe --version
) else (
    echo [ERROR] Build failed!
)
