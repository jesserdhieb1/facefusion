@echo off
setlocal
echo ========================================================
echo   Starting FaceFusion Low-Latency Event Kiosk
echo ========================================================

set CONDA_PYTHON="C:\Users\<username>\.conda\envs\facefusion\python.exe"

if exist %CONDA_PYTHON% (
    set "PYTHON_EXE=%CONDA_PYTHON%"
) else (
    set "PYTHON_EXE=python"
)

echo Using Python: %PYTHON_EXE%

%PYTHON_EXE% facefusion.py run ^
    --ui-layouts webcam ^
    --processors face_swapper ^
    --face-swapper-model hyperswap_1a_256 ^
    --face-detector-model scrfd ^
    --face-detector-size 320x320 ^
    --face-mask-types box occlusion ^
    --execution-providers cuda ^
    --execution-thread-count 4 ^
    --video-memory-strategy strict

pause
