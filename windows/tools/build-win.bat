@echo off
rem Native Windows build of the fork (llama-server, llama-quantize, llama-bench) with MSVC, CUDA and Ninja.
rem usage: build-win.bat SRC [ARCHS] [BUILD_DIR]
rem   SRC   : the llama.cpp tree with the patches applied (see windows\README.md)
rem   ARCHS : CUDA architectures, default 89 (RTX 40 series); 120 for RTX 50 series
rem   env VCVARS   : vcvars64.bat of Visual Studio 2022 (default: the Community edition path below)
rem   env VCVARS_VER: optional MSVC toolset version, e.g. 14.42 (the version used for the published numbers)
rem   env CUDA_PATH: CUDA toolkit root (set by the CUDA installer; 13.0 was used)
rem   env JOBS     : parallel compile jobs, default 4 (nvcc/ptxas crashed at random on 8+ jobs on the test PC)
rem The CUDA runtime DLLs are copied next to the binaries.
setlocal
set SRC=%~1
if "%SRC%"=="" (echo usage: build-win.bat SRC [ARCHS] [BUILD_DIR] & exit /b 1)
set ARCH=%~2
if "%ARCH%"=="" set ARCH=89
set BDIR=%~3
if "%BDIR%"=="" set BDIR=build
if "%JOBS%"=="" set JOBS=4
if "%VCVARS%"=="" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%SRC%\CMakeLists.txt" (echo NO SOURCE TREE %SRC% & exit /b 1)
if not exist "%VCVARS%" (echo vcvars64.bat not found: set VCVARS & exit /b 1)
if "%CUDA_PATH%"=="" (echo CUDA_PATH is not set & exit /b 1)
if "%VCVARS_VER%"=="" (call "%VCVARS%" >nul) else (call "%VCVARS%" -vcvars_ver=%VCVARS_VER% >nul)
set PATH=%CUDA_PATH%\bin;%PATH%
cd /d "%SRC%" || exit /b 1
cmake -B %BDIR% -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=%ARCH%" -DLLAMA_CURL=OFF ^
  "-DCUDAToolkit_ROOT=%CUDA_PATH%" "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe" > %BDIR%-cfg.log 2>&1
if errorlevel 1 (echo CONFIGURE FAILED & type %BDIR%-cfg.log & exit /b 1)
cmake --build %BDIR% --target llama-server llama-quantize llama-bench -j %JOBS% > %BDIR%.log 2>&1
if errorlevel 1 (echo BUILD FAILED, see %SRC%\%BDIR%.log & exit /b 1)
for %%F in (cudart64_13.dll cublas64_13.dll cublasLt64_13.dll) do (
  if exist "%CUDA_PATH%\bin\x64\%%F" copy /y "%CUDA_PATH%\bin\x64\%%F" "%BDIR%\bin\" >nul
  if exist "%CUDA_PATH%\bin\%%F" copy /y "%CUDA_PATH%\bin\%%F" "%BDIR%\bin\" >nul
)
echo BUILD OK %SRC%\%BDIR%\bin
