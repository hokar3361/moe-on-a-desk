@echo off
rem Chat server (OpenAI-compatible, 127.0.0.1:8091, thinking off) for Qwen3.8-Flash-Next 125B IQ3_XXS on an RTX 4090-class PC, native Windows.
rem usage: run-win.bat [cpu|gpu|base|stop]
rem   cpu  (default): the fastest setting measured (windows\README.md): GPU expert cache 14.6 GB + GPU-side lookup + CPU help for
rem                   cache misses (15 threads) + MTP with the per-round depth controller. CPU about 30 %.
rem   gpu           : the light setting: GPU-only (no CPU help), MTP depth 3. CPU under 10 %.
rem   base          : unmodified llama.cpp 781dbc5 with its expert cache 14 GB and MTP depth 3 (the comparison baseline)
rem   stop          : stop the server
rem env (required): BIN = folder of the fork's llama-server.exe, MODEL = first shard of the IQ3_XXS GGUF, MTP = the small MTP head
rem env (base only): BASEBIN = folder of the unmodified llama-server.exe, MTPQ8 = the Q8_0 MTP head
rem The model must be on an SSD. The experts are locked in RAM (-lm mmap+mlock): the server uses about 49 GB of RAM.
setlocal
if "%1"=="stop" (taskkill /im llama-server.exe /f & exit /b 0)
if "%MODEL%"=="" (echo set MODEL & exit /b 1)
set COMMON=-m "%MODEL%" -c 16384 --port 8091 --host 127.0.0.1 -np 1 --reasoning off -ngl 99 --cpu-moe -lm mmap+mlock
set MODE=%1
if "%MODE%"=="" set MODE=cpu
if "%MODE%"=="cpu" (
  set LLAMA_MOE_CACHE_FETCH=1
  set LLAMA_GRAPH_CACHE=16
  set LLAMA_MOE_CPU=1
  set LLAMA_MOE_CPU_THREADS=15
  set LLAMA_MOE_CPU_AFFINITY=2-14:2,16-23
  set LLAMA_MOE_CPU_JOBS=8
  set LLAMA_MOE_CPU_MAXTOK=4
  set LLAMA_MOE_CPU_ADMIT=16
  set LLAMA_MOE_CPU_PCIE=1
  set LLAMA_SPEC_DEPTH_CTL=1
  set LLAMA_SPEC_DEPTH_MIN=1
  "%BIN%\llama-server.exe" %COMMON% --moe-cache-mib 14600 -md "%MTP%" -otd "token_embd\.weight=CPU" --spec-type draft-mtp --spec-draft-n-max 5
  exit /b
)
if "%MODE%"=="gpu" (
  set LLAMA_MOE_CACHE_FETCH=1
  set LLAMA_GRAPH_CACHE=16
  "%BIN%\llama-server.exe" %COMMON% --moe-cache-mib 14600 -md "%MTP%" -otd "token_embd\.weight=CPU" --spec-type draft-mtp --spec-draft-n-max 3
  exit /b
)
if "%MODE%"=="base" (
  "%BASEBIN%\llama-server.exe" %COMMON% --moe-cache-mib 14000 -md "%MTPQ8%" --spec-type draft-mtp --spec-draft-n-max 3
  exit /b
)
echo unknown mode %MODE%
exit /b 1
