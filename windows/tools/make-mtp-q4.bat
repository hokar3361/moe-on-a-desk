@echo off
rem Smaller draft (MTP) head: its 512 experts (2.55 GB in Q8_0) re-quantized to Q4_K (gate/up) and Q5_0 (down: 640 columns are not a
rem multiple of 256, llama-quantize falls back to Q5_0); everything else stays Q8_0. 4.14 GB -> 2.98 GB, about 1.1 GB less VRAM.
rem The head only drafts: every drafted token is verified by the full model, so the output of the model does not change.
rem usage: make-mtp-q4.bat LLAMA_QUANTIZE_EXE SRC_Q8_0_GGUF DST_GGUF
if "%~3"=="" (echo usage: make-mtp-q4.bat llama-quantize.exe mtp-Qwen3.8-Flash-Next-Q8_0.gguf mtp-Qwen3.8-Flash-Next-Q8e4.gguf & exit /b 1)
"%~1" --allow-requantize --tensor-type "ffn_gate_exps=q4_k" --tensor-type "ffn_up_exps=q4_k" --tensor-type "ffn_down_exps=q4_k" "%~2" "%~3" Q8_0
