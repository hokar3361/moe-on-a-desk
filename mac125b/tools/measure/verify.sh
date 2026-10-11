#!/bin/bash
# greedy 200-token answer must match the baseline build byte for byte
M=$HOME/models/qwen38-gsq/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
P="Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include error handling."
h=$(env "$@" $HOME/llama.cpp/build/bin/llama-simple -m $M -n 200 -ngl 99 "$P" 2>/dev/null | md5 -q)
[ "$h" = "$(md5 -q $HOME/eval/base-gen.txt)" ] && echo "VERIFY OK $*" || echo "VERIFY FAILED $* ($h)"
