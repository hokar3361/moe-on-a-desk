#!/bin/bash
# TensorFold on this Mac under the fixed metric (thinking off, greedy): bench-strata 1K/4K + code/prose 400 by streaming.
# usage: tfab.sh NAME "ENV=VAL ..." "serve args"
NAME=$1; ENVS=$2; ARGS=$3
TF=$HOME/venv-tf/bin/tensorfold
MODEL=TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP
OUT=$HOME/tfab; mkdir -p $OUT
pkill -f "[l]lama-server"; pkill -f "[t]ensorfold serve"; sleep 3
env $ENVS nohup $TF serve $MODEL --port 8090 --name m --no-thinking --no-update-check $ARGS > $OUT/$NAME.log 2>&1 < /dev/null &
for i in $(seq 1 400); do curl -s -m 2 http://127.0.0.1:8090/v1/models | grep -q '"m"' && break; sleep 3; done
if ! curl -s -m 2 http://127.0.0.1:8090/v1/models | grep -q '"m"'; then echo "$NAME: server did not come up"; tail -20 $OUT/$NAME.log; exit 1; fi
b=$(python3 $HOME/bench-strata.py --url http://127.0.0.1:8090/v1 --model m --lengths 1024,4096 --max-tokens 256 --repeat 3 2>&1 | awk '$1 ~ /^(1024|4096)$/ {printf "%s ", $4}')
c=$(python3 -I $HOME/kbench/streambench.py http://127.0.0.1:8090/v1 m code | awk '{print $1}')
p=$(python3 -I $HOME/kbench/streambench.py http://127.0.0.1:8090/v1 m prose | awk '{print $1}')
echo "$NAME | bench 1K/4K: $b| code $c | prose $p"
grep -iE "budget|pool|ssd|resident|GiB" $OUT/$NAME.log | head -8
pkill -f "[t]ensorfold serve"
