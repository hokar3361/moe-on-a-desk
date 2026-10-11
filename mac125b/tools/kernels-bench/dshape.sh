#!/bin/bash
# dense op time (us, GPU) at B = 1..4 for the main shapes of the verify pass
cd ~/kbench
for spec in "bf16 10240 320 96" "bf16 320 10240 96" "bf16 2560 512 96" "q6_K 2560 10240 24" "q6_K 6144 2560 24" "iq4_xs 2560 10240 24" "q5_K 2560 10240 24" "q4_K 2560 10240 24" "iq4_xs 2560 6144 24" "q4_K 2560 640 96" "iq4_nl 640 2560 96"; do
  set -- $spec; line=""
  for B in 1 2 3 4; do
    r=$(./kbench time dense $1 $2 $3 1 1 $B $4 30 2>&1 | python3 -I kparse.py | sed -e "s/.*: \([0-9.]*\) us.*/\1/")
    line="$line $r"
  done
  echo "$1 K=$2 N=$3:$line"
done
