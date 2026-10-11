#!/bin/bash
# fixed metric for the best state: runs the FIXED measurement (~/mtpab.sh) with the best build and configuration.
# (no other timing method; the comparison targets are fixed values and are never re-measured)
E="GGML_ALLOC_FREE_DELAY=4 LLAMA_SPEC_MTP_MERGE=1 LLAMA_GRAPH_CACHE=16 LLAMA_SPEC_STOP=1"
~/run-best.sh stop > /dev/null 2>&1
BIN_OVERRIDE=$HOME/llama-best/build/bin/llama-server bash $HOME/mtpab.sh "best:$E|--reasoning off --spec-draft-n-max 7" 2>&1 | head -1
