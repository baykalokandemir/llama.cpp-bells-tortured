#!/usr/bin/env bash
# ABAB: same hc-mmvf binary, old dispatch (GGML_CUDA_MMVF_THIN_ROWS=0) vs thin-row mmvf (default 8)
G=/home/god/dev/llama.cpp-graph/local/bench/graphtest.sh; B=/home/god/dev/llama.cpp-hcmmvf/build/bin
run() { L=$1; shift; echo "== $L"; $G hm-$L $B --spec-draft-n-max 2 -lzm off "$@" </dev/null | grep -E "median|graphs"; }
for r in 1 2; do
  EXTRA_ENV=GGML_CUDA_MMVF_THIN_ROWS=0 run off$r
  run on$r
done
echo ALLDONE
