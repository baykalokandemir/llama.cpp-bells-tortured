#!/usr/bin/env bash
# ABAB: same hc-mmvf binary, old dispatch (GGML_CUDA_MMVF_FALLBACK_MIN_ROWS=-1) vs mmvf for mmf-rejected rows (default >= 16)
G=/home/god/dev/llama.cpp-graph/local/bench/graphtest.sh; B=/home/god/dev/llama.cpp-hcmmvf/build/bin
vm() { grep -E "^(compact_stall|compact_daemon_wake|pgmigrate_success) " /proc/vmstat | tr "\n" " "; echo; }
run() { L=$1; shift; echo "== $L"; vm; $G hm2-$L $B --spec-draft-n-max 2 -lzm off "$@" </dev/null | grep -E "median|graphs"; vm; }
for r in 3 4; do
  run on$r
  EXTRA_ENV=GGML_CUDA_MMVF_FALLBACK_MIN_ROWS=-1 run off$r
done
echo ALLDONE
