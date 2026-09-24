#!/usr/bin/env bash
# re-check earlier small-effect results with guest proactive compaction off (claim 103)
G=/home/god/dev/llama.cpp-graph/local/bench/graphtest.sh; B=/home/god/dev/llama.cpp-qsa/build/bin
R=/home/god/dev/llama.cpp-graph/local/results
run() { L=$1; shift; echo "== $L"; $G rc-$L $B --spec-draft-n-max 2 "$@" </dev/null | grep median; }
run base -lzm off
EXTRA_ENV=GGML_CUDA_GRAPH_KEY_LEGACY=1 run legacykey -lzm off
run lzmon -lzm on
run t8 -lzm off -t 8
run t48 -lzm off -t 48
run pmin075 -lzm off --spec-draft-p-min 0.75
run base2 -lzm off
echo ALLDONE
