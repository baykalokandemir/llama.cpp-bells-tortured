#!/usr/bin/env bash
# FR-Spec draft vocab A/B on the graphtest workload. Target output is identical in every arm
# (greedy verify against the full head), so only draft acceptance and pass time move.
G=/home/god/dev/llama.cpp-graph/local/bench/graphtest.sh; B=/home/god/dev/llama.cpp-mtpvocab/build/bin
R=/home/god/dev/llama.cpp-mtpvocab/local/results/fr
run() { L=$1; shift; echo "== $L"; $G fr-$L $B --spec-draft-n-max 2 -lzm off "$@" </dev/null | grep -E "median|graphs";
  grep -m1 "MTP draft head reduced" /home/god/dev/llama.cpp-graph/local/results/fr-$L-server.log | sed "s/.*reduced/reduced/";
  python3 -c "import json;r=json.load(open('/home/god/dev/llama.cpp-graph/local/results/fr-$L.json'));print('acc', [(x['case'][0], x['draft_n'], x['draft_acc']) for x in r[:2]])"; }
ARMS=${ARMS:-"base ident n32k n16k n64k base2"}
for a in $ARMS; do
  case $a in
    base*)  EXTRA_ENV= run $a ;;
    ident)  EXTRA_ENV="LLAMA_MTP_VOCAB=$R/ident.ids" run $a ;;
    n*k)    n=$(( ${a:1:-1} * 1024 )); EXTRA_ENV="LLAMA_MTP_VOCAB=$R/rank.ids LLAMA_MTP_VOCAB_N=$n" run $a ;;
  esac
done
echo ALLDONE
