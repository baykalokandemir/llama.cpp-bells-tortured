#!/bin/bash
# usage: navarm.sh <label> <llama-server path>
# The friend's 5090 command, verbatim except for underscores restored (the paste ate them).
set -u
LABEL=$1; BIN=$2
D=/home/god/dev/navin-ab; mkdir -p $D
M=/opt/stacks/llm-stack/models/navin-ad427/Qwen3.8-Flash-Next-Uncensored-AD-4.27-main-00001-of-00034.gguf
L=$(seq -s'|' 14 47)
OT="blk\.($L)\.ffn_(up|down|gate|gate_up)_(ch_|)exps=CPU,per_layer_token_embd=CPU"

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "GPU busy, aborting"; nvidia-smi --query-compute-apps=pid,process_name --format=csv; exit 2
fi

export BELLS_HOST_ONLY=1 LLAMA_DRAFT_UBATCH=128
$BIN -m $M -ngl 999 ${TS:+-ts $TS} -ot "$OT" \
  -fit off -lm none -lzm auto -fa on -t 8 -tb 24 \
  -c ${CTX:-262144} -b 512 -ub 512 -np 1 -ctk q4_0 -ctv q4_0 \
  -ctxcp 0 -cms 2048 -cram 0 --bells-slots 16 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.75 \
  --spec-draft-type-k f16 --spec-draft-type-v f16 \
  --jinja --host 127.0.0.1 --port 8199 --alias rvn-flash-next -lv 3 \
  > $D/$LABEL-server.log 2>&1 &
SPID=$!
T0=$(date +%s)

# 1 Hz monitor: per-card VRAM and host MemAvailable
( while kill -0 $SPID 2>/dev/null; do
    echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ') $(awk '/MemAvailable/{print $2}' /proc/meminfo)"
    sleep 1; done ) > $D/$LABEL-mon.log &

until curl -sf 127.0.0.1:8199/health >/dev/null; do
  kill -0 $SPID 2>/dev/null || { echo "server died"; tail -30 $D/$LABEL-server.log; exit 1; }
  sleep 2
done
echo "load_s $(( $(date +%s) - T0 ))" | tee $D/$LABEL-bench.log
python3 $D/navbench.py http://127.0.0.1:8199 3 2>&1 | tee -a $D/$LABEL-bench.log
kill -INT $SPID; wait $SPID 2>/dev/null
awk '{g0=($2>g0)?$2:g0; g1=($3>g1)?$3:g1; if(m==""||$4<m)m=$4} END{printf "peak_vram_mib %d %d total %d  min_avail_gib %.2f\n",g0,g1,g0+g1,m/1048576}' $D/$LABEL-mon.log | tee -a $D/$LABEL-bench.log
