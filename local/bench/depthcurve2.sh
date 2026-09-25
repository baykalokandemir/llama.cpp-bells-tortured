#!/usr/bin/env bash
# usage: depthcurve.sh LABEL DEPTHS REPS server-args...
# env: BIN (llama.cpp build/bin), MODEL (main gguf), OUT (output dir), EXTRA_ENV, FILLER (long text for depth prompts)
LABEL=$1; DEPTHS=$2; REPS=$3; shift 3
B=${BIN:-$(cd "$(dirname "$0")/../.." && pwd)/build/bin}
M=${MODEL:-/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf}
O=${OUT:-.}; D=$(dirname "$0")
env LD_LIBRARY_PATH=$B $EXTRA_ENV $B/llama-server -m $M --host 127.0.0.1 --port 8094 "$@" > $O/dc-$LABEL-server.log 2>&1 &
P=$!
for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; kill -0 $P 2>/dev/null || { echo "$LABEL LOADFAIL"; grep -iE "out of memory|error" $O/dc-$LABEL-server.log | head -3; exit 2; }; sleep 2; done
echo "$LABEL vram-load $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tr "\n" " ")"
python3 ${CLIENT:-$D/depthcurve2.py} 8094 $O/dc-$LABEL.json $DEPTHS $REPS 2>&1 | sed "s/^/$LABEL /"
echo "$LABEL vram-end $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tr "\n" " ")"
grep -m20 "shall_use_sparse" $O/dc-$LABEL-server.log | sed "s/^/$LABEL LOG /" | cut -c1-200
kill $P; wait $P 2>/dev/null
