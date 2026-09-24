#!/usr/bin/env bash
# usage: pftest.sh LABEL NTOK args...
LABEL=$1; NTOK=$2; shift 2
B=${BIN:-/home/god/dev/llama.cpp-bells-mtp/build/bin}
M=/opt/stacks/llm-stack/models/qwen38-flash-next-gsq/IQ3_XXS_HCQ8/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-HCQ8-00001-of-00002.gguf
O=/home/god/dev/iq3-explore
env LD_LIBRARY_PATH=$B $EXTRA_ENV $B/llama-server -m $M --host 127.0.0.1 --port 8094 "$@" > $O/pf-$LABEL-server.log 2>&1 &
P=$!
for i in $(seq 1 300); do [ "$(curl -s -m 2 -o /dev/null -w %{http_code} http://127.0.0.1:8094/health)" = 200 ] && break; kill -0 $P 2>/dev/null || { echo "$LABEL LOADFAIL"; grep -iE "error|out of memory" $O/pf-$LABEL-server.log | head -3; exit 2; }; sleep 2; done
echo "$LABEL vram $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tr "\n" " ")"
python3 $O/pftest.py 8094 $O/pf-$LABEL.json $NTOK 2>&1 | sed "s/^/$LABEL /"
kill $P; wait $P 2>/dev/null
