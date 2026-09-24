#!/usr/bin/env bash
cd /home/god/dev/iq3-explore
H=/opt/stacks/llm-stack/models/qwen38-flash-next-mtp/MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
C="-ngl 99 -sm layer -ts 28,20 -lm mmap -lzm on -t 32 -tb 32 -fa on --fit off --parallel 1 --jinja --cpu-moe-pinned -md $H --spec-type draft-mtp --spec-draft-n-max 2 -devd CUDA1 -ngld 99 -ub 2048 -b 2048"
export EXTRA_ENV="LLAMA_DRAFT_UBATCH=256 GGML_CUDA_SPARSE_LOG=1"
CTX=65536; DEPTHS=8000,16000,32000,60000; S=
for s in 240 225 210; do
  ./depthcurve.sh f16-64k-s$s $DEPTHS 2 $C -c $CTX -ctk f16 -ctv f16 --bells-slots $s > dc-f16-64k-s$s.out 2>&1
  if ! grep -q LOADFAIL dc-f16-64k-s$s.out; then S=$s; break; fi
done
if [ -z "$S" ]; then
  CTX=32768; DEPTHS=8000,16000,30000
  for s in 255 240; do
    ./depthcurve.sh f16-32k-s$s $DEPTHS 2 $C -c $CTX -ctk f16 -ctv f16 --bells-slots $s > dc-f16-32k-s$s.out 2>&1
    if ! grep -q LOADFAIL dc-f16-32k-s$s.out; then S=$s; break; fi
  done
  ./depthcurve.sh q8-32k-s$S $DEPTHS 2 $C -c $CTX -ctk q8_0 -ctv q8_0 --bells-slots $S > dc-q8-32k-s$S.out 2>&1
else
  ./depthcurve.sh q8-64k-s$S $DEPTHS 2 $C -c $CTX -ctk q8_0 -ctv q8_0 --bells-slots $S > dc-q8-64k-s$S.out 2>&1
fi
echo "done ctx=$CTX slots=$S" > dc-batch1.done
