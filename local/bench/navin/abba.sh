#!/bin/bash
UP=/home/god/dev/llama.cpp-bellsup/build/bin/llama-server
OURS=/home/god/dev/llama.cpp-qsa/build/bin/llama-server
export CTX=131072 TS=11,39
for a in "up1 $UP" "ours1 $OURS" "ours2 $OURS" "up2 $UP"; do
  set -- $a; echo "=== $1"; ./navarm.sh $1 $2; echo "=== $1 exit $?"; sleep 5
done
echo ABBA-DONE
