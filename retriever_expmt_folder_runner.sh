#!/bin/bash
# run for each file in a directory
RUN_DIR="$1"
echo "running retriever fine-tuning experiment with all files in ${RUN_DIR} :"
for f in $(find $RUN_DIR -name '*.json')
do
  echo "- $f"
done

for f in $(find $RUN_DIR -name '*.json')
do
  CUDA_VISIBLE_DEVICES=0 python src/refpydst/retriever/code/retriever_finetuning.py "$f"
done