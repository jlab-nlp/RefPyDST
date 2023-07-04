#!/bin/bash
# run for each file in a directory
RUN_DIR="$1"
echo "running codex experiment with all files in ${RUN_DIR} :"
for f in $(find $RUN_DIR -name '*.json')
do
  echo "- $f"
done

for f in $(find $RUN_DIR -name '*.json')
do
  python src/refpydst/run_codex_experiment.py "$f"
done