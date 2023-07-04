#!/bin/bash
# run for each file in a directory
RUN_DIR="$1"
echo "running codex (mw21 sim) experiment with all files in ${RUN_DIR} :"
for f in $(find $RUN_DIR -name '*.json')
do
  echo "- $f"
done

for f in $(find $RUN_DIR -name '*.json')
do
  python src/refpydst/simulate_codex_mw21_experiment.py "$f"
done