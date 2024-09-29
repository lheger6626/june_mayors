#!/bin/bash
NUM_PARALLEL_JOBS=10
configs=""
results=""
topic="q68_no_policies/"
find $"$configs$topic" -maxdepth 1 -mindepth 1 | parallel --progress -j${NUM_PARALLEL_JOBS} touch ${results}logs/${topic}{/}.log ";" "python scripts/run_simulation.py \
--path-to-config={} --output-folder=${results}results/$topic{/} > ${results}logs/${topic}{/}.log"
