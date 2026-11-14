#!/bin/sh
# Bootstrap workers: each does at most 5 new (model,seed,condition) units then exits (bounds memory); resumes from data/cache.
for s in 0 1 2; do
  i=0
  while [ $i -lt 40 ]; do
    python -m src.qrd.stats $s 5 >> data/stats_s$s.log 2>&1; rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -eq 3 ] || sleep 20
    i=$((i+1))
  done
done
python -m src.qrd.stats > data/stats_final.log 2>&1
