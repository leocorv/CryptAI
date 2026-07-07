#!/usr/bin/env bash
# CryptAI Batch Launcher — trains Transformer → TCN → LSTM → loops
BASE="/mnt/hive_storage/CryptAI"
VENV="$BASE/venv/bin/python3"
LOG="$BASE/training/logs"

while true; do
  for model in transformer tcn lstm; do
    echo "[batch] Starting $model at $(date)"
    $VENV $BASE/src/training/train_clean_v4.py $model > $LOG/${model}_full.log 2>&1
    echo "[batch] $model done: $(tail -1 $LOG/${model}_full.log)"
  done
  echo "[batch] Cycle complete at $(date) — restarting"
done