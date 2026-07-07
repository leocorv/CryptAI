#!/usr/bin/env bash
# CryptAI Batch Launcher v2 — thermal-aware
BASE="/mnt/hive_storage/CryptAI"
VENV="$BASE/venv/bin/python3"
LOG="$BASE/training/logs"
THERMAL="/mnt/hive_storage/thermal_guard"
STATE="$THERMAL/thermal_state.json"
PAUSE="$THERMAL/pause_training.flag"
RESUME="$THERMAL/resume_training.flag"

check_thermal() {
  # Pause flag → stop
  if [ -f "$PAUSE" ]; then
    echo "[$(date)] THERMAL PAUSE — waiting for cooldown" >> "$LOG/batch_loop.log"
    return 1
  fi

  # Resume flag → go
  if [ -f "$RESUME" ]; then
    echo "[$(date)] THERMAL RESUME — relancing training" >> "$LOG/batch_loop.log"
    rm -f "$RESUME"
    return 0
  fi

  # State file
  if [ -f "$STATE" ]; then
    LEVEL=$($VENV -c "import json; print(json.load(open('$STATE'))['level'])" 2>/dev/null)
    case "$LEVEL" in
      emergency)
        echo "[$(date)] THERMAL EMERGENCY — skip" >> "$LOG/batch_loop.log"
        return 1
        ;;
      heavy|moderate)
        echo "[$(date)] THERMAL $LEVEL — training with reduced batch" >> "$LOG/batch_loop.log"
        return 0
        ;;
      full)
        return 0
        ;;
    esac
  fi
  return 0
}

while true; do
  for model in transformer tcn lstm; do
    if ! check_thermal; then
      sleep 60
      continue 2
    fi
    echo "[$(date)] Starting $model" >> "$LOG/batch_loop.log"
    $VENV $BASE/src/training/train_clean.py $model > "$LOG/${model}_full.log" 2>&1
    echo "[$(date)] $model done" >> "$LOG/batch_loop.log"
  done
  echo "[$(date)] Cycle complete — restarting" >> "$LOG/batch_loop.log"
done