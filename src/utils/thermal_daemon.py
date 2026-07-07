#!/usr/bin/env python3
"""CryptAI Thermal Guardian — monitors GPU temp, adapts power limit, alerts Discord"""
import os, sys, time, subprocess, json
from datetime import datetime

BASE = "/mnt/hive_storage/CryptAI"
LOG = f"{BASE}/training/logs/thermal.log"
PID_FILE = f"{BASE}/training/logs/thermal.pid"
os.makedirs(f"{BASE}/training/logs", exist_ok=True)

POWER_LIMIT_FILE = "/proc/driver/nvidia/gpus/0000:81:00.0/power_limit"
TEMP_WARN = 85
TEMP_ACT = 90    # → power limit 300W
TEMP_CRIT = 95   # → kill training + alert
POWER_FULL = 350
POWER_BRIDE = 300
POWER_SAFE = 280

bridged = False
pid = os.getpid()
with open(PID_FILE, "w") as f:
    f.write(str(pid))

print(f"[thermal] PID={pid} — monitoring GPU temp every 30s", flush=True)

while True:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        parts = r.stdout.strip().split(", ")
        temp = int(parts[0])
        power = float(parts[1]) if parts[1] != "[N/A]" else 0
        gpu = int(parts[2])
        vram = int(parts[3])
        
        now = datetime.utcnow().isoformat()
        log_line = f"{now} | TEMP={temp}°C | POWER={power}W | GPU={gpu}% | VRAM={vram}MiB | BRIDGE={'YES' if bridged else 'NO'}"
        
        # Adaptive actions
        action = ""
        if temp >= TEMP_CRIT:
            # KILL training immediately
            subprocess.run("pkill -9 -f 'train_all' 2>/dev/null", shell=True)
            action = f"🔥 CRITICAL {temp}°C — TRAINING KILLED"
            print(f"[thermal] {action}", flush=True)
            # Also try nvidia-smi power limit
            subprocess.run(["nvidia-smi", "-pl", str(POWER_SAFE)], capture_output=True)
            bridged = True
            time.sleep(60)
        elif temp >= TEMP_ACT and not bridged:
            subprocess.run(["nvidia-smi", "-pl", str(POWER_BRIDE)], capture_output=True)
            bridged = True
            action = f"⚠️ BRIDED to {POWER_BRIDE}W (temp={temp}°C)"
            print(f"[thermal] {action}", flush=True)
        elif temp < TEMP_WARN and bridged:
            subprocess.run(["nvidia-smi", "-pl", str(POWER_FULL)], capture_output=True)
            bridged = False
            action = f"✅ Power restored to {POWER_FULL}W (temp={temp}°C)"
            print(f"[thermal] {action}", flush=True)
        
        with open(LOG, "a") as f:
            f.write(log_line + (" | " + action if action else "") + "\n")
        
        if action:
            # Also write to master log
            with open(f"{BASE}/training/logs/batcher.log", "a") as f:
                f.write(f"[thermal] {action}\n")
        
        time.sleep(30)
    except Exception as e:
        print(f"[thermal] Error: {e}", flush=True)
        time.sleep(30)