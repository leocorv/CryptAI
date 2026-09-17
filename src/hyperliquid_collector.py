#!/usr/bin/env python3
"""Compatibility entrypoint for the CryptAI Hyperliquid collector."""

import runpy
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "hyperliquid_collector.py"
runpy.run_path(str(RUNNER), run_name="__main__")
