#!/usr/bin/env python3
"""Compatibility module for ``scripts.train``."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import *  # noqa: F401,F403


if __name__ == "__main__":
    train_champion()
