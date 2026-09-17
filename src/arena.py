#!/usr/bin/env python3
"""Compatibility module for the canonical CryptAI arena implementation in ``lib/``."""

import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parents[1] / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from arena import *  # noqa: F401,F403
