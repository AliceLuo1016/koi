#!/usr/bin/env python3
"""Standalone runner for koi CLI."""

import sys
import os
from pathlib import Path

# Add src to path. Resolve symlinks so this works from ~/.local/bin/koi.
src_dir = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src_dir))

# Import and run CLI
from koi.cli import main

if __name__ == "__main__":
    main()