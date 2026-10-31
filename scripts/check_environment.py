#!/usr/bin/env python3
"""Check that the core scientific Python dependencies are available."""

from __future__ import annotations

import importlib

REQUIRED = [
    "numpy",
    "scipy",
    "networkx",
    "torch",
    "transformers",
]

missing = []
for module in REQUIRED:
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        missing.append(module)

if missing:
    print("Missing dependencies:")
    for item in missing:
        print(f"- {item}")
    raise SystemExit(1)

print("All required dependencies are installed and ready.")
