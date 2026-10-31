# FID — Functional Information Deduplication

A small research prototype to inventory tensor parameters, compute simple weight fingerprints and compare modules for potential redundancy, with a first focus on Hugging Face models and MoE-style experts.

## Goals

- Load a real model from Hugging Face on CPU.
- Inventory parameters and buffers automatically.
- Compute lightweight fingerprints across tensors.
- Detect similar modules and expert-like patterns.
- Expose a fast CLI for local experimentation.

## Project structure

```text
quantification-algo/
├── fid/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── tensor_inventory.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── functional_similarity.py
│   │   ├── redundancy_detector.py
│   │   └── weight_fingerprint.py
│   └── reports/
│       └── analysis_report.py
├── scripts/
│   ├── check_environment.py
│   ├── run_cpu_check.py
│   ├── run_demo.py
│   └── setup.sh
├── tests/
│   ├── test_tensor_inventory.py
│   └── test_redundancy_detector.py
├── examples/
│   └── quickstart.py
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick validation

```bash
python scripts/check_environment.py
pytest -q
```

## Real model analysis

```bash
python -m fid --model sshleifer/tiny-gpt2 --device cpu --limit 10 --top-similar 5
```

This loads a tiny GPT-2 model from Hugging Face, inventories its tensors and prints a short redundancy summary.

## Example usage in Python

```python
from fid.core.model_loader import ModelLoader
from fid.analysis.redundancy_detector import RedundancyDetector

loader = ModelLoader("sshleifer/tiny-gpt2", device="cpu")
spec = loader.load()

detector = RedundancyDetector()
print(spec.inventory.count())
print(detector.fingerprints_from_model(spec.model)[0])
```

## Notes

This is intentionally incremental and research-oriented: the goal is not to perform a full MoE optimization yet, but to provide a robust base for tensor exploration, weight characterization and module comparison.
