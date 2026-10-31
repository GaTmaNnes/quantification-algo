#!/usr/bin/env python3
"""Simple CLI entry point for the FID research prototype."""

from __future__ import annotations

from fid.analysis.redundancy_detector import RedundancyDetector
from fid.core.model_loader import ModelLoader


def main() -> None:
    print("FID - Functional Information Deduplication")
    print("Loading a small Hugging Face model for a quick local demo...")

    loader = ModelLoader("sshleifer/tiny-gpt2", device="cpu")
    spec = loader.load()
    inventory = spec.inventory

    if inventory is None:
        raise RuntimeError("Tensor inventory was not created.")

    print(f"Model: {spec.name}")
    print(f"Parameters: {spec.num_parameters}")
    print(f"Tensor inventory entries: {inventory.count()}")

    detector = RedundancyDetector()
    fingerprints = detector.fingerprints_from_model(spec.model)
    print(f"Fingerprints computed: {len(fingerprints)}")

    if fingerprints:
        sample = fingerprints[0]
        print(f"Sample fingerprint: {sample.name} | mean={sample.mean:.5f} | std={sample.std:.5f} | l1={sample.l1_norm:.5f}")

    print("Demo complete.")


if __name__ == "__main__":
    main()
