#!/usr/bin/env python3
"""Quick example for the FID analysis workflow."""

from __future__ import annotations

from fid.core.model_loader import ModelLoader


def main() -> None:
    model_name = "sshleifer/tiny-gpt2"
    loader = ModelLoader(model_name, device="cpu")
    spec = loader.load()
    inventory = spec.inventory

    print(f"Model: {spec.name}")
    print(f"Inventory entries: {inventory.count() if inventory else 0}")
    if inventory is not None:
        for record in inventory.as_list()[:5]:
            print(f"- {record.name} | {record.module} | {record.shape}")


if __name__ == "__main__":
    main()
