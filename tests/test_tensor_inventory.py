"""Quick example for the FID analysis workflow."""

from __future__ import annotations

from fid.analysis.redundancy_detector import RedundancyDetector
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

    detector = RedundancyDetector()
    fingerprints = detector.fingerprints_from_model(spec.model)
    print(f"Computed {len(fingerprints)} fingerprints.")
    if fingerprints:
        for fp in fingerprints[:3]:
            print(f"{fp.name}: mean={fp.mean:.4f}, std={fp.std:.4f}, l1={fp.l1_norm:.4f}")


if __name__ == "__main__":
    main()
