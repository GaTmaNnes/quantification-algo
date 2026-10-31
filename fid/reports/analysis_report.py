"""Command line entry points for FID."""

from __future__ import annotations

import argparse

from fid.analysis.redundancy_detector import RedundancyDetector
from fid.core.model_loader import ModelLoader
from fid.reports.analysis_report import AnalysisReport


def main() -> None:
    parser = argparse.ArgumentParser(description="FID - Functional Information Deduplication")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="Hugging Face model identifier")
    parser.add_argument("--revision", default="main", help="Model revision or branch")
    parser.add_argument("--device", default="cpu", help="Torch device: cpu or cuda")
    parser.add_argument("--limit", type=int, default=10, help="Number of tensors to print in the summary")
    parser.add_argument("--top-similar", type=int, default=5, help="Number of similar module pairs to print")
    args = parser.parse_args()

    print("FID - Functional Information Deduplication")
    print(f"Loading model: {args.model} (rev={args.revision}, device={args.device})")

    loader = ModelLoader(args.model, revision=args.revision, device=args.device)
    spec = loader.load()

    inventory = spec.inventory
    if inventory is None:
        raise RuntimeError("Tensor inventory was not created. The model could not be loaded properly.")

    summary = {
        "model": spec.name,
        "device": spec.device,
        "parameters": spec.metadata.get("num_parameters", 0),
        "inventory_count": inventory.count(),
    }

    print("Model loaded successfully.")
    print(f"Kind: {spec.kind}")
    print(f"Parameters: {summary['parameters']}")
    print(f"Tensor inventory entries: {inventory.count()}")
    print(f"Modules discovered: {len(loader.list_modules())}")
    print("Sample tensors:")
    for record in inventory.as_list()[: args.limit]:
        print(f"- {record.name} | module={record.module} | shape={record.shape} | dtype={record.dtype}")

    detector = RedundancyDetector()
    fingerprints = detector.fingerprints_from_model(spec.model)
    if fingerprints:
        print("\nFingerprint summary:")
        for fp in fingerprints[: args.limit]:
            print(
                f"- {fp.name} | module={fp.module} | mean={fp.mean:.5f} | "
                f"std={fp.std:.5f} | l1={fp.l1_norm:.5f} | sparsity={fp.sparsity:.3f}"
            )

        similar = detector.detect_similar_modules(spec.model, limit=args.top_similar)
        moe_candidates = detector.detect_moe_experts(spec.model, limit=args.top_similar)
        report = AnalysisReport()
        rendered = report.render(summary, similar)
        print("\n" + rendered)
        if moe_candidates:
            print("\nPossible MoE expert candidates:")
            for candidate in moe_candidates:
                print(
                    f"- {candidate['left_expert']} vs {candidate['right_expert']} "
                    f"| score={candidate['score']:.4f}"
                )
    else:
        print("No parameter fingerprints could be computed for this model.")


if __name__ == "__main__":
    main()
