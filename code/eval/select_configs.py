"""
Select best configuration per architecture level (D49).

Usage::

    python -m code.eval.select_configs --summary <csv> --out <json>
"""

import argparse
import csv
import json
from datetime import date
from pathlib import Path


def select(summary_path: str) -> dict:
    """Return best config per architecture level by budget_640_mean, ties by recall_10_mean."""
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Classify architecture from config name
    arch_map = {}
    for r in rows:
        name = r["config_name"]
        if "linkexp" in name:
            arch_map[name] = "linkexp"
        elif "pcreturn" in name:
            arch_map[name] = "pcreturn"
        else:
            arch_map[name] = "flat"

    best = {}
    for name, arch in arch_map.items():
        r = next(row for row in rows if row["config_name"] == name)
        budget = float(r["budget_640_mean"])
        recall = float(r["recall_10_mean"])
        key = (-budget, -recall, name)
        if arch not in best or key < best[arch][0]:
            best[arch] = (key, name, budget, recall)

    result = {}
    for arch in ["flat", "pcreturn", "linkexp"]:
        if arch in best:
            _, name, budget, recall = best[arch]
            result[arch] = {
                "config_name": name,
                "budget_640_mean": budget,
                "recall_10_mean": recall,
            }
    return result


def main():
    ap = argparse.ArgumentParser(description="Select best configs per architecture")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = select(args.summary)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))

    for arch, info in sorted(result.items()):
        print(f"  {arch}: {info['config_name']} "
              f"(budget={info['budget_640_mean']:.3f}, R@10={info['recall_10_mean']:.3f})")


if __name__ == "__main__":
    main()
