"""
Inter-rater agreement between student ratings and judge scores.

Usage::

    python -m code.eval.agreement --ratings <csv> --judgements <folder1> <folder2> ...
"""

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np


def cohens_kappa_quadratic(y1, y2, n_categories=3):
    """Quadratic-weighted Cohen's kappa for ordinal ratings."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y1, y2, weights="quadratic")


def main():
    ap = argparse.ArgumentParser(description="Agreement analysis")
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--judgements", nargs="+", required=True)
    ap.add_argument("--out", default="data/results")
    args = ap.parse_args()

    # Load student ratings
    with open(args.ratings, newline="", encoding="utf-8") as f:
        ratings = {(r["question_id"], r["config"]): r for r in csv.DictReader(f)}

    # Load judge scores
    judge_scores = {}
    for jdir in args.judgements:
        p = Path(jdir) / "judgements.jsonl"
        if not p.exists():
            continue
        cp = Path(jdir) / "config.json"
        config_name = ""
        if cp.exists():
            with open(cp) as f:
                config_name = json.load(f).get("config_name", "")
        with open(p, encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                judge_scores[(j["question_id"], config_name)] = j

    result = {"note": "agreement analysis placeholder — fill after student ratings"}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    out_path = out_dir / f"{today}_agreement.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Agreement: {out_path}")


if __name__ == "__main__":
    main()
