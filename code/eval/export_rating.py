"""
Export blind rating sheet for student validation.

Usage::

    python -m code.eval.export_rating --config config/paths.yaml \\
        --judge config/judge.yaml --evalset <csv> --selected <json>
"""

import argparse
import csv
import json
import random
from datetime import date
from pathlib import Path

import yaml

from code.common.run_registry import load_paths


def main():
    ap = argparse.ArgumentParser(description="Export blind rating sheet")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--judge", default="config/judge.yaml")
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--selected", required=True)
    args = ap.parse_args()

    paths = load_paths(args.config)
    with open(args.judge) as f:
        judge_cfg = yaml.safe_load(f)
    with open(args.selected) as f:
        selected = json.load(f)
    with open(args.evalset, newline="", encoding="utf-8") as f:
        questions = {r["question_id"]: r for r in csv.DictReader(f)}

    n_sample = judge_cfg["rated_sample_per_config"]
    seed = judge_cfg["rated_sample_seed"]
    runs_dir = Path(paths["runs"])
    evalset_dir = Path(paths["evalset"])
    evalset_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for arch, info in sorted(selected.items()):
        config_name = info["config_name"]
        # Find generate run
        gen_dir = None
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir() and "generate" in d.name:
                cp = d / "config.json"
                if cp.exists():
                    with open(cp) as f:
                        if json.load(f).get("config_name") == config_name:
                            gen_dir = d

        if not gen_dir or not (gen_dir / "answers.jsonl").exists():
            continue

        with open(gen_dir / "answers.jsonl", encoding="utf-8") as f:
            answers = [json.loads(l) for l in f]

        # Stratified sample by tier
        rng = random.Random(seed)
        by_tier = {}
        for a in answers:
            q = questions.get(a["question_id"], {})
            tier = q.get("tier", "")
            by_tier.setdefault(tier, []).append(a)

        sampled = []
        remaining = n_sample
        tiers = sorted(by_tier.keys())
        per_tier = max(1, n_sample // len(tiers)) if tiers else 0
        for tier in tiers:
            pool = by_tier[tier]
            n = min(per_tier, len(pool), remaining)
            sampled.extend(rng.sample(pool, n))
            remaining -= n

        for a in sampled:
            q = questions.get(a["question_id"], {})
            rows.append({
                "question_id": a["question_id"],
                "config": config_name,
                "tier": q.get("tier", ""),
                "question": q.get("question", ""),
                "context": "\n".join(a.get("labels", [])),
                "answer": a["answer"],
                "reference_answer": q.get("reference_answer", ""),
                "student_faithfulness": "",
                "student_accuracy": "",
                "student_copies_or_answers": "",
                "notes": "",
            })

    today = date.today().isoformat()
    out_path = evalset_dir / f"{today}_rating_sheet_v0.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "config", "tier", "question", "context", "answer",
            "reference_answer", "student_faithfulness", "student_accuracy",
            "student_copies_or_answers", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rating sheet: {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
