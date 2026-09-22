"""
Export blind rating sheet and key file for student validation (T6b-3, D55/D56).

One draw of ``rated_sample_per_config`` question ids from the frozen evalset,
stratified by ``rated_by_tier`` (20 / 13 / 17), random within tier with
``rated_sample_seed``; the same ids under each of the three D49 configurations
→ 150 rows.

Sheet columns (blind — no question_id, no configuration, no judge score):
  sheet_row, question, context, answer, reference_answer,
  faithfulness_hand, accuracy_hand, copies_or_answers, notes

Key file (kept out of the sheet the student rates):
  sheet_row, question_id, config, generate_run, judge_run

Usage::

    python -m code.eval.export_rating --config config/paths.yaml \\
        --judge config/judge.yaml --evalset <csv> --selected <json> \\
        --out-sheet <csv> --out-key <csv>
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import yaml

from code.common.run_registry import load_paths

# ── constants ────────────────────────────────────────────────────────

SHEET_COLUMNS = [
    "sheet_row", "question", "context", "answer", "reference_answer",
    "faithfulness_hand", "accuracy_hand", "copies_or_answers", "notes",
]

KEY_COLUMNS = [
    "sheet_row", "question_id", "config", "generate_run", "judge_run",
]

FORBIDDEN_SHEET_COLUMNS = {
    "question_id", "config", "configuration",
    "faithfulness", "accuracy", "judge_faithfulness", "judge_accuracy",
}


# ── stratified draw ─────────────────────────────────────────────────

def draw_question_ids(
    questions: list[dict],
    rated_by_tier: dict[str, int],
    seed: int,
) -> list[str]:
    """Draw question_ids stratified by tier.

    Returns a list of question_ids, random within each tier.
    """
    rng = random.Random(seed)

    by_tier: dict[str, list[str]] = {}
    for q in questions:
        tier = q.get("tier", "")
        by_tier.setdefault(tier, []).append(q["question_id"])

    drawn: list[str] = []
    for tier, n in sorted(rated_by_tier.items()):
        pool = by_tier.get(tier, [])
        if len(pool) < n:
            raise ValueError(
                f"Tier {tier}: need {n} but only {len(pool)} available")
        drawn.extend(rng.sample(pool, n))

    return drawn


# ── sheet + key assembly ─────────────────────────────────────────────

def build_sheet_and_key(
    drawn_ids: list[str],
    configs: dict[str, dict],
    questions_by_id: dict[str, dict],
    answers_by_config: dict[str, dict[str, dict]],
    generate_runs: dict[str, str],
    judge_runs: dict[str, str],
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Build the blind sheet rows and key rows.

    Returns (sheet_rows, key_rows) both sorted by sheet_row.
    """
    # Build one row per (question_id, config)
    pre_rows: list[dict] = []
    for qid in drawn_ids:
        q = questions_by_id.get(qid, {})
        for arch, info in sorted(configs.items()):
            config_name = info["config_name"]
            ans = answers_by_config.get(config_name, {}).get(qid)
            if not ans:
                continue

            # Context = the passages as given to the generator
            # Prefer context_text (full passages) if stored in answers.jsonl;
            # fall back to labels joined (for backward compat)
            context = ans.get("context_text") or "\n\n".join(
                f"{label}" for label in ans.get("labels", [])
            )

            pre_rows.append({
                "question_id": qid,
                "config": config_name,
                "question": q.get("question", ""),
                "context": context,
                "answer": ans.get("answer", ""),
                "reference_answer": q.get("reference_answer", ""),
                "generate_run": generate_runs.get(config_name, ""),
                "judge_run": judge_runs.get(config_name, ""),
            })

    # Shuffle all rows with seed
    rng = random.Random(seed)
    rng.shuffle(pre_rows)

    # Assign sheet_row (1-based)
    sheet_rows: list[dict] = []
    key_rows: list[dict] = []
    for i, r in enumerate(pre_rows, 1):
        sheet_rows.append({
            "sheet_row": i,
            "question": r["question"],
            "context": r["context"],
            "answer": r["answer"],
            "reference_answer": r["reference_answer"],
            "faithfulness_hand": "",
            "accuracy_hand": "",
            "copies_or_answers": "",
            "notes": "",
        })
        key_rows.append({
            "sheet_row": i,
            "question_id": r["question_id"],
            "config": r["config"],
            "generate_run": r["generate_run"],
            "judge_run": r["judge_run"],
        })

    return sheet_rows, key_rows


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export blind rating sheet (T6b-3)")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--judge", default="config/judge.yaml")
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--out-sheet", required=True)
    ap.add_argument("--out-key", required=True)
    args = ap.parse_args(argv)

    paths = load_paths(args.config)
    with open(args.judge) as f:
        judge_cfg = yaml.safe_load(f)
    with open(args.selected) as f:
        selected = json.load(f)
    with open(args.evalset, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    questions_by_id = {q["question_id"]: q for q in questions}

    hand_rating = judge_cfg["hand_rating"]
    rated_by_tier = hand_rating["rated_by_tier"]
    seed = hand_rating.get("rated_sample_seed", 20260918)
    n_sample = hand_rating["rated_sample_per_config"]

    # Validate tier counts sum
    tier_sum = sum(rated_by_tier.values())
    if tier_sum != n_sample:
        print(f"WARNING: rated_by_tier sums to {tier_sum}, "
              f"rated_sample_per_config={n_sample}", file=sys.stderr)

    # ── draw question ids ────────────────────────────────────────────
    drawn_ids = draw_question_ids(questions, rated_by_tier, seed)

    # ── load answers per config ──────────────────────────────────────
    runs_dir = Path(paths["runs"])
    answers_by_config: dict[str, dict[str, dict]] = {}
    generate_runs: dict[str, str] = {}
    judge_runs: dict[str, str] = {}

    for arch, info in selected.items():
        config_name = info["config_name"]

        # Find generate run
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir() and "generate" in d.name:
                cp = d / "config.json"
                if cp.exists():
                    with open(cp) as f:
                        if json.load(f).get("config_name") == config_name:
                            gen_dir = d
                            generate_runs[config_name] = d.name
                            ans_path = d / "answers.jsonl"
                            if ans_path.exists():
                                with open(ans_path, encoding="utf-8") as af:
                                    answers_by_config[config_name] = {
                                        json.loads(l)["question_id"]: json.loads(l)
                                        for l in af
                                    }

        # Find judge run
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir() and "judge" in d.name:
                cp = d / "config.json"
                if cp.exists():
                    with open(cp) as f:
                        if json.load(f).get("config_name") == config_name:
                            judge_runs[config_name] = d.name

    # ── build sheet and key ──────────────────────────────────────────
    sheet_rows, key_rows = build_sheet_and_key(
        drawn_ids, selected, questions_by_id,
        answers_by_config, generate_runs, judge_runs, seed)

    # ── write sheet ──────────────────────────────────────────────────
    sheet_path = Path(args.out_sheet)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = sheet_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(sheet_rows)
    tmp.replace(sheet_path)

    # ── write key ────────────────────────────────────────────────────
    key_path = Path(args.out_key)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = key_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEY_COLUMNS)
        w.writeheader()
        w.writerows(key_rows)
    tmp.replace(key_path)

    # ── report ───────────────────────────────────────────────────────
    configs_with_answers = len(answers_by_config)
    by_tier = {}
    for qid in drawn_ids:
        q = questions_by_id.get(qid, {})
        t = q.get("tier", "?")
        by_tier[t] = by_tier.get(t, 0) + 1

    print(f"drawn question ids: {len(drawn_ids)}")
    print(f"  per tier: {dict(sorted(by_tier.items()))}")
    print(f"configs with answers: {configs_with_answers}")
    print(f"sheet rows: {len(sheet_rows)}")
    print(f"  per config: {len(drawn_ids)} × {configs_with_answers} "
          f"= {len(drawn_ids) * configs_with_answers}")
    print(f"sheet: {sheet_path}")
    print(f"key:   {key_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
