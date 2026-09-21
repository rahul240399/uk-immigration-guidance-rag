"""
Inter-rater agreement between student hand ratings and judge scores (T6b-4, D54).

Accuracy: Cohen's kappa with quadratic weights on levels [0, 0.5, 1].
Faithfulness: unweighted kappa between faithfulness_hand (Y/N) and
  judge_binarised (share < 1.0 → Y, else N).

Also reports: exact agreement, F1 for accuracy == 1, confusion matrices;
per configuration and pooled; null judge scores excluded and counted.

Compare each kappa with agreement.threshold (0.60); below → unvalidated: true
for that score.

Usage::

    python -m code.eval.agreement --config config/paths.yaml \\
        --judge config/judge.yaml --sheet <filled csv> --key <key csv> \\
        --judgements <folder1> <folder2> <folder3>
"""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from code.common.run_registry import load_paths, start_run


# ── pure metric functions (testable) ─────────────────────────────────

def _to_str_labels(values: list) -> list[str]:
    """Convert numeric accuracy levels to strings for sklearn compatibility."""
    return [str(v) for v in values]


def quadratic_kappa(y_hand: list, y_judge: list,
                    labels: list | None = None) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal accuracy levels."""
    str_hand = _to_str_labels(y_hand)
    str_judge = _to_str_labels(y_judge)
    str_labels = _to_str_labels(labels) if labels else None
    return float(cohen_kappa_score(str_hand, str_judge, weights="quadratic",
                                   labels=str_labels))


def unweighted_kappa(y_hand: list, y_judge: list,
                     labels: list | None = None) -> float:
    """Unweighted Cohen's kappa for binary faithfulness."""
    return float(cohen_kappa_score(y_hand, y_judge, weights=None,
                                   labels=labels))


def exact_agreement(y1: list, y2: list) -> float:
    """Fraction of pairs where y1[i] == y2[i]."""
    if not y1:
        return 0.0
    return sum(a == b for a, b in zip(y1, y2)) / len(y1)


def f1_for_class(y_true: list, y_pred: list, pos_label) -> float:
    """F1 score for a specific positive class."""
    binary_true = [1 if y == pos_label else 0 for y in y_true]
    binary_pred = [1 if y == pos_label else 0 for y in y_pred]
    return float(f1_score(binary_true, binary_pred, zero_division=0.0))


def make_confusion(y_true: list, y_pred: list,
                   labels: list) -> list[list[int]]:
    """Confusion matrix as nested list (rows=true, cols=pred)."""
    str_true = _to_str_labels(y_true)
    str_pred = _to_str_labels(y_pred)
    str_labels = _to_str_labels(labels)
    cm = confusion_matrix(str_true, str_pred, labels=str_labels)
    return cm.tolist()


def binarise_faithfulness(judge_score: float | None) -> str | None:
    """Judge faithfulness share → Y (unsupported present) / N.

    share < 1.0 → Y (there is an unsupported claim); else N.
    None → None (excluded).
    """
    if judge_score is None:
        return None
    return "Y" if judge_score < 1.0 else "N"


# ── agreement computation ────────────────────────────────────────────

def compute_agreement(
    pairs: list[dict],
    threshold: float,
) -> dict:
    """Compute agreement metrics for a set of (hand, judge) pairs.

    Each pair dict has: accuracy_hand, accuracy_judge, faithfulness_hand,
    faithfulness_judge (the raw share).

    Returns a dict with accuracy and faithfulness sub-dicts.
    """
    ACC_LABELS = [0, 0.5, 1]
    FAITH_LABELS = ["N", "Y"]

    # ── accuracy ─────────────────────────────────────────────────────
    acc_hand = []
    acc_judge = []
    acc_null = 0
    for p in pairs:
        h = p["accuracy_hand"]
        j = p["accuracy_judge"]
        if j is None:
            acc_null += 1
            continue
        acc_hand.append(h)
        acc_judge.append(j)

    acc_result = {"n": len(acc_hand), "null_excluded": acc_null}
    if len(acc_hand) >= 2:
        kappa = quadratic_kappa(acc_hand, acc_judge, labels=ACC_LABELS)
        acc_result["kappa"] = round(kappa, 4)
        acc_result["exact_agreement"] = round(
            exact_agreement(acc_hand, acc_judge), 4)
        acc_result["f1_accuracy_1"] = round(
            f1_for_class(acc_hand, acc_judge, pos_label=1), 4)
        acc_result["confusion_matrix"] = make_confusion(
            acc_hand, acc_judge, labels=ACC_LABELS)
        acc_result["unvalidated"] = kappa < threshold
    else:
        acc_result["kappa"] = None
        acc_result["unvalidated"] = True

    # ── faithfulness ─────────────────────────────────────────────────
    faith_hand = []
    faith_judge = []
    faith_null = 0
    for p in pairs:
        h = p["faithfulness_hand"]
        j_bin = binarise_faithfulness(p["faithfulness_judge"])
        if j_bin is None:
            faith_null += 1
            continue
        faith_hand.append(h)
        faith_judge.append(j_bin)

    faith_result = {"n": len(faith_hand), "null_excluded": faith_null}
    if len(faith_hand) >= 2:
        kappa = unweighted_kappa(faith_hand, faith_judge, labels=FAITH_LABELS)
        faith_result["kappa"] = round(kappa, 4)
        faith_result["exact_agreement"] = round(
            exact_agreement(faith_hand, faith_judge), 4)
        faith_result["confusion_matrix"] = make_confusion(
            faith_hand, faith_judge, labels=FAITH_LABELS)
        faith_result["unvalidated"] = kappa < threshold
    else:
        faith_result["kappa"] = None
        faith_result["unvalidated"] = True

    return {"accuracy": acc_result, "faithfulness": faith_result}


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Agreement analysis (T6b-4)")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--judge", default="config/judge.yaml")
    ap.add_argument("--sheet", required=True, help="Filled rating sheet CSV")
    ap.add_argument("--key", required=True, help="Key file CSV")
    ap.add_argument("--judgements", nargs="+", required=True,
                    help="Judge run folders")
    args = ap.parse_args(argv)

    paths = load_paths(args.config)
    with open(args.judge) as f:
        judge_cfg = yaml.safe_load(f)

    threshold = judge_cfg.get("agreement", {}).get("threshold", 0.60)

    # ── load sheet + key ─────────────────────────────────────────────
    with open(args.sheet, newline="", encoding="utf-8") as f:
        sheet_rows = list(csv.DictReader(f))
    with open(args.key, newline="", encoding="utf-8") as f:
        key_rows = list(csv.DictReader(f))

    key_map = {int(k["sheet_row"]): k for k in key_rows}

    # ── load judgements by (question_id, config) ─────────────────────
    judge_scores: dict[tuple[str, str], dict] = {}
    for jdir in args.judgements:
        jpath = Path(jdir)
        cp = jpath / "config.json"
        config_name = ""
        if cp.exists():
            with open(cp) as f:
                config_name = json.load(f).get("config_name", "")
        jp = jpath / "judgements.jsonl"
        if jp.exists():
            with open(jp, encoding="utf-8") as f:
                for line in f:
                    j = json.loads(line)
                    judge_scores[(j["question_id"], config_name)] = j

    # ── build pairs ──────────────────────────────────────────────────
    all_pairs: list[dict] = []
    by_config: dict[str, list[dict]] = {}

    for s in sheet_rows:
        sr = int(s["sheet_row"])
        k = key_map.get(sr)
        if not k:
            continue
        qid = k["question_id"]
        config = k["config"]
        j = judge_scores.get((qid, config), {})

        # Parse hand ratings
        acc_hand_str = s.get("accuracy_hand", "").strip()
        faith_hand = s.get("faithfulness_hand", "").strip()

        if not acc_hand_str or not faith_hand:
            continue

        acc_hand = float(acc_hand_str)
        acc_judge = j.get("accuracy")
        faith_judge = j.get("faithfulness")

        pair = {
            "question_id": qid,
            "config": config,
            "accuracy_hand": acc_hand,
            "accuracy_judge": acc_judge,
            "faithfulness_hand": faith_hand,
            "faithfulness_judge": faith_judge,
        }
        all_pairs.append(pair)
        by_config.setdefault(config, []).append(pair)

    # ── compute per config and pooled ────────────────────────────────
    results: dict = {"threshold": threshold, "per_config": {}, "pooled": {}}

    for config, pairs in sorted(by_config.items()):
        results["per_config"][config] = compute_agreement(pairs, threshold)

    results["pooled"] = compute_agreement(all_pairs, threshold)

    # ── register run and write output ────────────────────────────────
    ctx = start_run("agreement", "agreement-v1", {
        "sheet": args.sheet,
        "key": args.key,
        "judgement_folders": args.judgements,
        "threshold": threshold,
        "n_pairs": len(all_pairs),
    }, paths)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    results_dir = Path(paths["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{stamp}_agreement.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    ctx.log(f"pairs={len(all_pairs)} configs={len(by_config)}")
    ctx.finish("ok", f"agreement: {len(all_pairs)} pairs, "
               f"{len(by_config)} configs")

    # ── report ───────────────────────────────────────────────────────
    print(f"pairs: {len(all_pairs)}")
    pooled = results["pooled"]
    acc = pooled["accuracy"]
    faith = pooled["faithfulness"]
    print(f"pooled accuracy:     kappa={acc.get('kappa')} "
          f"exact={acc.get('exact_agreement')} "
          f"f1@1={acc.get('f1_accuracy_1')} "
          f"unvalidated={acc.get('unvalidated')}")
    print(f"pooled faithfulness: kappa={faith.get('kappa')} "
          f"exact={faith.get('exact_agreement')} "
          f"unvalidated={faith.get('unvalidated')}")
    for config, r in sorted(results["per_config"].items()):
        a = r["accuracy"]
        ff = r["faithfulness"]
        print(f"  {config}: acc_kappa={a.get('kappa')} "
              f"faith_kappa={ff.get('kappa')}")
    print(f"output: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
