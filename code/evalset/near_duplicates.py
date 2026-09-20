"""Find near-duplicate question pairs in a candidate evalset (T2c-2, D57).

Usage::

    python -m code.evalset.near_duplicates \\
        --config config/paths.yaml \\
        --questions <candidate csv> \\
        --out evalset/<date>_evalset_neardup_v1.csv

Output: the union of
  - pairs with cosine >= 0.90
  - the 20 most similar pairs
  - pairs with an identical gold_paragraph_ids string

No corpus text is printed.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from code.common.run_registry import load_paths, start_run

# ── constants ────────────────────────────────────────────────────────

COSINE_THRESHOLD = 0.90
TOP_K = 20

OUT_COLUMNS = [
    "pair_id", "qid_a", "qid_b", "cosine", "same_gold",
    "reason_set", "route_a", "route_b", "tier_a", "tier_b",
    "question_a", "question_b", "verdict", "reworded_question_b", "note",
]


# ── helpers ──────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix for L2-normalised embeddings."""
    # embeddings are already normalised by encode_query, so dot == cosine
    return embeddings @ embeddings.T


def find_pairs(
    questions: list[dict],
    cosine_mat: np.ndarray,
) -> tuple[list[dict], dict]:
    """Identify the union of threshold / top-20 / same-gold pair sets.

    Returns (pair_rows, report) where pair_rows are dicts with OUT_COLUMNS
    and report gives the set sizes.
    """
    n = len(questions)

    # ── build indexed lookups ────────────────────────────────────────
    # Upper-triangle pairs only (i < j)
    pair_cosines: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_cosines.append((i, j, float(cosine_mat[i, j])))

    # ── set 1: cosine >= threshold ───────────────────────────────────
    threshold_set: set[tuple[int, int]] = set()
    for i, j, c in pair_cosines:
        if c >= COSINE_THRESHOLD:
            threshold_set.add((i, j))

    # ── set 2: top 20 most similar ───────────────────────────────────
    pair_cosines.sort(key=lambda x: x[2], reverse=True)
    top20_set: set[tuple[int, int]] = set()
    for i, j, c in pair_cosines[:TOP_K]:
        top20_set.add((i, j))

    # ── set 3: identical gold_paragraph_ids string ───────────────────
    same_gold_set: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if questions[i]["gold_paragraph_ids"] == questions[j]["gold_paragraph_ids"]:
                same_gold_set.add((i, j))

    # ── union ────────────────────────────────────────────────────────
    union = threshold_set | top20_set | same_gold_set

    # ── build output rows ────────────────────────────────────────────
    rows: list[dict] = []
    for idx, (i, j) in enumerate(sorted(union), start=1):
        qi, qj = questions[i], questions[j]
        cos_val = float(cosine_mat[i, j])
        sg = qi["gold_paragraph_ids"] == qj["gold_paragraph_ids"]

        reasons = []
        if (i, j) in threshold_set:
            reasons.append("threshold")
        if (i, j) in top20_set:
            reasons.append("top20")
        if (i, j) in same_gold_set:
            reasons.append("same_gold")

        rows.append({
            "pair_id": f"P{idx:04d}",
            "qid_a": qi["question_id"],
            "qid_b": qj["question_id"],
            "cosine": f"{cos_val:.6f}",
            "same_gold": "Y" if sg else "N",
            "reason_set": "+".join(reasons),
            "route_a": qi["route"],
            "route_b": qj["route"],
            "tier_a": qi["tier"],
            "tier_b": qj["tier"],
            "question_a": qi["question"],
            "question_b": qj["question"],
            "verdict": "",
            "reworded_question_b": "",
            "note": "",
        })

    report = {
        "threshold_count": len(threshold_set),
        "top20_count": len(top20_set),
        "same_gold_count": len(same_gold_set),
        "union_count": len(union),
    }

    # top-20 cosine range
    if pair_cosines:
        top_slice = pair_cosines[:TOP_K]
        report["top20_cosine_max"] = f"{top_slice[0][2]:.6f}"
        report["top20_cosine_min"] = f"{top_slice[-1][2]:.6f}"

    return rows, report


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Near-duplicate detection for candidate evalset (T2c-2)")
    ap.add_argument("--config", default="config/paths.yaml",
                    help="paths.yaml config file")
    ap.add_argument("--questions", required=True,
                    help="Candidate questions CSV")
    ap.add_argument("--out", required=True,
                    help="Output near-duplicates CSV")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5",
                    help="Embedding model name")
    args = ap.parse_args(argv)

    paths = load_paths(args.config)
    q_path = Path(args.questions)
    out_path = Path(args.out)

    if not q_path.exists():
        print(f"ERROR: questions file not found: {q_path}", file=sys.stderr)
        return 1

    # ── read questions ───────────────────────────────────────────────
    with q_path.open(encoding="utf-8", newline="") as f:
        questions = list(csv.DictReader(f))

    # ── register run ─────────────────────────────────────────────────
    params = {
        "questions": args.questions,
        "questions_sha256": _sha256(q_path),
        "model": args.model,
        "cosine_threshold": COSINE_THRESHOLD,
        "top_k": TOP_K,
        "n_questions": len(questions),
    }
    ctx = start_run("evalset", "neardup-v1", params, paths)
    ctx.log(f"questions: {len(questions)} rows from {q_path}")

    # ── embed ────────────────────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    from code.index.embed import encode_query

    device = "cpu"
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    ctx.log(f"loading model {args.model} on {device}")
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = 512

    texts = [q["question"] for q in questions]
    embeddings = np.array(
        [encode_query(model, t) for t in texts], dtype=np.float32
    )
    ctx.log(f"embedded {len(texts)} questions, shape {embeddings.shape}")

    # ── compute pairs ────────────────────────────────────────────────
    cosine_mat = compute_cosine_matrix(embeddings)
    pair_rows, report = find_pairs(questions, cosine_mat)

    # ── write output atomically ──────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(pair_rows)
    tmp.replace(out_path)

    output_sha = _sha256(out_path)
    report["output_sha256"] = output_sha
    report["questions_sha256"] = params["questions_sha256"]

    # ── manifest ─────────────────────────────────────────────────────
    manifest_path = Path(str(out_path).replace(".csv", "_manifest.json"))
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx.log(f"wrote {len(pair_rows)} pairs → {out_path}")
    ctx.finish("ok", f"{len(pair_rows)} near-duplicate pairs", metrics=report)

    # ── stdout: counts and paths, never corpus text ──────────────────
    print(f"pairs with cosine >= {COSINE_THRESHOLD}: {report['threshold_count']}")
    print(f"top-20 pairs cosine range: "
          f"{report.get('top20_cosine_min', 'n/a')} – "
          f"{report.get('top20_cosine_max', 'n/a')}")
    print(f"same-gold pairs: {report['same_gold_count']}")
    print(f"union size: {report['union_count']}")
    print(f"output: {out_path}")
    print(f"output sha256: {output_sha}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
