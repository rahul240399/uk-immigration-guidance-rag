"""Compute per-question covariates for the candidate evalset (T2c-3, D58).

Usage::

    python -m code.evalset.covariates \\
        --config config/paths.yaml \\
        --questions <candidate csv> \\
        --out evalset/<date>_evalset_covariates_v1.csv

Runs para-bge-bm25-flat retrieval (depth 100) on the candidate questions,
registers the run (stage grid, config para-bge-bm25-flat-covariates), then
writes one row per question_id with anchor-based covariates.

No corpus text is printed.
"""
import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from code.common.run_registry import load_paths, start_run
from code.index.build_index import _tokenize_bm25

# ── constants ────────────────────────────────────────────────────────

BM25_DEPTH = 100

OUT_COLUMNS = [
    "question_id", "anchor_id", "drafting_style", "section_records",
    "siblings_same_heading", "links_out_v2", "n_gold", "link_type",
    "gold_tokens", "bm25_gold_rank", "bm25_overlap", "lexically_easy",
]


# ── helpers ──────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bm25(idx_dir: Path):
    with open(idx_dir / "bm25.pkl", "rb") as f:
        return pickle.load(f)


def load_chunk_ids(idx_dir: Path) -> list[str]:
    """Load chunk_ids in index order (aligns with BM25 internal indices)."""
    ids = []
    with open(idx_dir / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            ids.append(c["chunk_id"])
    return ids


def load_units(index_dir: Path) -> dict[str, dict]:
    """Load units_v1.jsonl as {unit_id: unit_dict}."""
    units = {}
    with open(index_dir / "units_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line.strip())
            units[u["unit_id"]] = u
    return units


def load_covariates(processed_dir: Path) -> dict[str, dict]:
    """Load rules-covariates_v1.csv as {paragraph_id: row_dict}."""
    covs = {}
    with open(processed_dir / "rules-covariates_v1.csv", newline="",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            covs[row["paragraph_id"]] = row
    return covs


def retrieve_bm25(bm25, chunk_ids: list[str], query: str,
                  depth: int) -> list[tuple[str, float]]:
    """BM25 retrieval, matching run_grid.py semantics exactly."""
    scores = bm25.get_scores(_tokenize_bm25(query))
    top = np.argsort(scores)[-depth:][::-1]
    return [(chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]


# ── anchor logic ─────────────────────────────────────────────────────

def find_anchor(gold_ids: list[str], link_type: str,
                units_by_id: dict[str, dict]) -> str | None:
    """Return the anchor paragraph_id for a question.

    - Default: first Rules paragraph_id in gold order.
    - guidance_para: first Rules id after the guidance source
      (i.e. skip guidance ids, take the first rules id).
    """
    if link_type == "guidance_para":
        # Skip guidance ids, take first rules id
        for gid in gold_ids:
            u = units_by_id.get(gid, {})
            if u.get("corpus") == "rules":
                return gid
        # Fallback: first id regardless
        return gold_ids[0] if gold_ids else None
    else:
        # First Rules paragraph_id in gold order
        for gid in gold_ids:
            u = units_by_id.get(gid, {})
            if u.get("corpus") == "rules":
                return gid
        # Fallback if no rules id (shouldn't happen)
        return gold_ids[0] if gold_ids else None


def compute_bm25_overlap(question: str, anchor_text: str) -> float:
    """Share of the question's BM25 tokens present in the anchor unit text.

    Uses the same _tokenize_bm25 rule as the index.
    """
    q_tokens = set(_tokenize_bm25(question))
    if not q_tokens:
        return 0.0
    a_tokens = set(_tokenize_bm25(anchor_text))
    return len(q_tokens & a_tokens) / len(q_tokens)


def compute_bm25_gold_rank(retrieved: list[tuple[str, float]],
                           gold_ids: set[str]) -> str:
    """Rank of the first retrieved unit carrying any gold id.

    Returns the 1-based rank as a string, or "absent" if beyond depth.
    For the para chunker, chunk_id == unit_id == paragraph_id.
    """
    for rank, (cid, _score) in enumerate(retrieved, 1):
        if cid in gold_ids:
            return str(rank)
    return "absent"


# ── main logic (testable) ───────────────────────────────────────────

def build_covariates(
    questions: list[dict],
    bm25,
    chunk_ids: list[str],
    units_by_id: dict[str, dict],
    covariates_by_id: dict[str, dict],
) -> tuple[list[dict], dict]:
    """Compute covariates for each question. Returns (rows, report)."""

    rows: list[dict] = []
    empty_anchors = 0
    easy_by_tier: dict[str, int] = {}
    absent_count = 0

    for q in questions:
        qid = q["question_id"]
        link_type = q.get("link_type") or ""
        tier = q.get("tier", "")
        gold_str = q.get("gold_paragraph_ids", "")
        gold_ids = [g.strip() for g in gold_str.split(";") if g.strip()]
        gold_set = set(gold_ids)

        # Anchor
        anchor_id = find_anchor(gold_ids, link_type, units_by_id)
        if not anchor_id:
            empty_anchors += 1

        # Covariates from rules-covariates_v1.csv
        cov = covariates_by_id.get(anchor_id, {}) if anchor_id else {}
        drafting_style = cov.get("drafting_style", "")
        section_records = cov.get("section_records", "")
        siblings = cov.get("siblings_same_heading", "")
        links_out = cov.get("links_out_v2", "")

        # n_gold
        n_gold = len(gold_ids)

        # gold_tokens: sum of unit_token_count over gold ids
        gold_tokens = 0
        for gid in gold_ids:
            u = units_by_id.get(gid, {})
            gold_tokens += u.get("unit_token_count", 0)

        # BM25 retrieval
        retrieved = retrieve_bm25(bm25, chunk_ids, q["question"], BM25_DEPTH)

        # bm25_gold_rank
        bm25_rank = compute_bm25_gold_rank(retrieved, gold_set)
        if bm25_rank == "absent":
            absent_count += 1

        # bm25_overlap
        anchor_text = ""
        if anchor_id:
            u = units_by_id.get(anchor_id, {})
            anchor_text = u.get("text", "")
        bm25_overlap = compute_bm25_overlap(q["question"], anchor_text)

        # lexically_easy: 1 when first retrieved unit carries a gold id
        lex_easy = 0
        if retrieved and retrieved[0][0] in gold_set:
            lex_easy = 1

        easy_by_tier[tier] = easy_by_tier.get(tier, 0) + lex_easy

        rows.append({
            "question_id": qid,
            "anchor_id": anchor_id or "",
            "drafting_style": drafting_style,
            "section_records": section_records,
            "siblings_same_heading": siblings,
            "links_out_v2": links_out,
            "n_gold": str(n_gold),
            "link_type": link_type,
            "gold_tokens": str(gold_tokens),
            "bm25_gold_rank": bm25_rank,
            "bm25_overlap": f"{bm25_overlap:.4f}",
            "lexically_easy": str(lex_easy),
        })

    report = {
        "rows": len(rows),
        "empty_anchors": empty_anchors,
        "absent_count": absent_count,
        "lexically_easy_per_tier": easy_by_tier,
    }
    return rows, report


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compute per-question covariates (T2c-3)")
    ap.add_argument("--config", default="config/paths.yaml",
                    help="paths.yaml config file")
    ap.add_argument("--questions", required=True,
                    help="Candidate questions CSV")
    ap.add_argument("--out", required=True,
                    help="Output covariates CSV")
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

    # ── load data ────────────────────────────────────────────────────
    index_dir = Path(paths["index"])
    processed_dir = Path(paths["processed"])
    para_dir = index_dir / "para"

    bm25 = load_bm25(para_dir)
    chunk_ids = load_chunk_ids(para_dir)
    units_by_id = load_units(index_dir)
    covariates_by_id = load_covariates(processed_dir)

    # ── register BM25 retrieval run ──────────────────────────────────
    params = {
        "questions": args.questions,
        "questions_sha256": _sha256(q_path),
        "chunker": "para",
        "retrieval": "bm25",
        "architecture": "flat",
        "bm25_depth": BM25_DEPTH,
        "n_questions": len(questions),
    }
    ctx = start_run("grid", "para-bge-bm25-flat-covariates", params, paths)
    ctx.log(f"questions: {len(questions)} rows from {q_path}")

    # ── compute covariates ───────────────────────────────────────────
    rows, report = build_covariates(
        questions, bm25, chunk_ids, units_by_id, covariates_by_id)

    # ── write retrieved.jsonl for the registered run ─────────────────
    retrieved_path = ctx.run_dir / "retrieved.jsonl"
    with retrieved_path.open("w", encoding="utf-8") as f:
        for q in questions:
            retrieved = retrieve_bm25(bm25, chunk_ids, q["question"], BM25_DEPTH)
            f.write(json.dumps({
                "question_id": q["question_id"],
                "retrieved": [{"unit_id": cid, "score": round(s, 6)}
                              for cid, s in retrieved],
            }, ensure_ascii=False) + "\n")

    # ── write output atomically ──────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(out_path)

    output_sha = _sha256(out_path)
    report["output_sha256"] = output_sha
    report["questions_sha256"] = params["questions_sha256"]
    report["run_dir"] = str(ctx.run_dir)

    # ── manifest ─────────────────────────────────────────────────────
    manifest_path = Path(str(out_path).replace(".csv", "_manifest.json"))
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx.log(f"covariates: {report['rows']} rows, "
            f"empty_anchors={report['empty_anchors']}")
    ctx.finish("ok", f"{report['rows']} covariates rows", metrics=report)

    # ── stdout: counts and paths, never corpus text ──────────────────
    print(f"rows: {report['rows']}")
    print(f"empty anchors: {report['empty_anchors']}")
    print(f"bm25_gold_rank absent: {report['absent_count']}")
    print(f"lexically_easy per tier:")
    for tier in sorted(report["lexically_easy_per_tier"]):
        print(f"  {tier}: {report['lexically_easy_per_tier'][tier]}")
    print(f"output: {out_path}")
    print(f"output sha256: {output_sha}")
    print(f"run folder: {ctx.run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
