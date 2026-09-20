"""Merge validated synthetic and hand-written questions into a candidate evalset (T2c-1).

Usage:
    python -m code.evalset.merge_questions \\
        --config config/paths.yaml \\
        --synthetic <validated csv> --hand <hand csv> \\
        --out evalset/<date>_evalset_v1-candidate.csv

Rules:
  - Keep synthetic rows with validated_by_student == Y; report N counts per
    stratum and notes column as counts only.
  - Append hand rows: source must be ``hand``, tier T3, link_type in
    {rules_section_xcut, rules_section_other}, validated Y; reject the file
    otherwise.
  - Assert: every gold id in rules-paragraphs_v1 or guidance-paragraphs_v1 and
    not deleted; question_ids unique; D53 columns exactly; order synthetic by
    question_id then hand by seed order.
  - Manifest: counts per route × tier × link_type and per source; sha256 of
    both inputs and the output.
  - Registers the run through code/common/run_registry.py.

No corpus text is printed.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from code.common.run_registry import load_paths, start_run

# ── constants ────────────────────────────────────────────────────────

COLUMNS = [
    "question_id", "route", "tier", "link_type", "question",
    "reference_answer", "gold_paragraph_ids", "source",
    "generator_model", "embedding_model", "validated_by_student", "notes",
]

VALID_ROUTES = {"skilled_worker", "student", "graduate", "visitor", "family"}
ROUTE_RE = {r: re.compile(r"\b" + re.escape(r) + r"\b") for r in VALID_ROUTES}

HAND_TIERS = {"T3"}
HAND_LINK_TYPES = {"rules_section_xcut", "rules_section_other"}


# ── helpers ──────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_paragraph_ids(processed_dir: Path) -> set[str]:
    """Load the union of paragraph_id from both corpus JSONL files."""
    ids: set[str] = set()
    for name in ("rules-paragraphs_v1.jsonl", "guidance-paragraphs_v1.jsonl"):
        p = processed_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Required corpus file missing: {p}")
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ids.add(rec["paragraph_id"])
    return ids


def _validate_hand(rows: list[dict]) -> list[str]:
    """Return a list of problems if the hand file violates constraints."""
    problems: list[str] = []
    for i, r in enumerate(rows):
        if r.get("source") != "hand":
            problems.append(f"row {i}: source={r.get('source')!r}, expected 'hand'")
        if r.get("tier") not in HAND_TIERS:
            problems.append(f"row {i}: tier={r.get('tier')!r}, expected T3")
        if r.get("link_type") not in HAND_LINK_TYPES:
            problems.append(f"row {i}: link_type={r.get('link_type')!r}, "
                            f"expected one of {HAND_LINK_TYPES}")
        if r.get("validated_by_student") != "Y":
            problems.append(f"row {i}: validated_by_student={r.get('validated_by_student')!r}, "
                            "expected 'Y'")
    return problems


def _validate_gold_ids(rows: list[dict], corpus_ids: set[str]) -> list[str]:
    """Return problems for any gold id not in the corpus."""
    problems: list[str] = []
    for r in rows:
        golds = r.get("gold_paragraph_ids", "")
        if not golds:
            problems.append(f"{r['question_id']}: empty gold_paragraph_ids")
            continue
        for gid in golds.split(";"):
            gid = gid.strip()
            if gid and gid not in corpus_ids:
                problems.append(f"{r['question_id']}: gold id {gid!r} not in corpus")
    return problems


def _validate_route_names(rows: list[dict]) -> list[str]:
    """Return problems for any route not in the valid set (\b boundaries)."""
    problems: list[str] = []
    for r in rows:
        route = r.get("route", "")
        if route not in VALID_ROUTES:
            problems.append(f"{r.get('question_id', '?')}: route {route!r} not a valid route")
    return problems


def _strata_counts(rows: list[dict]) -> dict[str, int]:
    """Counts per route × tier × link_type."""
    counts: dict[str, int] = {}
    for r in rows:
        lt = r.get("link_type") or ""
        key = f"{r['route']}_{r['tier']}_{lt}".rstrip("_")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _source_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        s = r.get("source") or ""
        counts[s] = counts.get(s, 0) + 1
    return counts


# ── main logic (testable) ───────────────────────────────────────────

def merge(synthetic_rows: list[dict], hand_rows: list[dict],
          corpus_ids: set[str]) -> tuple[list[dict], dict]:
    """Merge synthetic + hand rows.  Returns (merged_rows, report_dict).

    Raises ValueError on validation failures.
    """
    report: dict = {
        "synthetic_total": len(synthetic_rows),
        "hand_total": len(hand_rows),
    }

    # ── synthetic: keep Y, report N per stratum ──────────────────────
    kept_syn: list[dict] = []
    rejected_n: dict[str, int] = {}
    notes_counts: dict[str, int] = {}
    for r in synthetic_rows:
        if r.get("validated_by_student") == "Y":
            kept_syn.append(r)
        else:
            lt = r.get("link_type") or ""
            key = f"{r['route']}_{r['tier']}_{lt}".rstrip("_")
            rejected_n[key] = rejected_n.get(key, 0) + 1
        # count notes values (as counts only, never the text)
        note = (r.get("notes") or "").strip()
        if note:
            notes_counts[note] = notes_counts.get(note, 0) + 1

    report["synthetic_Y"] = len(kept_syn)
    report["synthetic_N_per_stratum"] = rejected_n
    report["synthetic_notes_counts"] = {f"<{len(k)} chars>": v
                                        for k, v in notes_counts.items()}

    # ── hand: validate constraints, reject file on any violation ─────
    hand_problems = _validate_hand(hand_rows)
    if hand_problems:
        raise ValueError(
            f"Hand file rejected ({len(hand_problems)} problems): "
            + "; ".join(hand_problems[:5])
            + ("..." if len(hand_problems) > 5 else "")
        )
    report["hand_appended"] = len(hand_rows)

    # ── route name checks (\b boundaries) ────────────────────────────
    route_problems = _validate_route_names(kept_syn + hand_rows)
    if route_problems:
        raise ValueError(
            f"Route validation failed: " + "; ".join(route_problems[:5])
        )

    # ── gold id validation ───────────────────────────────────────────
    gold_problems = _validate_gold_ids(kept_syn + hand_rows, corpus_ids)
    if gold_problems:
        raise ValueError(
            f"Gold-id validation failed ({len(gold_problems)} problems): "
            + "; ".join(gold_problems[:10])
        )

    # ── columns: D53 exactly ─────────────────────────────────────────
    for label, rows in [("synthetic", kept_syn), ("hand", hand_rows)]:
        for r in rows:
            extra = set(r.keys()) - set(COLUMNS)
            missing = set(COLUMNS) - set(r.keys())
            if extra or missing:
                raise ValueError(
                    f"{label} row {r.get('question_id','?')}: "
                    f"extra={extra}, missing={missing}"
                )

    # ── order: synthetic by question_id, hand by seed order (as-is) ──
    kept_syn.sort(key=lambda r: r["question_id"])
    # hand rows keep their original order (seed order)

    merged = kept_syn + hand_rows

    # ── question_ids unique ──────────────────────────────────────────
    all_ids = [r["question_id"] for r in merged]
    if len(set(all_ids)) != len(all_ids):
        dupes = [qid for qid in all_ids if all_ids.count(qid) > 1]
        raise ValueError(f"Duplicate question_ids: {sorted(set(dupes))}")

    report["merged_rows"] = len(merged)
    report["ids_unique"] = True
    report["counts_per_stratum"] = _strata_counts(merged)
    report["counts_per_source"] = _source_counts(merged)

    return merged, report


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Merge validated synthetic and hand questions (T2c-1)")
    ap.add_argument("--config", default="config/paths.yaml",
                    help="paths.yaml config file")
    ap.add_argument("--synthetic", required=True,
                    help="Validated synthetic questions CSV")
    ap.add_argument("--hand", required=True,
                    help="Hand-written questions CSV")
    ap.add_argument("--out", required=True,
                    help="Output candidate CSV path")
    args = ap.parse_args(argv)

    paths = load_paths(args.config)
    processed_dir = Path(paths["processed"])

    syn_path = Path(args.synthetic)
    hand_path = Path(args.hand)
    out_path = Path(args.out)

    if not syn_path.exists():
        print(f"ERROR: synthetic file not found: {syn_path}", file=sys.stderr)
        return 1
    if not hand_path.exists():
        print(f"ERROR: hand file not found: {hand_path}", file=sys.stderr)
        return 1

    # ── read inputs ──────────────────────────────────────────────────
    with syn_path.open(encoding="utf-8", newline="") as f:
        synthetic_rows = list(csv.DictReader(f))
    with hand_path.open(encoding="utf-8", newline="") as f:
        hand_rows = list(csv.DictReader(f))

    # ── load corpus ids for gold validation ──────────────────────────
    corpus_ids = _load_paragraph_ids(processed_dir)

    # ── register run ─────────────────────────────────────────────────
    params = {
        "synthetic": args.synthetic,
        "hand": args.hand,
        "synthetic_sha256": _sha256(syn_path),
        "hand_sha256": _sha256(hand_path),
    }
    ctx = start_run("evalset", "merge-v1", params, paths)
    ctx.log(f"synthetic: {len(synthetic_rows)} rows from {syn_path}")
    ctx.log(f"hand: {len(hand_rows)} rows from {hand_path}")

    # ── merge ────────────────────────────────────────────────────────
    try:
        merged, report = merge(synthetic_rows, hand_rows, corpus_ids)
    except ValueError as exc:
        ctx.log(f"FAILED: {exc}")
        ctx.finish("failed", str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # ── write output atomically ──────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(merged)
    tmp.replace(out_path)

    output_sha = _sha256(out_path)

    # ── manifest ─────────────────────────────────────────────────────
    manifest = {
        "merged_rows": report["merged_rows"],
        "synthetic_Y": report["synthetic_Y"],
        "synthetic_N_per_stratum": report["synthetic_N_per_stratum"],
        "hand_appended": report["hand_appended"],
        "counts_per_stratum": report["counts_per_stratum"],
        "counts_per_source": report["counts_per_source"],
        "ids_unique": report["ids_unique"],
        "synthetic_sha256": params["synthetic_sha256"],
        "hand_sha256": params["hand_sha256"],
        "output_sha256": output_sha,
    }
    manifest_path = Path(str(out_path).replace(".csv", "_manifest.json"))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx.log(f"merged {report['merged_rows']} rows → {out_path}")
    ctx.finish("ok", f"merged {report['merged_rows']} rows",
               metrics=manifest)

    # ── stdout: counts and paths, never corpus text ──────────────────
    print(f"synthetic rows read: {report['synthetic_total']}")
    print(f"synthetic Y: {report['synthetic_Y']}")
    if report["synthetic_N_per_stratum"]:
        print("synthetic N per stratum:")
        for k, v in sorted(report["synthetic_N_per_stratum"].items()):
            print(f"  {k}: {v}")
    else:
        print("synthetic N per stratum: (none)")
    print(f"hand rows appended: {report['hand_appended']}")
    print(f"merged rows: {report['merged_rows']}")
    print("counts per route × tier × link_type:")
    for k, v in sorted(report["counts_per_stratum"].items()):
        print(f"  {k}: {v}")
    print(f"output: {out_path}")
    print(f"output sha256: {output_sha}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
