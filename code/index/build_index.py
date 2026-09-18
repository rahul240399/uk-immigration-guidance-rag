"""
Build retrieval indexes: para, parentchild, win64.

Usage::

    python -m code.index.build_index --config config/paths.yaml --chunker para
    python -m code.index.build_index --config config/paths.yaml --chunker parentchild
    python -m code.index.build_index --config config/paths.yaml --chunker win64 [--rebuild]
"""

import argparse
import hashlib
import json
import logging
import pickle
import re
import statistics
from collections import defaultdict
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

from code.common.run_registry import load_paths, start_run, index_dir

TOKENIZER_NAME = "BAAI/bge-base-en-v1.5"
BM25_K1 = 1.5
BM25_B = 0.75

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tokenize_bm25(text: str) -> list[str]:
    """BM25 tokenization: lowercase, split on non-alphanumerics, drop length-1."""
    return [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(t) > 1]


def _load_units(index_dir: Path) -> list[dict]:
    units = []
    with open(index_dir / "units_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            units.append(json.loads(line.strip()))
    return units


def _build_bm25(chunks: list[dict]) -> BM25Okapi:
    corpus = [_tokenize_bm25(c["text"]) for c in chunks]
    return BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)


def build_para(units: list[dict]) -> list[dict]:
    """One chunk per unit."""
    return [{
        "chunk_id": u["unit_id"],
        "text": u["text"],
        "paragraph_ids": [u["unit_id"]],
        "section_base_path": u["section_base_path"],
        "corpus": u["corpus"],
        "route": u["route"],
        "token_count": u["unit_token_count"],
    } for u in units]


def build_parentchild(units: list[dict]) -> tuple[list[dict], list[dict]]:
    """Children = para chunks. Parents = units sharing heading_path within section."""
    children = build_para(units)

    # Group by (section, heading_line)
    groups: dict[str, list[dict]] = defaultdict(list)
    for u in units:
        hp = u.get("heading_path", [])
        hl = " > ".join(hp) if hp else u.get("section_base_path", "")
        key = u["section_base_path"] + "|" + hl
        groups[key].append(u)

    parents = []
    for key, group_units in groups.items():
        section, heading = key.split("|", 1)
        group_units.sort(key=lambda u: u["seq"])
        unit_ids = [u["unit_id"] for u in group_units]
        parent_text = heading + "\n" + "\n".join(u["text"].split("\n", 1)[-1] for u in group_units)
        pid = hashlib.sha1(key.encode()).hexdigest()[:12]
        parents.append({
            "parent_id": pid,
            "section_base_path": section,
            "heading": heading,
            "unit_ids": unit_ids,
            "text": parent_text,
            "token_count": sum(u["unit_token_count"] for u in group_units),
        })

    # Add parent_id to children
    unit_to_parent = {}
    for p in parents:
        for uid in p["unit_ids"]:
            unit_to_parent[uid] = p["parent_id"]
    for c in children:
        c["parent_id"] = unit_to_parent.get(c["chunk_id"], "")

    return children, parents


def build_win64(units: list[dict], tok) -> tuple[list[dict], dict]:
    """64-token windows with offset-mapped text, strict overlap accounting.
    
    Returns (windows, stats) where stats has invariant counts.
    """
    WINDOW_SIZE = 64

    # Build unit token count lookup
    unit_tc: dict[str, int] = {u["unit_id"]: u["token_count"] for u in units}

    # Group by section
    groups: dict[str, list[dict]] = defaultdict(list)
    for u in units:
        groups[u["section_base_path"]].append(u)

    windows: list[dict] = []
    dropped_heading_only = 0

    # Per-record overlap accumulator (for invariant 3a)
    record_overlap_total: dict[str, int] = defaultdict(int)

    for section, section_units in groups.items():
        section_units.sort(key=lambda u: u["seq"])
        slug = section.rstrip("/").split("/")[-1] or "index"

        # Build the original text stream and token attribution
        stream_parts: list[str] = []  # original text pieces
        attribution: list[str | None] = []  # one entry per character: paragraph_id or None
        last_heading = None

        for u in section_units:
            hp = u.get("heading_path", [])
            hl = " > ".join(hp) if hp else u.get("section_base_path", "")

            if hl != last_heading:
                for ch in hl:
                    stream_parts.append(ch)
                    attribution.append(None)  # heading chars
                last_heading = hl

            record_text = u["text"].split("\n", 1)[-1] if "\n" in u["text"] else u["text"]
            for ch in record_text:
                stream_parts.append(ch)
                attribution.append(u["unit_id"])

        full_stream = "".join(stream_parts)

        # Tokenize with offset mapping
        enc = tok(full_stream, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]  # list of (start, end) char positions

        # Attribute each token to a paragraph via its character span
        token_attrib: list[str | None] = []
        for start, end in offsets:
            # Find the paragraph_id that owns the majority of this token's chars
            if start >= end:
                token_attrib.append(None)
                continue
            char_owners: dict[str | None, int] = defaultdict(int)
            for ci in range(start, min(end, len(attribution))):
                char_owners[attribution[ci]] += 1
            # Pick the owner with most chars; None = heading
            best = max(char_owners, key=char_owners.get)
            token_attrib.append(best)

        # Cut into windows
        win_num = 0
        for w_start in range(0, len(token_ids), WINDOW_SIZE):
            w_end = min(w_start + WINDOW_SIZE, len(token_ids))
            if w_start >= w_end:
                continue

            # Window text = exact substring via offset mapping
            char_start = offsets[w_start][0]
            char_end = offsets[w_end - 1][1]
            window_text = full_stream[char_start:char_end]

            # Count tokens per paragraph in this window (ALL records)
            para_tokens: dict[str, int] = defaultdict(int)
            heading_token_count = 0
            for ti in range(w_start, w_end):
                pid = token_attrib[ti]
                if pid is None:
                    heading_token_count += 1
                else:
                    para_tokens[pid] += 1

            # Drop heading-only windows
            if not para_tokens:
                dropped_heading_only += 1
                continue

            win_num += 1
            window_id = f"{slug}:w{win_num}"

            # overlap_tokens: ALL records, not just carried
            overlap_per_id = dict(para_tokens)

            # Update global overlap accumulator
            for pid, cnt in para_tokens.items():
                record_overlap_total[pid] += cnt

            # 50% rule: carried = records with overlap >= 50% of their own total tokens
            carried = []
            for pid, cnt in para_tokens.items():
                total = unit_tc.get(pid, cnt)
                if total > 0 and cnt >= total * 0.5:
                    carried.append(pid)

            # Fallback: largest overlap
            if not carried:
                best_pid = max(para_tokens, key=para_tokens.get)
                carried = [best_pid]

            windows.append({
                "window_id": window_id,
                "section_base_path": section,
                "text": window_text,
                "token_count": w_end - w_start,
                "paragraph_ids": carried,
                "overlap_tokens": overlap_per_id,
                "heading_tokens": heading_token_count,
            })

    # Invariant stats
    # 3a: records where sum of overlap != token_count
    overlap_mismatch = 0
    for u in units:
        uid = u["unit_id"]
        expected = u["token_count"]
        actual = record_overlap_total.get(uid, 0)
        if actual != expected:
            overlap_mismatch += 1

    # 3b: records not carried by any window
    carried_set: set[str] = set()
    for w in windows:
        carried_set.update(w["paragraph_ids"])
    not_carried = sum(1 for u in units if u["unit_id"] not in carried_set)

    # Fallback count
    fallback_count = 0
    for w in windows:
        # A window used fallback if none of its carried records met 50%
        for pid in w["paragraph_ids"]:
            total = unit_tc.get(pid, 1)
            overlap = w["overlap_tokens"].get(pid, 0)
            if total > 0 and overlap >= total * 0.5:
                break
        else:
            fallback_count += 1

    ids_per_win = [len(w["paragraph_ids"]) for w in windows]
    import statistics as _stats
    stats = {
        "window_count": len(windows),
        "dropped_heading_only": dropped_heading_only,
        "ids_per_window_min": min(ids_per_win) if ids_per_win else 0,
        "ids_per_window_median": _stats.median(ids_per_win) if ids_per_win else 0,
        "ids_per_window_mean": round(_stats.mean(ids_per_win), 2) if ids_per_win else 0,
        "ids_per_window_max": max(ids_per_win) if ids_per_win else 0,
        "fallback_windows": fallback_count,
        "fallback_share": round(fallback_count / len(windows), 4) if windows else 0,
        "records_carried": len(carried_set),
        "records_total": len(units),
        "invariant_overlap_mismatch": overlap_mismatch,
        "invariant_not_carried": not_carried,
    }

    return windows, stats


def main():
    ap = argparse.ArgumentParser(description="Build retrieval index")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--chunker", required=True, choices=["para", "parentchild", "win64"])
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    paths = load_paths(args.config)
    idx_path = index_dir(args.chunker, paths, rebuild=args.rebuild)

    units = _load_units(Path(paths["index"]))
    log.info("Loaded %d units", len(units))

    manifest_path = Path(paths["index"]) / "units_v1.manifest.json"
    units_sha = _sha256_file(manifest_path) if manifest_path.exists() else ""

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    ctx = start_run("index", args.chunker, {
        "chunker": args.chunker,
        "window_size": 64,
        "tokenizer": TOKENIZER_NAME,
        "bm25_k1": BM25_K1,
        "bm25_b": BM25_B,
        "units_manifest_sha256": units_sha,
    }, paths)

    if args.chunker == "para":
        chunks = build_para(units)
        _write_chunks(idx_path, chunks, "chunks.jsonl")
        bm25 = _build_bm25(chunks)
        _write_bm25(idx_path, bm25)
        ctx.log(f"para: {len(chunks)} chunks")
        headline = f"para: {len(chunks)} chunks"

    elif args.chunker == "parentchild":
        children, parents = build_parentchild(units)
        _write_chunks(idx_path, children, "chunks.jsonl")
        _write_chunks(idx_path, parents, "parents.jsonl")
        bm25 = _build_bm25(children)
        _write_bm25(idx_path, bm25)
        ctx.log(f"parentchild: {len(children)} children, {len(parents)} parents")
        headline = f"parentchild: {len(children)} children, {len(parents)} parents"

    elif args.chunker == "win64":
        windows, win_stats = build_win64(units, tok)
        _write_chunks(idx_path, windows, "windows.jsonl")
        bm25 = _build_bm25(windows)
        _write_bm25(idx_path, bm25)

        # Extended build.log
        log_lines = [
            f"median_rules_token_count: {win_stats.get('median_rules_tc', 'N/A')}",
            f"window_size: 64",
            f"window_count: {win_stats['window_count']}",
            f"dropped_heading_only: {win_stats['dropped_heading_only']}",
            f"ids_per_window: min={win_stats['ids_per_window_min']} median={win_stats['ids_per_window_median']} mean={win_stats['ids_per_window_mean']} max={win_stats['ids_per_window_max']}",
            f"fallback_windows: {win_stats['fallback_windows']} ({win_stats['fallback_share']:.2%})",
            f"records_carried: {win_stats['records_carried']}/{win_stats['records_total']}",
            f"invariant_overlap_mismatch: {win_stats['invariant_overlap_mismatch']}",
            f"invariant_not_carried: {win_stats['invariant_not_carried']}",
        ]
        (idx_path / "build.log").write_text("\n".join(log_lines) + "\n")

        # Tokenizer revision
        tok_revision = getattr(tok, "name_or_path", TOKENIZER_NAME)

        build_config = {
            "chunker": "win64",
            "window_size": 64,
            "tokenizer": TOKENIZER_NAME,
            "tokenizer_revision": tok_revision,
            "bm25": {"k1": BM25_K1, "b": BM25_B,
                     "tokenization": "lowercase, split on non-alphanumerics, drop length-1"},
            "units_manifest_sha256": units_sha,
            "chunk_count": len(windows),
            "invariants": win_stats,
        }
        (idx_path / "config.json").write_text(json.dumps(build_config, indent=2))

        ctx.log(f"win64: {len(windows)} windows, {win_stats['dropped_heading_only']} dropped, "
                f"fallback={win_stats['fallback_windows']}, "
                f"overlap_mismatch={win_stats['invariant_overlap_mismatch']}, "
                f"not_carried={win_stats['invariant_not_carried']}")
        headline = (f"win64: {len(windows)} windows, "
                    f"carried={win_stats['records_carried']}/{win_stats['records_total']}")
        ctx.finish("ok", headline)
        log.info("Done: %s → %s", args.chunker, idx_path)
        log.info("Stats: %s", json.dumps(win_stats))
        return

    # Write build config
    build_config = {
        "chunker": args.chunker,
        "window_size": 64,
        "tokenizer": TOKENIZER_NAME,
        "bm25": {"k1": BM25_K1, "b": BM25_B,
                 "tokenization": "lowercase, split on non-alphanumerics, drop length-1"},
        "units_manifest_sha256": units_sha,
        "chunk_count": len(windows) if args.chunker == "win64" else len(chunks if args.chunker == "para" else children),
    }
    (idx_path / "config.json").write_text(json.dumps(build_config, indent=2))

    # Write build.log marker
    (idx_path / "build.log").write_text(f"built {args.chunker}\n")

    ctx.finish("ok", headline)
    log.info("Done: %s → %s", args.chunker, idx_path)


def _write_chunks(idx_path: Path, chunks: list[dict], filename: str):
    with open(idx_path / filename, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def _write_bm25(idx_path: Path, bm25: BM25Okapi):
    with open(idx_path / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)


if __name__ == "__main__":
    main()
