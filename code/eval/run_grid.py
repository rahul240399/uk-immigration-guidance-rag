"""
Run the retrieval grid over an evaluation set.

Usage::

    python -m code.eval.run_grid --config config/paths.yaml \\
        --grid config/grid.yaml --evalset <csv> [--only <config-name>]
"""

import argparse
import csv
import hashlib
import json
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from rank_bm25 import BM25Okapi

from code.common.run_registry import load_paths, start_run
from code.index.build_index import _tokenize_bm25
from code.index.embed import encode_query, QUERY_PREFIX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ── Index loaders ────────────────────────────────────────────────────

def load_bm25(idx_dir: Path):
    with open(idx_dir / "bm25.pkl", "rb") as f:
        return pickle.load(f)


def load_embeddings(idx_dir: Path):
    model_slug = "BAAI_bge-base-en-v1.5"
    emb = np.load(idx_dir / f"emb_{model_slug}.npy")
    with open(idx_dir / f"emb_{model_slug}.id_map.json") as f:
        ids = json.load(f)
    return emb, ids


def load_chunks(idx_dir: Path, chunker: str) -> list[dict]:
    fname = "windows.jsonl" if chunker == "win64" else "chunks.jsonl"
    chunks = []
    with open(idx_dir / fname, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_parents(idx_dir: Path) -> dict:
    """Load parent_id -> parent dict."""
    parents = {}
    with open(idx_dir / "parents.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            parents[p["parent_id"]] = p
    return parents


def load_links_v2(links_path: Path) -> dict:
    """Load resolved paragraph links: from_id -> list of resolved_to ids."""
    links = defaultdict(list)
    with open(links_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["target_type"] == "paragraph" and row.get("status") == "resolved" and row.get("resolved_to"):
                links[row["from_paragraph_id"]].append(row["resolved_to"])
    return dict(links)


# ── Retrieval methods ────────────────────────────────────────────────

def retrieve_bm25(bm25, chunk_ids, query: str, depth: int) -> list[tuple[str, float]]:
    scores = bm25.get_scores(_tokenize_bm25(query))
    top = np.argsort(scores)[-depth:][::-1]
    return [(chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]


def retrieve_dense(emb, chunk_ids, query_vec, depth: int) -> list[tuple[str, float]]:
    scores = emb @ query_vec
    top = np.argsort(scores)[-depth:][::-1]
    return [(chunk_ids[i], float(scores[i])) for i in top]


def rrf(lists: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over multiple ranked lists."""
    scores: dict[str, float] = defaultdict(float)
    for ranked in lists:
        for rank, (cid, _) in enumerate(ranked, 1):
            scores[cid] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rerank(reranker, query: str, candidates: list[tuple[str, float]],
           chunk_text: dict, top_n: int, final_k: int) -> list[tuple[str, float]]:
    """Rerank top_n candidates, return final_k."""
    to_rerank = candidates[:top_n]
    pairs = [[query, chunk_text.get(cid, "")] for cid, _ in to_rerank]
    if not pairs:
        return []
    rr_scores = reranker.predict(pairs)
    combined = [(to_rerank[i][0], float(rr_scores[i])) for i in range(len(to_rerank))]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:final_k]


# ── Architecture post-processing ─────────────────────────────────────

def apply_architecture(ranked: list[tuple[str, float]], arch: str,
                       chunks_by_id: dict, parents: dict | None,
                       links: dict | None, units_by_id: dict | None,
                       top_k: int) -> list[dict]:
    results = []
    seen_appended: set[str] = set()  # For linkexp dedup across ranked list
    for cid, score in ranked[:top_k]:
        chunk = chunks_by_id.get(cid, {})
        id_key = "window_id" if "window_id" in chunk else "chunk_id"
        paragraph_ids = list(chunk.get("paragraph_ids", [cid]))
        token_count = chunk.get("token_count", 0)

        entry = {
            "unit_id": cid,
            "score": round(score, 6),
            "paragraph_ids": paragraph_ids,
            "token_count": token_count,
        }

        if arch == "linkexp" and links:
            appended = []
            appended_tokens = 0
            section_refs = 0
            absent_targets = 0
            for pid in paragraph_ids:
                for target_id in links.get(pid, []):
                    if target_id not in seen_appended and target_id not in paragraph_ids:
                        target = units_by_id.get(target_id, {})
                        if target:
                            appended.append(target_id)
                            appended_tokens += target.get("unit_token_count", target.get("token_count", 0))
                            seen_appended.add(target_id)
                        else:
                            absent_targets += 1
            # paragraph_ids = own ids + appended targets
            entry["paragraph_ids"] = paragraph_ids + appended
            entry["appended_targets"] = appended
            entry["appended_tokens"] = appended_tokens
            entry["section_refs"] = section_refs
            entry["absent_targets"] = absent_targets

        elif arch == "pcreturn" and parents:
            # D44: alternating expansion from the child outward within 512 tokens
            parent_id = chunk.get("parent_id", "")
            parent = parents.get(parent_id, {})
            all_siblings = parent.get("unit_ids", [])

            # Find child's position in siblings
            child_idx = None
            for si, sid in enumerate(all_siblings):
                if sid == cid:
                    child_idx = si
                    break

            # Heading tokens = unit_token_count - token_count for the child
            # (same for all siblings sharing the heading)
            child_unit = units_by_id.get(cid, {}) if units_by_id else {}
            heading_tokens = child_unit.get("unit_token_count", 0) - child_unit.get("token_count", 0)
            child_record_tokens = child_unit.get("token_count", 0)

            # Start with heading + child
            budget = 512
            used = heading_tokens + child_record_tokens
            included = {cid}
            included_ordered = [cid]

            if child_idx is not None:
                # Alternate: before (child_idx-1), after (child_idx+1), before-2, after+2, ...
                lo = child_idx - 1
                hi = child_idx + 1
                while lo >= 0 or hi < len(all_siblings):
                    # Try before
                    if lo >= 0:
                        sib_id = all_siblings[lo]
                        sib_unit = units_by_id.get(sib_id, {}) if units_by_id else {}
                        sib_tc = sib_unit.get("token_count", 0)
                        if used + sib_tc <= budget:
                            included.add(sib_id)
                            included_ordered.insert(0, sib_id)
                            used += sib_tc
                            lo -= 1
                        else:
                            lo = -1  # stop expanding before
                    # Try after
                    if hi < len(all_siblings):
                        sib_id = all_siblings[hi]
                        sib_unit = units_by_id.get(sib_id, {}) if units_by_id else {}
                        sib_tc = sib_unit.get("token_count", 0)
                        if used + sib_tc <= budget:
                            included.add(sib_id)
                            included_ordered.append(sib_id)
                            used += sib_tc
                            hi += 1
                        else:
                            hi = len(all_siblings)  # stop expanding after

            entry["parent_id"] = parent_id
            entry["paragraph_ids"] = included_ordered
            entry["token_count"] = used

        results.append(entry)
    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run retrieval grid")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--grid", default="config/grid.yaml")
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--only", default=None, help="Run only this config name")
    ap.add_argument("--label", default=None,
                    help="Suffix appended to config name for the run folder "
                         "(e.g. '-dryrun'); omit for no suffix")
    args = ap.parse_args()

    paths = load_paths(args.config)
    with open(args.grid) as f:
        grid = yaml.safe_load(f)

    evalset_path = Path(args.evalset)
    with open(evalset_path, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))

    qid_list = [q["question_id"] for q in questions]
    qid_sha = _sha256_str(";".join(qid_list))
    log.info("Questions: %d, qid sha256: %s", len(questions), qid_sha[:16])

    params = grid["parameters"]
    idx_root = Path(paths["index"])

    # Load sentence transformer for dense queries
    from sentence_transformers import SentenceTransformer, CrossEncoder
    st_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    reranker = None  # Lazy load

    # Load units for linkexp
    units_by_id = {}
    with open(idx_root / "units_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            units_by_id[u["unit_id"]] = u

    # Load links
    links = load_links_v2(Path(paths["processed"]) / "rules-crossrefs_v2.csv")
    # Also load guidance links
    guidance_links_path = Path(paths["processed"]) / "guidance-crossrefs_v2.csv"
    if guidance_links_path.exists():
        gl = load_links_v2(guidance_links_path)
        links.update(gl)

    configs = grid["configurations"]
    if args.only:
        configs = [c for c in configs if c["name"] == args.only]

    for cfg in configs:
        name = cfg["name"]
        chunker = cfg["chunker"]
        retrieval = cfg["retrieval"]
        arch = cfg["architecture"]

        log.info("=== %s ===", name)
        t0 = time.time()

        idx_dir = idx_root / chunker
        bm25 = load_bm25(idx_dir)
        emb, emb_ids = load_embeddings(idx_dir)
        chunks = load_chunks(idx_dir, chunker)

        id_key = "window_id" if chunker == "win64" else "chunk_id"
        chunk_ids = [c[id_key] for c in chunks]
        chunks_by_id = {c[id_key]: c for c in chunks}
        chunk_text = {c[id_key]: c["text"] for c in chunks}

        parents = load_parents(idx_dir) if chunker == "parentchild" else None

        # Register run
        run_name = name + (args.label if args.label else "")
        ctx = start_run("grid", run_name, {
            "evalset_sha256": _sha256_file(evalset_path),
            "grid_sha256": _sha256_file(Path(args.grid)),
            "qid_sha256": qid_sha,
            "chunker": chunker,
            "retrieval": retrieval,
            "architecture": arch,
            "links_version": "v2",
        }, paths)

        results = []
        for q in questions:
            query = q["question"]

            if retrieval == "bm25":
                ranked = retrieve_bm25(bm25, chunk_ids, query, params["bm25_depth"])
            elif retrieval == "dense":
                qvec = encode_query(st_model, query)
                ranked = retrieve_dense(emb, emb_ids, qvec, params["dense_depth"])
            elif retrieval == "hybrid":
                bm25_list = retrieve_bm25(bm25, chunk_ids, query, params["bm25_depth"])
                qvec = encode_query(st_model, query)
                dense_list = retrieve_dense(emb, emb_ids, qvec, params["dense_depth"])
                ranked = rrf([bm25_list, dense_list], k=params["rrf_k"])
            elif retrieval == "hybridrr":
                bm25_list = retrieve_bm25(bm25, chunk_ids, query, params["bm25_depth"])
                qvec = encode_query(st_model, query)
                dense_list = retrieve_dense(emb, emb_ids, qvec, params["dense_depth"])
                hybrid = rrf([bm25_list, dense_list], k=params["rrf_k"])
                if reranker is None:
                    reranker = CrossEncoder(params["reranker_model"], device="cpu")
                ranked = rerank(reranker, query, hybrid, chunk_text,
                                params["reranker_top_n"], params["final_top_k"])
            else:
                ranked = []

            enriched = apply_architecture(
                ranked, arch, chunks_by_id, parents,
                links if arch == "linkexp" else None,
                units_by_id if arch in ("linkexp", "pcreturn") else None,
                params["final_top_k"])

            results.append({
                "question_id": q["question_id"],
                "retrieved": enriched,
            })

        wall = time.time() - t0

        # Write retrieved.jsonl
        out_path = ctx.run_dir / "retrieved.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        ctx.log(f"questions={len(questions)} wall={wall:.1f}s")
        ctx.finish("ok", f"{name}: {len(questions)}q, {wall:.1f}s")
        log.info("  %s: %d questions, %.1fs → %s", name, len(questions), wall, ctx.run_dir.name)

    log.info("Grid complete. qid_sha256=%s", qid_sha)


if __name__ == "__main__":
    main()
