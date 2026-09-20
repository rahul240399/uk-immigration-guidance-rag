"""
Retrieval diagnostics: one row per (question, gold_id, configuration).

Usage::

    python -m code.eval.diagnose --config config/paths.yaml \\
        --grid config/grid.yaml \\
        --evalset <csv> --runs <manifest.json>
"""

import argparse
import csv
import hashlib
import json
import logging
import pickle
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from code.common.run_registry import load_paths
from code.index.build_index import _tokenize_bm25
from code.index.embed import encode_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RULE_ID_RE = re.compile(
    r'\b(?:[A-Z]{1,6}\s?\d+(?:\.\d+)*|'
    r'\d{1,3}[A-Z]{1,3}\d?(?:\.\d+)*|'
    r'\d{1,3}(?:\.\d+)+)\b'
)
NUMBER_RE = re.compile(r'\b\d+\b')
SECTION_NAME_RE = re.compile(r'(?:Appendix|Part)\s+[A-Z]', re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser(description="Retrieval diagnostics")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--grid", default="config/grid.yaml")
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--runs", required=True, help="Grid manifest JSON")
    args = ap.parse_args()

    paths = load_paths(args.config)
    idx_root = Path(paths["index"])
    processed = Path(paths["processed"])

    with open(args.evalset, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    log.info("Questions: %d", len(questions))

    # Load raw corpus records for text_type and attached_to
    corpus_records = {}
    for corpus_file in ["rules-paragraphs_v1.jsonl", "guidance-paragraphs_v1.jsonl"]:
        p = processed / corpus_file
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    corpus_records[r["paragraph_id"]] = r

    # Load units
    units_by_id = {}
    with open(idx_root / "units_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            units_by_id[u["unit_id"]] = u

    # Load covariates
    cov = {}
    cov_path = processed / "rules-covariates_v1.csv"
    if cov_path.exists():
        with open(cov_path, newline="") as f:
            for row in csv.DictReader(f):
                cov[row["paragraph_id"]] = row

    # Load guidance v2 links (for competitor_cites_gold)
    guidance_links_from = defaultdict(set)
    glp = processed / "guidance-crossrefs_v2.csv"
    if glp.exists():
        with open(glp, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "resolved" and row.get("resolved_to"):
                    guidance_links_from[row["from_paragraph_id"]].add(row["resolved_to"])

    # Load grid manifest and retrieved.jsonl per config
    with open(args.runs) as f:
        run_folders = json.load(f)["run_folders"]

    config_retrieved: dict[str, dict[str, dict]] = {}
    for config_name, run_dir_str in run_folders.items():
        run_dir = Path(run_dir_str)
        ret_path = run_dir / "retrieved.jsonl"
        if not ret_path.exists():
            continue
        qid_map = {}
        with open(ret_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                qid_map[r["question_id"]] = r
        config_retrieved[config_name] = qid_map

    # Load models and indexes per chunker
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")

    with open(args.grid) as f:
        grid_cfg = yaml.safe_load(f)

    # Map config_name -> chunker
    config_chunker = {c["name"]: c["chunker"] for c in grid_cfg["configurations"]}

    chunker_data = {}
    for cfg in grid_cfg["configurations"]:
        chunker = cfg["chunker"]
        if chunker in chunker_data:
            continue
        idx_dir = idx_root / chunker
        chunk_file = "windows.jsonl" if chunker == "win64" else "chunks.jsonl"
        id_key = "window_id" if chunker == "win64" else "chunk_id"

        chunks = []
        with open(idx_dir / chunk_file, encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        chunk_ids = [c[id_key] for c in chunks]
        unit_to_chunks = defaultdict(list)
        for i, c in enumerate(chunks):
            for pid in c.get("paragraph_ids", []):
                unit_to_chunks[pid].append(i)

        with open(idx_dir / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        emb = np.load(idx_dir / "emb_BAAI_bge-base-en-v1.5.npy")

        chunker_data[chunker] = {
            "bm25": bm25, "emb": emb, "chunks": chunks,
            "chunk_ids": chunk_ids,
            "unit_to_chunks": dict(unit_to_chunks),
        }

    # Identify flat configs per chunker (for parent_in_top10)
    flat_configs = {}
    for cfg in grid_cfg["configurations"]:
        if cfg["architecture"] == "flat":
            key = f"{cfg['chunker']}-{cfg['retrieval']}"
            flat_configs[key] = cfg["name"] + "-dryrun"

    config_names = sorted(config_retrieved.keys())

    rows = []
    for qi, q in enumerate(questions):
        qid = q["question_id"]
        tier = q["tier"]
        gold_ids = q["gold_paragraph_ids"].split(";")
        question_text = q["question"]
        gold_count = len(gold_ids)

        q_tokens = _tokenize_bm25(question_text)
        q_has_id = bool(RULE_ID_RE.search(question_text))
        q_has_number = bool(NUMBER_RE.search(question_text))
        q_names_section = bool(SECTION_NAME_RE.search(question_text))
        q_vec = encode_query(st_model, question_text)

        for gid in gold_ids:
            gold_unit = units_by_id.get(gid)
            if not gold_unit:
                continue

            # Get text_type and attached_to from corpus record
            corpus_rec = corpus_records.get(gid, {})
            text_type = corpus_rec.get("text_type", "")
            attached_to = corpus_rec.get("attached_to") or ""
            is_child = bool(attached_to)

            gold_cov = cov.get(gid, {})
            heading_path = gold_unit.get("heading_path", [])
            gold_tc = gold_unit["token_count"]
            gold_utc = gold_unit["unit_token_count"]
            heading_share = round((gold_utc - gold_tc) / gold_utc, 3) if gold_utc > 0 else 0

            # BM25 overlap
            gold_bm25_tokens = set(_tokenize_bm25(gold_unit["text"]))
            q_set = set(q_tokens)
            bm25_overlap = len(q_set & gold_bm25_tokens) / len(q_set) if q_set else 0

            # Cosine q-gold (use para chunker as reference)
            cosine_q_gold = 0.0
            for cn, cd in chunker_data.items():
                if gid in cd["unit_to_chunks"]:
                    ci = cd["unit_to_chunks"][gid][0]
                    cosine_q_gold = float(np.dot(q_vec, cd["emb"][ci]))
                    break

            # Twin cosine: max cosine to any other unit
            twin_cosine = 0.0
            twin_section = ""
            if "para" in chunker_data:
                cd = chunker_data["para"]
                if gid in cd["unit_to_chunks"]:
                    gi = cd["unit_to_chunks"][gid][0]
                    gold_vec = cd["emb"][gi]
                    sims = cd["emb"] @ gold_vec
                    sims[gi] = -1  # exclude self
                    best_i = int(np.argmax(sims))
                    twin_cosine = round(float(sims[best_i]), 4)
                    best_uid = cd["chunk_ids"][best_i]
                    twin_unit = units_by_id.get(best_uid, {})
                    twin_section = twin_unit.get("section_base_path", "")

            # Full-list ranks are computed per-config below
            # (they depend on the config's chunker)

            row = {
                "question_id": qid, "gold_id": gid, "tier": tier,
                "gold_count": gold_count, "q_tokens": len(q_tokens),
                "q_has_identifier": q_has_id, "q_has_number": q_has_number,
                "q_names_section": q_names_section,
                "bm25_overlap": round(bm25_overlap, 3),
                "cosine_q_gold": round(cosine_q_gold, 4),
                "text_type": text_type,
                "token_count": gold_tc, "unit_token_count": gold_utc,
                "heading_share": heading_share,
                "is_child": is_child, "parent_id": attached_to,
                "heading_empty": len(heading_path) == 0,
                "drafting_style": gold_cov.get("drafting_style", ""),
                "section": gold_unit["section_base_path"],
                "corpus": gold_unit["corpus"],
                "twin_exact": sum(1 for ids in [
                    [u for u in units_by_id.values() if u["text"] == gold_unit["text"] and u["unit_id"] != gid]
                ] for _ in ids),
                "twin_cosine": twin_cosine, "twin_section": twin_section,
                "truncated": gold_utc > 512,
            }

            # Per-config: rank, found_at, competitor, parent_in_top10
            for config_name in config_names:
                ret = config_retrieved.get(config_name, {}).get(qid, {})
                units_ret = ret.get("retrieved", [])

                # Rank of first unit carrying gold in stored top 20
                found_rank = None
                for rank, u in enumerate(units_ret, 1):
                    if gid in u.get("paragraph_ids", []):
                        found_rank = rank
                        break

                row[f"{config_name}__rank"] = found_rank or "absent"
                row[f"{config_name}__found_at_5"] = found_rank is not None and found_rank <= 5
                row[f"{config_name}__found_at_10"] = found_rank is not None and found_rank <= 10
                row[f"{config_name}__found_at_20"] = found_rank is not None and found_rank <= 20

                # Full-list BM25 and dense ranks from this config's chunker
                base_name = config_name.replace("-dryrun", "")
                chunker = config_chunker.get(base_name, "para")
                cd = chunker_data.get(chunker, {})

                bm25_rank_full = "absent"
                dense_rank_full = "absent"

                if cd and gid in cd.get("unit_to_chunks", {}):
                    gold_ci = cd["unit_to_chunks"][gid]

                    # BM25 rank: count units scoring above gold in full index
                    bm25_scores = cd["bm25"].get_scores(q_tokens)
                    gold_bm25 = max(bm25_scores[i] for i in gold_ci)
                    bm25_rank_full = int(np.sum(bm25_scores > gold_bm25)) + 1

                    # Dense rank
                    dense_scores = cd["emb"] @ q_vec
                    gold_dense = max(dense_scores[i] for i in gold_ci)
                    dense_rank_full = int(np.sum(dense_scores > gold_dense)) + 1

                row[f"{config_name}__bm25_rank_full"] = bm25_rank_full
                row[f"{config_name}__dense_rank_full"] = dense_rank_full

                # Competitor: top unit not carrying any gold
                gold_set = set(gold_ids)
                comp_id = comp_section = comp_corpus = ""
                comp_cosine = 0.0
                comp_cites_gold = False
                for u in units_ret:
                    if not any(pid in gold_set for pid in u.get("paragraph_ids", [])):
                        comp_id = u.get("unit_id", "")
                        comp_unit = units_by_id.get(comp_id, {})
                        comp_section = comp_unit.get("section_base_path", "")
                        comp_corpus = comp_unit.get("corpus", "")
                        # Cosine to query
                        for cn, cd in chunker_data.items():
                            if comp_id in cd["unit_to_chunks"]:
                                ci = cd["unit_to_chunks"][comp_id][0]
                                comp_cosine = round(float(np.dot(q_vec, cd["emb"][ci])), 4)
                                break
                        # Does competitor cite gold?
                        if comp_corpus == "guidance":
                            comp_cites_gold = any(
                                gid in guidance_links_from.get(comp_id, set())
                                for gid in gold_ids)
                        break

                row[f"{config_name}__competitor_id"] = comp_id
                row[f"{config_name}__competitor_section"] = comp_section
                row[f"{config_name}__competitor_corpus"] = comp_corpus
                row[f"{config_name}__competitor_cosine"] = comp_cosine
                row[f"{config_name}__competitor_cites_gold"] = comp_cites_gold

                # parent_in_top10: for a child gold, whether parent_id is in top 10
                # of the corresponding flat run
                parent_in_top10 = False
                if is_child and attached_to:
                    # Find the flat config for this chunker+retrieval
                    base_name = config_name.replace("-dryrun", "")
                    parts = base_name.split("-")
                    # chunker-bge-retrieval-arch -> chunker-retrieval
                    if len(parts) >= 4:
                        flat_key = f"{parts[0]}-{parts[2]}"
                        flat_name = flat_configs.get(flat_key, "")
                        flat_ret = config_retrieved.get(flat_name, {}).get(qid, {})
                        flat_units = flat_ret.get("retrieved", [])
                        for rank, u in enumerate(flat_units[:10], 1):
                            if attached_to in u.get("paragraph_ids", []):
                                parent_in_top10 = True
                                break

                row[f"{config_name}__parent_in_top10"] = parent_in_top10

            rows.append(row)

        if (qi + 1) % 20 == 0:
            log.info("  %d/%d questions", qi + 1, len(questions))

    # Write CSV
    today = date.today().isoformat()
    results_dir = Path(paths["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{today}_diagnostics.csv"

    fieldnames = [
        "question_id", "gold_id", "tier", "gold_count", "q_tokens",
        "q_has_identifier", "q_has_number", "q_names_section",
        "bm25_overlap", "cosine_q_gold",
        "text_type", "token_count", "unit_token_count", "heading_share",
        "is_child", "parent_id", "heading_empty", "drafting_style",
        "section", "corpus", "twin_exact", "twin_cosine", "twin_section",
        "truncated",
    ]
    for cn in config_names:
        fieldnames.extend([
            f"{cn}__rank", f"{cn}__found_at_5", f"{cn}__found_at_10",
            f"{cn}__found_at_20",
            f"{cn}__bm25_rank_full", f"{cn}__dense_rank_full",
            f"{cn}__competitor_id",
            f"{cn}__competitor_section", f"{cn}__competitor_corpus",
            f"{cn}__competitor_cosine", f"{cn}__competitor_cites_gold",
            f"{cn}__parent_in_top10",
        ])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = results_dir / f"{today}_diagnostics_manifest.json"
    manifest_path.write_text(json.dumps({
        "rows": len(rows), "questions": len(questions),
        "configs": config_names, "runs_manifest": str(args.runs),
    }, indent=2))

    log.info("Wrote %d rows to %s", len(rows), out_path)

    # Assertion: no row has found_at_10=False with rank_full <= 10,
    # and no row has found_at_10=True with rank_full > 10
    # (checked on flat BM25 configs where __rank maps to bm25_rank_full)
    assertion_violations = 0
    for r in rows:
        for cn in config_names:
            base = cn.replace("-dryrun", "")
            retrieval = ""
            arch = ""
            for c in grid_cfg["configurations"]:
                if c["name"] == base:
                    retrieval = c["retrieval"]
                    arch = c["architecture"]
                    break
            # Only assert on flat configs with a single retriever
            if arch != "flat" or retrieval not in ("bm25", "dense"):
                continue

            found = r.get(f"{cn}__found_at_10", False)
            if retrieval == "bm25":
                rf = r.get(f"{cn}__bm25_rank_full", "absent")
            else:
                rf = r.get(f"{cn}__dense_rank_full", "absent")
            if rf == "absent":
                continue
            if found and rf > 10:
                assertion_violations += 1
            if not found and rf <= 10:
                assertion_violations += 1

    print(f"\nAssertion check: {assertion_violations} violations "
          f"({'PASS' if assertion_violations == 0 else 'FAIL'})")

    # Report
    first_cfg = config_names[0]
    print(f"\nRows: {len(rows)}")
    print(f"\nfound_at_10 by text_type ({first_cfg}):")
    tt_counts = Counter()
    tt_found = Counter()
    for r in rows:
        tt_counts[r["text_type"]] += 1
        if r.get(f"{first_cfg}__found_at_10"):
            tt_found[r["text_type"]] += 1
    for t in sorted(tt_counts):
        print(f"  {t:15s} {tt_found[t]:3d}/{tt_counts[t]:3d}")


if __name__ == "__main__":
    main()
