"""
Generate answers for selected configurations.

Usage::

    python -m code.eval.run_generation --config config/paths.yaml \\
        --judge config/judge.yaml --evalset <csv> --selected <json> [--limit N]
"""

import argparse
import csv
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import requests
import yaml
from transformers import AutoTokenizer

from code.common.run_registry import load_paths, start_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
TOKENIZER_NAME = "BAAI/bge-base-en-v1.5"


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _call_ollama(prompt, model, options, think):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": model, "prompt": prompt,
            "stream": False, "think": think, "options": options,
        }, timeout=180)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.warning("Ollama error: %s", e)
    return None


def _label(record, corpus):
    if corpus == "rules":
        section_title = record.get("section_title", "")
        ref = record.get("rule_ref") or record.get("paragraph_id", "")
        return f"[{section_title} | {ref}]"
    else:
        title = record.get("section_title", "")
        pid = record.get("paragraph_id", "")
        return f"[{title} | {pid}]"


def build_context(units_ret, arch, units_by_id, corpus_records, tok):
    """Build context passages from retrieved.jsonl units."""
    passages = []
    labels = []

    for u in units_ret:
        pids = u.get("paragraph_ids", [])

        if arch == "pcreturn":
            # Render paragraph_ids in seq order with heading once
            recs = [corpus_records.get(pid) for pid in pids if pid in corpus_records]
            recs.sort(key=lambda r: r.get("seq", 0))
            last_heading = None
            text_parts = []
            for rec in recs:
                hp = rec.get("heading_path", [])
                hl = " > ".join(hp) if hp else rec.get("section_title", "")
                if hl != last_heading:
                    text_parts.append(hl)
                    last_heading = hl
                text_parts.append(rec.get("text", ""))
            unit_text = "\n".join(text_parts)
            corpus = recs[0].get("corpus", "rules") if recs else "rules"
            label = _label(recs[0], corpus) if recs else ""
            passages.append(f"{label}\n{unit_text}")
            labels.append(label)

        elif arch == "linkexp":
            # Unit's own text
            uid = u.get("unit_id", "")
            unit = units_by_id.get(uid, {})
            corpus = unit.get("corpus", "rules")
            rec = corpus_records.get(uid, {})
            label = _label(rec, corpus)
            passages.append(f"{label}\n{unit.get('text', '')}")
            labels.append(label)

            # Appended targets
            for tid in u.get("appended_targets", []):
                trec = corpus_records.get(tid, {})
                tcorpus = units_by_id.get(tid, {}).get("corpus", "rules")
                tref = trec.get("rule_ref") or tid
                tlabel = f"[Cross-referenced: {tref}]"
                tunit = units_by_id.get(tid, {})
                passages.append(f"{tlabel}\n{tunit.get('text', '')}")
                labels.append(tlabel)

        else:  # flat
            uid = u.get("unit_id", "")
            unit = units_by_id.get(uid, {})
            corpus = unit.get("corpus", "rules")
            rec = corpus_records.get(uid, {})
            label = _label(rec, corpus)
            passages.append(f"{label}\n{unit.get('text', '')}")
            labels.append(label)

    context = "\n\n".join(passages)
    context_tokens = len(tok.encode(context, add_special_tokens=False))
    return context, labels, context_tokens


def main():
    ap = argparse.ArgumentParser(description="Generate answers")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--judge", default="config/judge.yaml")
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    paths = load_paths(args.config)
    with open(args.judge) as f:
        judge_cfg = yaml.safe_load(f)
    with open(args.selected) as f:
        selected = json.load(f)
    with open(args.evalset, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    if args.limit:
        questions = questions[:args.limit]

    idx_root = Path(paths["index"])
    processed = Path(paths["processed"])

    # Load units and corpus records
    units_by_id = {}
    with open(idx_root / "units_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            units_by_id[u["unit_id"]] = u

    corpus_records = {}
    for fname in ["rules-paragraphs_v1.jsonl", "guidance-paragraphs_v1.jsonl"]:
        p = processed / fname
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    corpus_records[r["paragraph_id"]] = r

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    prompt_template = Path("prompts/answer_v1.md").read_text(encoding="utf-8")
    prompt_sha = _sha256_file(Path("prompts/answer_v1.md"))

    gen_cfg = judge_cfg["generator"]
    model = gen_cfg["model"]
    options = gen_cfg["options"]
    think = gen_cfg.get("think", False)
    context_units = judge_cfg["context_units"]

    # Find run folders for selected configs
    runs_dir = Path(paths["runs"])
    config_run_dirs = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or "grid" not in d.name:
            continue
        cp = d / "config.json"
        if cp.exists():
            with open(cp) as f:
                cn = json.load(f).get("config_name", "")
            config_run_dirs[cn] = d

    for arch, info in sorted(selected.items()):
        config_name = info["config_name"]
        run_dir = config_run_dirs.get(config_name)
        if not run_dir:
            log.error("No run folder for %s", config_name)
            continue

        ret_path = run_dir / "retrieved.jsonl"
        with open(ret_path, encoding="utf-8") as f:
            retrievals = {json.loads(l)["question_id"]: json.loads(l) for l in f}

        ctx = start_run("generate", config_name, {
            "evalset_sha256": _sha256_file(Path(args.evalset)),
            "selected_sha256": _sha256_file(Path(args.selected)),
            "prompt_sha256": prompt_sha,
            "model": model, "context_units": context_units,
        }, paths)

        answers = []
        for qi, q in enumerate(questions, 1):
            qid = q["question_id"]
            ret = retrievals.get(qid, {})
            units_ret = ret.get("retrieved", [])[:context_units]

            context, labels, context_tokens = build_context(
                units_ret, arch, units_by_id, corpus_records, tok)

            prompt = prompt_template.replace("{passages}", context).replace(
                "{question}", q["question"])

            t0 = time.time()
            resp = _call_ollama(prompt, model, options, think)
            latency = time.time() - t0

            answer_text = resp.get("response", "") if resp else ""

            # Extract cited labels
            cited = re.findall(r'\[([^\]]+)\]', answer_text)

            answers.append({
                "question_id": qid,
                "labels": labels,
                "context_tokens": context_tokens,
                "answer": answer_text,
                "cited_labels": cited,
                "prompt_sha256": prompt_sha,
                "model": model,
                "digest": gen_cfg.get("digest", ""),
                "latency_s": round(latency, 2),
            })

            if qi % 5 == 0:
                log.info("  %s: %d/%d", config_name, qi, len(questions))

        out_path = ctx.run_dir / "answers.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for a in answers:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

        not_found = sum(1 for a in answers if "Not found" in a["answer"])
        mean_tokens = sum(a["context_tokens"] for a in answers) / len(answers) if answers else 0
        mean_latency = sum(a["latency_s"] for a in answers) / len(answers) if answers else 0

        ctx.log(f"answers={len(answers)} not_found={not_found}")
        ctx.finish("ok", f"{config_name}: {len(answers)} answers, {not_found} not-found")

        print(f"  {config_name}: {len(answers)} answers, {not_found} not-found, "
              f"mean_tokens={mean_tokens:.0f}, mean_latency={mean_latency:.1f}s")


if __name__ == "__main__":
    main()
