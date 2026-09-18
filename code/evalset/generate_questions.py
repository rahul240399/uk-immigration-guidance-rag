"""
Generate synthetic questions from seed groups via local LLM.

Usage::

    python -m code.evalset.generate_questions \\
        --config config/paths.yaml --evalset config/evalset.yaml \\
        --seeds data/evalset/<date>_evalset_seeds_v0.csv
"""

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path

import requests
import yaml

from code.common.run_registry import load_paths, start_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROMPT_PATH = Path("prompts/generate_question_v1.md")
OLLAMA_URL = "http://localhost:11434/api/generate"


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _call_ollama(prompt: str, model: str, options: dict, think: bool) -> dict | None:
    """Call Ollama REST API, return parsed JSON or None on failure."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": options,
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.warning("Ollama error: %s", e)
    return None


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from LLM response text."""
    # Try to find JSON block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw JSON
    m = re.search(r"\{[^{}]*\"question\"[^{}]*\"reference_answer\"[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser(description="Generate questions from seed groups")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--evalset", default="config/evalset.yaml")
    ap.add_argument("--seeds", required=True)
    args = ap.parse_args()

    paths = load_paths(args.config)
    with open(args.evalset) as f:
        evalcfg = yaml.safe_load(f)

    processed = Path(paths["processed"])
    evalset_dir = Path(paths["evalset"])
    evalset_dir.mkdir(parents=True, exist_ok=True)

    seeds_path = Path(args.seeds)
    corpus_path = processed / evalcfg["corpus"]

    # LLM config
    llm_cfg = evalcfg["llm"]
    model = llm_cfg["model"]
    options = llm_cfg["options"]
    think = llm_cfg.get("think", False)
    emb_cfg = evalcfg["embeddings"]

    # Load corpus
    records_by_id: dict[str, dict] = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            records_by_id[r["paragraph_id"]] = r

    # Load seeds
    with open(seeds_path, newline="", encoding="utf-8") as f:
        seeds = list(csv.DictReader(f))

    prompt_template = _load_prompt()
    prompt_sha = _sha256_file(PROMPT_PATH)

    from importlib.metadata import version as pkg_version
    ragas_version = "unknown"
    try:
        ragas_version = pkg_version("ragas")
    except Exception:
        pass

    # Register run
    params = {
        "evalset_yaml_sha256": _sha256_file(Path(args.evalset)),
        "seeds_sha256": _sha256_file(seeds_path),
        "prompt_path": str(PROMPT_PATH),
        "prompt_sha256": prompt_sha,
        "ragas_version": ragas_version,
        "model": model,
        "digest": llm_cfg.get("digest", ""),
    }
    ctx = start_run("evalset", "synthetic-v0", params, paths)
    ctx.log(f"model={model} seeds={len(seeds)} ragas={ragas_version}")

    today = date.today().isoformat()
    out_path = evalset_dir / f"{today}_evalset_questions-synthetic_v0.csv"

    rows_written = 0
    failures = 0
    source_counts: dict[str, int] = {}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_id", "route", "tier", "question", "reference_answer",
            "gold_paragraph_ids", "source", "generator_model", "embedding_model",
            "validated_by_student", "notes",
        ])
        writer.writeheader()

        for i, seed in enumerate(seeds, 1):
            gold_ids = seed["gold_paragraph_ids"].split(";")

            # Build input paragraphs (paragraph_id + text only, no raw corpus text in output)
            para_texts = []
            for pid in gold_ids:
                rec = records_by_id.get(pid)
                if rec:
                    ref_label = rec.get("rule_ref") or pid
                    para_texts.append(f"[{ref_label}] {rec['text']}")

            if not para_texts:
                failures += 1
                ctx.log(f"SKIP {seed['seed_id']}: no records found")
                continue

            paragraphs_block = "\n\n".join(para_texts)
            input_hash = _sha256_str(paragraphs_block)

            prompt = prompt_template.replace("{paragraphs}", paragraphs_block)
            source = "prompt-v1"

            resp = _call_ollama(prompt, model, options, think)
            if resp is None:
                failures += 1
                ctx.log(f"FAIL {seed['seed_id']}: ollama returned None, input_hash={input_hash}")
                continue

            response_text = resp.get("response", "")
            parsed = _extract_json(response_text)

            if parsed is None:
                failures += 1
                ctx.log(f"FAIL {seed['seed_id']}: could not parse JSON from response, input_hash={input_hash}")
                continue

            question = parsed.get("question", "")
            ref_answer = parsed.get("reference_answer", "")

            if not question:
                failures += 1
                ctx.log(f"FAIL {seed['seed_id']}: empty question, input_hash={input_hash}")
                continue

            source_counts[source] = source_counts.get(source, 0) + 1

            writer.writerow({
                "question_id": f"Q{i:04d}",
                "route": seed["route"],
                "tier": seed["tier"],
                "question": question,
                "reference_answer": ref_answer,
                "gold_paragraph_ids": seed["gold_paragraph_ids"],
                "source": source,
                "generator_model": model,
                "embedding_model": emb_cfg["model"],
                "validated_by_student": "N",
                "notes": "",
            })
            rows_written += 1

            if i % 10 == 0 or i == len(seeds):
                log.info("%d/%d done (written=%d failures=%d)", i, len(seeds),
                         rows_written, failures)

    ctx.log(f"written={rows_written} failures={failures}")
    ctx.finish("ok", f"synthetic-v0: {rows_written} questions, {failures} failures")

    print(f"run folder: {ctx.run_dir.name}")
    print(f"rows written: {rows_written}")
    print(f"failures: {failures}")
    print(f"rows by source: {source_counts}")
    print(f"output: {out_path}")
    print(f"sha256: {_sha256_file(out_path)}")


if __name__ == "__main__":
    main()
