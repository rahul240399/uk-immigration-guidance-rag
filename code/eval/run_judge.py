"""
Judge generated answers for faithfulness and accuracy.

Usage::

    python -m code.eval.run_judge --judge config/judge.yaml --answers <folder> \\
        --config config/paths.yaml --evalset <csv>
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

from code.common.run_registry import load_paths, start_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"


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


def _extract_json(text):
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def judge_faithfulness(passages, answer, model, options, think):
    template = Path("prompts/judge_faithfulness_v1.md").read_text(encoding="utf-8")
    prompt = template.replace("{passages}", passages).replace("{answer}", answer)
    resp = _call_ollama(prompt, model, options, think)
    if not resp:
        return None, True
    parsed = _extract_json(resp.get("response", ""))
    if not parsed or "statements" not in parsed:
        return None, True
    stmts = parsed["statements"]
    if not stmts:
        return 1.0, False
    supported = sum(1 for s in stmts if s.get("verdict", "").lower() == "supported")
    return round(supported / len(stmts), 4), False


def judge_accuracy(question, ref_answer, answer, model, options, think):
    template = Path("prompts/judge_accuracy_v1.md").read_text(encoding="utf-8")
    prompt = (template.replace("{question}", question)
              .replace("{reference_answer}", ref_answer)
              .replace("{answer}", answer))
    resp = _call_ollama(prompt, model, options, think)
    if not resp:
        return None, "", True
    parsed = _extract_json(resp.get("response", ""))
    if not parsed or "score" not in parsed:
        return None, "", True
    return parsed["score"], parsed.get("reason", ""), False


def main():
    ap = argparse.ArgumentParser(description="Judge generated answers")
    ap.add_argument("--judge", default="config/judge.yaml")
    ap.add_argument("--answers", required=True, help="Generate run folder")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--evalset", required=True)
    args = ap.parse_args()

    paths = load_paths(args.config)
    with open(args.judge) as f:
        judge_cfg = yaml.safe_load(f)

    with open(args.evalset, newline="", encoding="utf-8") as f:
        questions = {r["question_id"]: r for r in csv.DictReader(f)}

    answers_dir = Path(args.answers)
    answers_path = answers_dir / "answers.jsonl"
    with open(answers_path, encoding="utf-8") as f:
        answers = [json.loads(l) for l in f]

    jmodel = judge_cfg["judge"]["model"]
    joptions = judge_cfg["judge"]["options"]
    jthink = judge_cfg["judge"].get("think", False)

    # Load units for passage reconstruction
    idx_root = Path(paths["index"])
    units_by_id = {}
    with open(idx_root / "units_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            units_by_id[u["unit_id"]] = u

    # Get config name from the generate run's config.json
    gen_config = json.loads((answers_dir / "config.json").read_text())
    config_name = gen_config.get("config_name", "unknown")

    # Find the grid run for this config to get retrieved.jsonl
    runs_dir = Path(paths["runs"])
    grid_run = None
    for d in sorted(runs_dir.iterdir()):
        if d.is_dir() and "grid" in d.name:
            cp = d / "config.json"
            if cp.exists():
                with open(cp) as f:
                    if json.load(f).get("config_name") == config_name:
                        grid_run = d

    retrievals = {}
    if grid_run:
        ret_path = grid_run / "retrieved.jsonl"
        if ret_path.exists():
            with open(ret_path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    retrievals[r["question_id"]] = r

    ctx = start_run("judge", config_name, {
        "judge_model": jmodel,
        "answers_dir": str(answers_dir),
    }, paths)

    judgements = []
    null_faith = 0
    null_acc = 0

    for i, ans in enumerate(answers, 1):
        qid = ans["question_id"]
        answer_text = ans["answer"]
        q = questions.get(qid, {})
        gold_ids = set(q.get("gold_paragraph_ids", "").split(";"))

        # Check "Not found" rule
        is_not_found = "Not found" in answer_text
        if is_not_found:
            # Gold absent from passages?
            ret = retrievals.get(qid, {})
            passage_pids = set()
            for u in ret.get("retrieved", [])[:5]:
                passage_pids.update(u.get("paragraph_ids", []))
            gold_absent = not gold_ids.intersection(passage_pids)
            faith_score = 1.0 if gold_absent else 0.0
            faith_null = False
        else:
            # Build passages from labels
            passages = "\n".join(ans.get("labels", []))
            faith_score, faith_null = judge_faithfulness(
                passages, answer_text, jmodel, joptions, jthink)

        if faith_null:
            null_faith += 1

        acc_score, acc_reason, acc_null = judge_accuracy(
            q.get("question", ""), q.get("reference_answer", ""),
            answer_text, jmodel, joptions, jthink)
        if acc_null:
            null_acc += 1

        judgements.append({
            "question_id": qid,
            "faithfulness": faith_score,
            "accuracy": acc_score,
            "accuracy_reason": acc_reason,
            "faithfulness_null": faith_null,
            "accuracy_null": acc_null,
            "is_not_found": is_not_found,
        })

        if i % 5 == 0:
            log.info("  %s: %d/%d judged", config_name, i, len(answers))

    out_path = ctx.run_dir / "judgements.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for j in judgements:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    # Summary
    valid_faith = [j["faithfulness"] for j in judgements if j["faithfulness"] is not None]
    valid_acc = [j["accuracy"] for j in judgements if j["accuracy"] is not None]

    summary = {
        "config_name": config_name,
        "n": len(judgements),
        "faithfulness_mean": round(sum(valid_faith) / len(valid_faith), 4) if valid_faith else None,
        "accuracy_mean": round(sum(valid_acc) / len(valid_acc), 4) if valid_acc else None,
        "null_faithfulness": null_faith,
        "null_accuracy": null_acc,
        "not_found_count": sum(1 for j in judgements if j["is_not_found"]),
    }
    (ctx.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    ctx.log(f"judged={len(judgements)} null_f={null_faith} null_a={null_acc}")
    ctx.finish("ok", f"{config_name}: {len(judgements)} judged")

    print(f"  {config_name}: faith={summary['faithfulness_mean']}, "
          f"acc={summary['accuracy_mean']}, null_f={null_faith}, null_a={null_acc}")


if __name__ == "__main__":
    main()
