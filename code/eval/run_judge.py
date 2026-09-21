"""
Judge generated answers for faithfulness and accuracy (T6b-2).

Call A: extract atomic statements from the answer, verdict each as
supported / unsupported.  faithfulness = supported / total.
"Not found" answer: statements=[], faithfulness=1.0 when no gold id among
passage ids, else 0.0; recorded as ``not_found_case``.

Call B: accuracy (1 / 0.5 / 0), reason (one sentence).

Malformed output → score is null and ``flag`` names the problem; never a
guessed value.

``--verify <folder>``: recompute faithfulness from stored statements/verdicts
and compare; exit 1 on any difference.

Usage::

    python -m code.eval.run_judge --judge config/judge.yaml \\
        --answers <folder> --config config/paths.yaml --evalset <csv>
    python -m code.eval.run_judge --verify <folder> --evalset <csv>
"""

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests
import yaml

from code.common.run_registry import load_paths, start_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
VALID_VERDICTS = {"supported", "unsupported"}
VALID_ACCURACY = {0, 0.5, 1}


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ── Ollama ───────────────────────────────────────────────────────────

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


# ── JSON extraction ──────────────────────────────────────────────────

def _extract_json(text):
    """Extract a JSON object from text (fenced or bare). Returns dict or None."""
    if text is None:
        return None
    # Try fenced markdown first
    m = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare JSON — find the outermost { ... } by brace counting
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# ── Faithfulness (Call A) ────────────────────────────────────────────

def compute_faithfulness(statements: list[str],
                         verdicts: list[str]) -> float:
    """faithfulness = supported / total. Pure, no I/O."""
    if not statements:
        return 1.0  # vacuously faithful (no claims)
    supported = sum(1 for v in verdicts if v == "supported")
    return round(supported / len(statements), 6)


def parse_faithfulness(raw_text: str | None) -> dict:
    """Parse Call A output.

    Returns dict with keys: statements, verdicts, faithfulness, flag.
    On malformed output, faithfulness is None and flag names the problem.
    """
    if raw_text is None:
        return {"statements": None, "verdicts": None,
                "faithfulness": None, "flag": "no_response"}

    parsed = _extract_json(raw_text)
    if parsed is None:
        return {"statements": None, "verdicts": None,
                "faithfulness": None, "flag": "unparseable_json"}

    stmts_raw = parsed.get("statements")
    if not isinstance(stmts_raw, list):
        return {"statements": None, "verdicts": None,
                "faithfulness": None, "flag": "missing_statements_list"}

    # Extract statements and verdicts
    statements = []
    verdicts = []
    for item in stmts_raw:
        if not isinstance(item, dict):
            return {"statements": None, "verdicts": None,
                    "faithfulness": None, "flag": "statement_not_dict"}
        s = item.get("statement", "")
        v = item.get("verdict", "").lower().strip()
        if v not in VALID_VERDICTS:
            return {"statements": None, "verdicts": None,
                    "faithfulness": None,
                    "flag": f"invalid_verdict:{v!r}"}
        statements.append(s)
        verdicts.append(v)

    faith = compute_faithfulness(statements, verdicts)
    return {"statements": statements, "verdicts": verdicts,
            "faithfulness": faith, "flag": None}


def judge_faithfulness_call(passages: str, answer: str,
                            model: str, options: dict, think: bool) -> dict:
    """Run Call A via Ollama and parse the result."""
    template = Path("prompts/judge_faithfulness_v1.md").read_text(encoding="utf-8")
    prompt = template.replace("{passages}", passages).replace("{answer}", answer)
    resp = _call_ollama(prompt, model, options, think)
    raw = resp.get("response", "") if resp else None
    return parse_faithfulness(raw)


# ── Accuracy (Call B) ────────────────────────────────────────────────

def parse_accuracy(raw_text: str | None) -> dict:
    """Parse Call B output.

    Returns dict with keys: accuracy, reason, flag.
    On malformed output, accuracy is None and flag names the problem.
    """
    if raw_text is None:
        return {"accuracy": None, "reason": None, "flag": "no_response"}

    parsed = _extract_json(raw_text)
    if parsed is None:
        return {"accuracy": None, "reason": None, "flag": "unparseable_json"}

    score = parsed.get("score")
    if score is None:
        return {"accuracy": None, "reason": None, "flag": "missing_score"}

    # Normalise to float
    try:
        score = float(score)
    except (ValueError, TypeError):
        return {"accuracy": None, "reason": None,
                "flag": f"score_not_numeric:{score!r}"}

    if score not in VALID_ACCURACY:
        return {"accuracy": None, "reason": None,
                "flag": f"score_outside_set:{score}"}

    reason = parsed.get("reason", "")
    return {"accuracy": score, "reason": reason, "flag": None}


def judge_accuracy_call(question: str, ref_answer: str, answer: str,
                        model: str, options: dict, think: bool) -> dict:
    """Run Call B via Ollama and parse the result."""
    template = Path("prompts/judge_accuracy_v1.md").read_text(encoding="utf-8")
    prompt = (template.replace("{question}", question)
              .replace("{reference_answer}", ref_answer)
              .replace("{answer}", answer))
    resp = _call_ollama(prompt, model, options, think)
    raw = resp.get("response", "") if resp else None
    return parse_accuracy(raw)


# ── Not-found handling ───────────────────────────────────────────────

def handle_not_found(gold_ids: set[str],
                     passage_pids: set[str]) -> dict:
    """Handle 'Not found in the provided rules.' answers.

    statements=[], faithfulness=1.0 when gold absent, else 0.0.
    """
    gold_absent = not gold_ids.intersection(passage_pids)
    return {
        "statements": [],
        "verdicts": [],
        "faithfulness": 1.0 if gold_absent else 0.0,
        "flag": None,
        "not_found_case": True,
    }


# ── Verify mode ──────────────────────────────────────────────────────

def verify(judge_folder: Path, evalset_path: Path) -> bool:
    """Recompute faithfulness from stored statements/verdicts and compare.

    Returns True if all match; logs and returns False on any difference.
    """
    jpath = judge_folder / "judgements.jsonl"
    if not jpath.exists():
        log.error("judgements.jsonl not found in %s", judge_folder)
        return False

    with jpath.open(encoding="utf-8") as f:
        judgements = [json.loads(line) for line in f if line.strip()]

    ok = True
    for j in judgements:
        stored = j.get("faithfulness")
        stmts = j.get("statements")
        vdcts = j.get("verdicts")

        # Skip null rows (malformed — can't recompute)
        if stmts is None or vdcts is None:
            continue

        recomputed = compute_faithfulness(stmts, vdcts)
        if stored is not None and abs(recomputed - stored) > 1e-6:
            log.error("  %s: stored=%.6f recomputed=%.6f",
                      j.get("question_id", "?"), stored, recomputed)
            ok = False

    return ok


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Judge answers (T6b-2)")
    ap.add_argument("--judge", default="config/judge.yaml")
    ap.add_argument("--answers", default=None, help="Generate run folder")
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--evalset", required=True)
    ap.add_argument("--verify", default=None, metavar="JUDGE_FOLDER",
                    help="Verify faithfulness in a judge folder")
    args = ap.parse_args(argv)

    # ── verify mode ──────────────────────────────────────────────────
    if args.verify:
        ok = verify(Path(args.verify), Path(args.evalset))
        status = "OK" if ok else "FAIL"
        print(f"VERIFY {status}: {Path(args.verify).name}")
        return 0 if ok else 1

    # ── judge mode ───────────────────────────────────────────────────
    if not args.answers:
        print("ERROR: --answers required when not using --verify",
              file=sys.stderr)
        return 1

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

    # Load units for passage-pid check (not-found rule)
    idx_root = Path(paths["index"])
    units_by_id = {}
    with open(idx_root / "units_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            units_by_id[u["unit_id"]] = u

    # Get config name from the generate run's config.json
    gen_config = json.loads((answers_dir / "config.json").read_text())
    config_name = gen_config.get("config_name", "unknown")

    # Find the grid run for passage pids
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

    context_units = judge_cfg.get("generation", {}).get("context_units", 5)

    ctx = start_run("judge", config_name, {
        "judge_model": jmodel,
        "answers_dir": str(answers_dir),
    }, paths)

    judgements = []
    null_count = 0

    for i, ans in enumerate(answers, 1):
        qid = ans["question_id"]
        answer_text = ans["answer"]
        q = questions.get(qid, {})
        gold_ids = set(g.strip() for g in
                       q.get("gold_paragraph_ids", "").split(";") if g.strip())

        # Passage pids for not-found check
        ret = retrievals.get(qid, {})
        passage_pids = set()
        for u in ret.get("retrieved", [])[:context_units]:
            passage_pids.update(u.get("paragraph_ids", []))

        # Check "Not found" rule
        is_not_found = "Not found" in answer_text

        # ── Call A: faithfulness ─────────────────────────────────────
        if is_not_found:
            faith_result = handle_not_found(gold_ids, passage_pids)
        else:
            passages = "\n\n".join(
                f"{lb}\n{units_by_id.get(uid, {}).get('text', '')}"
                for lb, uid in zip(ans.get("labels", []),
                                   [u.get("unit_id", "")
                                    for u in ret.get("retrieved", [])[:context_units]])
            ) if ans.get("labels") else ""
            faith_result = judge_faithfulness_call(
                passages, answer_text, jmodel, joptions, jthink)
            faith_result["not_found_case"] = False

        # ── Call B: accuracy ─────────────────────────────────────────
        acc_result = judge_accuracy_call(
            q.get("question", ""), q.get("reference_answer", ""),
            answer_text, jmodel, joptions, jthink)

        # ── Combine flags ────────────────────────────────────────────
        flags = []
        if faith_result.get("flag"):
            flags.append(f"faithfulness:{faith_result['flag']}")
        if acc_result.get("flag"):
            flags.append(f"accuracy:{acc_result['flag']}")
        combined_flag = "; ".join(flags) if flags else None

        if combined_flag:
            null_count += 1

        judgements.append({
            "question_id": qid,
            "statements": faith_result.get("statements"),
            "verdicts": faith_result.get("verdicts"),
            "faithfulness": faith_result.get("faithfulness"),
            "not_found_case": faith_result.get("not_found_case", False),
            "accuracy": acc_result.get("accuracy"),
            "reason": acc_result.get("reason"),
            "flag": combined_flag,
        })

        if i % 5 == 0:
            log.info("  %s: %d/%d judged", config_name, i, len(answers))

    out_path = ctx.run_dir / "judgements.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for j in judgements:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    # Summary
    valid_faith = [j["faithfulness"] for j in judgements
                   if j["faithfulness"] is not None]
    valid_acc = [j["accuracy"] for j in judgements
                 if j["accuracy"] is not None]
    not_found_count = sum(1 for j in judgements if j["not_found_case"])

    summary = {
        "config_name": config_name,
        "n": len(judgements),
        "faithfulness_mean": round(sum(valid_faith) / len(valid_faith), 4)
                            if valid_faith else None,
        "accuracy_mean": round(sum(valid_acc) / len(valid_acc), 4)
                         if valid_acc else None,
        "null_count": null_count,
        "not_found_count": not_found_count,
    }
    (ctx.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    ctx.log(f"judged={len(judgements)} nulls={null_count} "
            f"not_found={not_found_count}")
    ctx.finish("ok", f"{config_name}: {len(judgements)} judged")

    print(f"  {config_name}: faith={summary['faithfulness_mean']}, "
          f"acc={summary['accuracy_mean']}, nulls={null_count}, "
          f"not_found={not_found_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
