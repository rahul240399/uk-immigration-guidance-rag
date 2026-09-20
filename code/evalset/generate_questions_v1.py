"""
Generate synthetic questions v1 from seeds. Streams rows; supports --resume.

Usage::
    python -m code.evalset.generate_questions_v1 --config config/paths.yaml \
        --evalset config/evalset.yaml --seeds <seeds_v1.csv> [--resume]
"""
import argparse, csv, hashlib, json, logging, math, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import requests, yaml
from code.common.run_registry import load_paths, start_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
OLLAMA = "http://localhost:11434/api/generate"
DISPLAY = {"skilled_worker":"Skilled Worker","student":"Student","graduate":"Graduate","visitor":"Visitor","family":"Family"}

B6_RE = re.compile(r'\b(Appendix|Part\s+\d|paragraph\s+\d|para\.?\s*\d|[A-Z]{1,6}\s?\d+\.\d+|[A-Z]+-[A-Z]+\.\d|\([A-Z]\))', re.IGNORECASE)

def validate_row(row, seen_questions):
    violations = []
    q = row["question"]; a = row["reference_answer"]; rn = row["route"]
    dn = DISPLAY.get(rn, "")
    if B6_RE.search(q): violations.append("B6_regex_match")
    if dn not in q: violations.append("route_name_missing")
    if q.lower().count(dn.lower()) > 1: violations.append("route_name_repeated")
    if len(q.split()) < 8: violations.append("question_under_8_words")
    if len(a.split()) < 10: violations.append("answer_under_10_words")
    if a.rstrip().endswith(":"): violations.append("answer_ends_colon")
    if "the following" in a.lower(): violations.append("answer_the_following")
    if "listed in" in a.lower(): violations.append("answer_listed_in")
    ql = q.lower()
    if ql in seen_questions: violations.append("duplicate_question")
    return violations

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _call(prompt,model,options,think):
    try:
        r=requests.post(OLLAMA,json={"model":model,"prompt":prompt,"stream":False,"think":think,"options":options},timeout=180)
        if r.status_code==200: return r.json()
    except Exception as e: log.warning("Ollama: %s",e)
    return None
def _parse_json(text):
    m=re.search(r'```json\s*(\{.*?\})\s*```',text,re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m=re.search(r'\{[^{}]*"question"[^{}]*"reference_answer"[^{}]*\}',text,re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None

def _attached_tree(pid, children_of, records, max_depth=10):
    result, queue, visited = [], children_of.get(pid,[]), {pid}
    for _ in range(max_depth):
        if not queue: break
        nq = []
        for cid in queue:
            if cid in visited: continue
            visited.add(cid)
            rec = records.get(cid)
            if rec and rec["text_type"]!="deleted":
                result.append(cid)
                nq.extend(children_of.get(cid,[]))
        queue = nq
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="config/paths.yaml")
    ap.add_argument("--evalset",default="config/evalset.yaml")
    ap.add_argument("--seeds",required=True)
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--repair",action="store_true")
    ap.add_argument("--topup",default=None,help="Path to consolidated CSV to top up")
    args=ap.parse_args()

    paths=load_paths(args.config)
    with open(args.evalset) as f: ecfg=yaml.safe_load(f)
    processed=Path(paths["processed"]); evalset_dir=Path(paths["evalset"])
    evalset_dir.mkdir(parents=True,exist_ok=True)

    # Lock file
    lock_path = evalset_dir / ".generate.lock"
    if lock_path.exists():
        print(f"ERROR: lock file exists: {lock_path}"); sys.exit(1)
    lock_path.write_text(str(time.time()))

    seed=ecfg["seed"]; counts=ecfg["counts"]
    llm=ecfg["llm"]; model=llm["model"]; options=llm["options"]; think=llm.get("think",False)

    # Load seeds
    with open(args.seeds,newline="",encoding="utf-8") as f:
        seeds=list(csv.DictReader(f))

    # Load records
    recs={}
    with open(processed/"rules-paragraphs_v1.jsonl") as f:
        for l in f: r=json.loads(l); recs[r["paragraph_id"]]=r
    grecs={}
    gp=processed/"guidance-paragraphs_v1.jsonl"
    if gp.exists():
        with open(gp) as f:
            for l in f: r=json.loads(l); grecs[r["paragraph_id"]]=r

    children_of=defaultdict(list)
    for pid,r in recs.items():
        if r.get("attached_to"): children_of[r["attached_to"]].append(pid)

    prompt_path=Path("prompts/generate_question_v2.md")
    prompt_tpl=prompt_path.read_text(encoding="utf-8")
    prompt_sha=_sha(prompt_path)

    # Output path
    if args.topup:
        out_path = Path(args.topup)  # overwrite the consolidated CSV
    else:
        out_path=evalset_dir/"2026-09-19_evalset_questions-synthetic_v1.csv"
    fields=["question_id","route","tier","link_type","question","reference_answer",
            "gold_paragraph_ids","source","generator_model","embedding_model",
            "validated_by_student","notes"]

    # Resume/topup: load existing
    done_golds=set()
    existing_rows=[]
    if (args.resume or args.topup) and out_path.exists():
        with open(out_path,newline="",encoding="utf-8") as f:
            existing_rows=list(csv.DictReader(f))
        for r in existing_rows: done_golds.add(r["gold_paragraph_ids"])
        log.info("Resume: %d existing questions",len(done_golds))

    # Repair mode: validate existing rows, separate passing from failing
    repair_failing = []
    if args.repair and existing_rows:
        seen_q = set()
        passing = []
        for r in existing_rows:
            vs = validate_row(r, seen_q)
            if vs:
                repair_failing.append(r)
                log.info("REPAIR: %s fails: %s", r["question_id"], vs)
            else:
                passing.append(r)
                seen_q.add(r["question"].lower())
                done_golds.add(r["gold_paragraph_ids"])
        existing_rows = passing
        # Remove failing golds so they can be re-drawn
        for r in repair_failing:
            done_golds.discard(r["gold_paragraph_ids"])
        log.info("Repair: %d passing, %d failing", len(passing), len(repair_failing))

    # Open for writing (overwrite header + existing + new)
    out_fh=open(out_path,"w",newline="",encoding="utf-8")
    writer=csv.DictWriter(out_fh,fieldnames=fields)
    writer.writeheader()
    for r in existing_rows: writer.writerow(r)
    out_fh.flush()

    # Register run
    ctx=start_run("evalset","synthetic-v1-topup" if args.topup else "synthetic-v1",{
        "evalset_sha256":_sha(args.evalset),"prompt_sha256":prompt_sha,
        "model":model,"digest":llm.get("digest",""),"seed":seed,
    },paths)
    ctx.log("topup_only ignored: full draw per D61")

    # D52 stop counts per stratum
    stratum_target={}
    for rn in counts:
        for tier in counts[rn]:
            tc=counts[rn][tier]
            if tc>0:
                # Map tier names: T3_rules_para -> (T3, rules_para)
                if tier.startswith("T3_"):
                    lt=tier[3:]
                    stratum_target[(rn,"T3",lt)]=tc
                else:
                    stratum_target[(rn,tier,"")]=tc

    stratum_done=Counter()
    for r in existing_rows:
        stratum_done[(r["route"],r["tier"],r["link_type"])]+=1

    none_counts=Counter(); qc=len(existing_rows)

    # In repair mode, only process seeds whose gold matches a failing row
    repair_golds = {r["gold_paragraph_ids"] for r in repair_failing} if args.repair else set()
    # For repair: map failing gold -> original question_id to keep ids stable
    repair_qid_map = {r["gold_paragraph_ids"]: r["question_id"] for r in repair_failing} if args.repair else {}

    for s in seeds:
        rn=s["route"]; tier=s["tier"]; lt=s["link_type"]
        gids_str=s["gold_paragraph_ids"]; gids=gids_str.split(";")
        sk=(rn,tier,lt); target=stratum_target.get(sk,0)

        if stratum_done[sk]>=target: continue
        if gids_str in done_golds: continue
        # In repair mode, skip seeds not related to failing rows
        if args.repair and repair_golds and gids_str not in repair_golds: continue
        print(f"  seed {s['seed_id']} {rn}:{tier}:{lt} — generating (stratum {stratum_done[sk]}/{target})", flush=True)

        # Build text
        all_recs={**recs,**grecs}
        if tier=="T1":
            r0=all_recs.get(gids[0],{})
            texts=f"[{r0.get('rule_ref',gids[0])}]\n{r0.get('text','')}"
        elif tier=="T2":
            parts=[]
            for gid in gids:
                r0=recs.get(gid,{})
                parts.append(f"[{r0.get('rule_ref',gid)}]\n{r0.get('text','')}")
            texts="\n\n".join(parts)
        else: # T3
            src=all_recs.get(gids[0],{})
            src_text=f"Source:\n[{src.get('rule_ref',gids[0])}]\n{src.get('text','')}"
            tgt_parts=[]
            for gid in gids[1:]:
                tr=recs.get(gid,{})
                tgt_parts.append(f"[{tr.get('rule_ref',gid)}]\n{tr.get('text','')}")
            texts=src_text+"\n\nTarget:\n"+"\n\n".join(tgt_parts)

        display=DISPLAY[rn]
        ti=""
        if tier=="T2": ti="- T2: the answer covers every subparagraph that answers the question."
        elif tier=="T3": ti="- T3: the question must need both the source and the target (not answerable from either alone); the answer combines both."

        prompt=prompt_tpl.replace("{route}",display).replace("{tier_instruction}",ti).replace("{paragraphs}",texts)

        print(f"  → calling LLM for {s['seed_id']} {rn}:{tier}:{lt} ...", flush=True)
        t0=time.time()
        resp=_call(prompt,model,options,think)
        latency=round(time.time()-t0,1)
        resp_text=resp.get("response","") if resp else ""

        status="ok"
        if resp_text.strip()=="NONE" or resp_text.strip().startswith("NONE"):
            status="NONE"; none_counts[f"{rn}_{tier}_{lt}"]+=1
        else:
            parsed=_parse_json(resp_text)
            if not parsed or not parsed.get("question"):
                status="error"; none_counts[f"{rn}_{tier}_{lt}"]+=1
            else:
                qc+=1
                row={"question_id": repair_qid_map.get(gids_str, f"Q{qc:04d}"),"route":rn,"tier":tier,
                    "link_type":lt if tier=="T3" else "","question":parsed["question"],
                    "reference_answer":parsed.get("reference_answer",""),
                    "gold_paragraph_ids":gids_str,"source":"prompt-v2",
                    "generator_model":model,"embedding_model":ecfg["embeddings"]["model"],
                    "validated_by_student":"N","notes":""}
                # Validate
                seen_q = {r["question"].lower() for r in existing_rows}
                for rr in [r2 for r2 in (existing_rows or [])]: seen_q.add(rr.get("question","").lower())
                vs = validate_row(row, seen_q)
                if vs:
                    # Retry up to 2 times with violation feedback
                    retry_ok = False
                    for retry in range(2):
                        repair_prompt = prompt + f"\n\nYour previous output violated: {'; '.join(vs)}; produce a different question that satisfies every rule."
                        resp2 = _call(repair_prompt, model, options, think)
                        rt2 = resp2.get("response","") if resp2 else ""
                        p2 = _parse_json(rt2)
                        if p2 and p2.get("question"):
                            row["question"] = p2["question"]
                            row["reference_answer"] = p2.get("reference_answer","")
                            vs2 = validate_row(row, seen_q)
                            if not vs2:
                                retry_ok = True; break
                            vs = vs2
                    if not retry_ok:
                        status = "NONE-invalid"
                        none_counts[f"{rn}_{tier}_{lt}"] += 1
                        qc -= 1  # undo increment
                        ctx.log(f"NONE-invalid {s['seed_id']} after retries: {vs}")
                        continue
                writer.writerow(row); out_fh.flush()
                stratum_done[sk]+=1
                done_golds.add(gids_str)

        ctx.log(f"{s['seed_id']} {rn}:{tier}:{lt} {status} {latency}s done={dict(stratum_done)}")
        print(f"{s['seed_id']:6s} {rn:15s} {tier:4s} {lt:15s} {status:5s} {latency:5.1f}s  stratum={stratum_done[sk]}/{target}", flush=True)

    out_fh.close()

    # For topup: atomic write over the consolidated CSV
    if args.topup:
        import shutil
        with open(out_path, newline="", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
        tmp = out_path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(all_rows)
        shutil.move(str(tmp), str(out_path))

    # Rewrite manifest from file
    with open(out_path, newline="", encoding="utf-8") as f:
        final_rows = list(csv.DictReader(f))
    mname = "2026-09-20_evalset_questions-synthetic_v1_manifest.json" if args.topup else "2026-09-19_evalset_questions-synthetic_v1_manifest.json"
    manifest = {"seed": seed, "evalset_yaml_sha256": _sha(args.evalset),
        "prompt_v2_sha256": prompt_sha, "model": model, "digest": llm.get("digest", ""),
        "corpus_sha256": _sha(processed / "rules-paragraphs_v1.jsonl"),
        "links_sha256": _sha(processed / "rules-crossrefs_v2.csv"),
        "guidance_links_sha256": _sha(processed / "guidance-crossrefs_v2.csv") if (processed / "guidance-crossrefs_v2.csv").exists() else "",
        "total_questions": len(final_rows), "none_counts": dict(none_counts),
        "stratum_done": {f"{k[0]}_{k[1]}_{k[2]}": v for k, v in stratum_done.items()}}
    mp = evalset_dir / mname
    mp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Remove lock
    lock_path.unlink(missing_ok=True)

    ctx.finish("ok", f"synthetic-v1: {len(final_rows)} questions")
    print(f"\nTotal: {len(final_rows)} questions, NONE: {sum(none_counts.values())}")

if __name__ == "__main__": main()
