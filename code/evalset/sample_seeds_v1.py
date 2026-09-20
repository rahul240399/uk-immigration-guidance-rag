"""
Sample seeds v1 for all tiers (T1, T2, T3). No LLM calls.

Usage::
    python -m code.evalset.sample_seeds_v1 --config config/paths.yaml --evalset config/evalset.yaml
"""
import argparse, csv, hashlib, json, math, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import yaml

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def _eligible(r, xt, xc, mw):
    if r["text_type"] in xt or r["text_type"] == "deleted": return False
    if r.get("parse_confidence") in xc: return False
    if len(r.get("text","").split()) < mw: return False
    return True

def _attached_tree(pid, children_of, records, max_depth=10):
    result, queue, visited = [], children_of.get(pid,[]), {pid}
    for _ in range(max_depth):
        if not queue: break
        nq = []
        for cid in queue:
            if cid in visited: continue
            visited.add(cid)
            rec = records.get(cid)
            if rec and rec["text_type"] != "deleted":
                result.append(cid)
                nq.extend(children_of.get(cid,[]))
        queue = nq
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/paths.yaml")
    ap.add_argument("--evalset", default="config/evalset.yaml")
    ap.add_argument("--routes", default="config/routes.yaml")
    args = ap.parse_args()

    with open(args.config) as f: pcfg = yaml.safe_load(f)
    with open(args.evalset) as f: ecfg = yaml.safe_load(f)
    with open(args.routes) as f: rcfg = yaml.safe_load(f)

    processed = Path(pcfg["processed"])
    evalset_dir = Path(pcfg["evalset"]); evalset_dir.mkdir(parents=True, exist_ok=True)
    seed = ecfg["seed"]; counts = ecfg["counts"]
    xt = set(ecfg.get("exclude_text_types",[])); xc = set(ecfg.get("exclude_confidence",[]))
    mw = ecfg.get("min_words",15)

    rsecs = {r: set(rc.get("rules_sections",[])) for r,rc in rcfg.get("routes",{}).items()}
    cc = set(rcfg.get("cross_cutting",{}).get("rules_sections",[]))
    all_rsecs = set(); [all_rsecs.update(s) for s in rsecs.values()]

    # Load records
    recs = {}
    with open(processed/"rules-paragraphs_v1.jsonl") as f:
        for l in f: r=json.loads(l); recs[r["paragraph_id"]]=r
    grecs = {}
    gp = processed/"guidance-paragraphs_v1.jsonl"
    if gp.exists():
        with open(gp) as f:
            for l in f: r=json.loads(l); grecs[r["paragraph_id"]]=r

    children_of = defaultdict(list)
    for pid,r in recs.items():
        if r.get("attached_to"): children_of[r["attached_to"]].append(pid)
    has_attached = set(children_of.keys())

    # Links
    rplinks = defaultdict(list)
    with open(processed/"rules-crossrefs_v2.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["target_type"]=="paragraph" and row["status"]=="resolved" and row.get("resolved_to"):
                rplinks[row["from_paragraph_id"]].append(row["resolved_to"])
    gplinks = defaultdict(list)
    glp = processed/"guidance-crossrefs_v2.csv"
    if glp.exists():
        with open(glp, newline="") as f:
            for row in csv.DictReader(f):
                if row["target_type"]=="paragraph" and row["status"]=="resolved" and row.get("resolved_to"):
                    gplinks[row["from_paragraph_id"]].append(row["resolved_to"])

    rows = []; pools_m = {}; sc = [0]

    def _sample(rname, tier, lt, pool, target):
        od = math.ceil(1.5*target)
        pool_s = sorted(pool, key=lambda x: x["gids"][0])
        rng = random.Random(f"{seed}:{rname}:{tier}:{lt}")
        rng.shuffle(pool_s)
        drawn = pool_s[:od]
        pools_m[f"{rname}_{tier}_{lt}"] = {"pool":len(pool), "drawn":len(drawn)}
        for rank, item in enumerate(drawn, 1):
            sc[0] += 1
            rows.append({"seed_id":f"S{sc[0]:04d}","route":rname,"tier":tier,"link_type":lt,
                "gold_paragraph_ids":";".join(item["gids"]),
                "sections":";".join(item["secs"]),"draw_rank":rank})

    # T1
    for rn in ["skilled_worker","student","graduate","visitor","family"]:
        tc = counts[rn].get("T1",0)
        if not tc: continue
        pool = []
        for pid,r in recs.items():
            if r["section_base_path"] not in rsecs[rn]: continue
            if not _eligible(r,xt,xc,mw): continue
            if r["text_type"] not in ("rule","narrative"): continue
            if pid in has_attached: continue
            if r["text"].rstrip().endswith(":") or r["text"].rstrip().endswith("-"): continue
            pool.append({"gids":[pid],"secs":[r["section_base_path"]]})
        _sample(rn,"T1","",pool,tc)

    # T2
    for rn in ["skilled_worker","student","graduate","visitor","family"]:
        tc = counts[rn].get("T2",0)
        if not tc: continue
        pool = []
        for pid,r in recs.items():
            if r["section_base_path"] not in rsecs[rn]: continue
            if r["text_type"]!="rule": continue
            if not _eligible(r,xt,xc,mw): continue
            att = _attached_tree(pid,children_of,recs)
            if len(att)>5 or len(att)<1: continue
            grp = [pid]+att
            if not(2<=len(grp)<=6): continue
            pool.append({"gids":grp,"secs":[r["section_base_path"]]})
        _sample(rn,"T2","",pool,tc)

    # T3 rules_para
    for rn in ["skilled_worker","student","graduate","visitor","family"]:
        tc = counts[rn].get("T3_rules_para",0)
        if not tc: continue
        pool = []
        for pid,r in recs.items():
            if r["section_base_path"] not in rsecs[rn]: continue
            if not _eligible(r,xt,xc,mw): continue
            for tid in rplinks.get(pid,[]):
                tr = recs.get(tid)
                if not tr or tr["text_type"]=="deleted": continue
                if tr["section_base_path"]==r["section_base_path"]: continue
                ta = _attached_tree(tid,children_of,recs)
                if len(ta)>5: continue
                tg = [tid]+ta; gold = [pid]+tg
                if not(2<=len(gold)<=7): continue
                pool.append({"gids":gold,"secs":sorted(set([r["section_base_path"],tr["section_base_path"]]))})
                break
        _sample(rn,"T3","rules_para",pool,tc)

    # T3 guidance_para
    for rn in ["skilled_worker","student","graduate","visitor","family"]:
        tc = counts[rn].get("T3_guidance_para",0)
        if not tc: continue
        pool = []
        for pid,gr in grecs.items():
            rf = gr.get("route",[])
            if isinstance(rf,list):
                if rn not in rf: continue
                if rf==["cross_cutting"]: continue
            else:
                if rf!=rn: continue
            if not _eligible(gr,xt,xc,mw): continue
            for tid in gplinks.get(pid,[]):
                tr = recs.get(tid)
                if not tr or tr["text_type"]=="deleted": continue
                ta = _attached_tree(tid,children_of,recs)
                if len(ta)>5: continue
                tg = [tid]+ta; gold = [pid]+tg
                if not(2<=len(gold)<=7): continue
                pool.append({"gids":gold,"secs":sorted(set([gr.get("section_base_path",""),tr["section_base_path"]]))})
                break
        _sample(rn,"T3","guidance_para",pool,tc)

    # Write seeds CSV
    sp = evalset_dir/"2026-09-19_evalset_seeds_v1.csv"
    with open(sp,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed_id","route","tier","link_type","gold_paragraph_ids","sections","draw_rank"])
        w.writeheader(); w.writerows(rows)

    # Manifest
    manifest = {"seed":seed,"evalset_yaml_sha256":_sha(args.evalset),
        "corpus_sha256":_sha(processed/"rules-paragraphs_v1.jsonl"),
        "links_sha256":_sha(processed/"rules-crossrefs_v2.csv"),
        "guidance_links_sha256":_sha(glp) if glp.exists() else "",
        "pools":pools_m,"total_seeds":len(rows)}
    mp = evalset_dir/"2026-09-19_evalset_seeds_v1_manifest.json"
    mp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))

    print(f"Seeds: {len(rows)} rows -> {sp}")
    for k in sorted(pools_m):
        p=pools_m[k]; print(f"  {k:40s} pool={p['pool']:5d} drawn={p['drawn']:3d}")

if __name__=="__main__": main()
