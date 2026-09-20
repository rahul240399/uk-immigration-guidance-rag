# Remaining pipeline code: T2c, T6b, T7 — specification and acceptance
Version 2026-09-20. Owner: pipeline chat. Binding decisions: D33, D35, D44, D47, D49–D51, D53–D58, D62; flags F24, F27. Repository state at writing: origin HEAD 54daa03; code/eval holds run_grid.py, score.py, diagnose.py, select_configs.py, run_generation.py, run_judge.py, export_rating.py, agreement.py; none of the T2c or T7 modules exist. Rules for every part: read .kiro/steering/*; no git commands; no network except localhost Ollama; no corpus text in stdout or reports; every script takes --config config/paths.yaml and registers a run through code/common/run_registry.py with the sha256 of every input in params; outputs written atomically (temp file, rename); two attempts per part, then the remainder is reported as a limitation. Kiro does not launch generation or judging; where a long run is needed it prints the command for the student.

## Part T2c — question set v1: merge, near-duplicates, covariates, freeze

### T2c-1 merge_questions.py
`--synthetic <validated csv> --hand <hand csv> --out evalset/<date>_evalset_v1-candidate.csv`
- Keep synthetic rows with validated_by_student == Y; report the N count per stratum and the notes column as counts only.
- Append hand rows: source must be `hand`, tier T3, link_type in {rules_section_xcut, rules_section_other}, validated Y; reject the file otherwise.
- Assert: every gold id exists in rules-paragraphs_v1 or guidance-paragraphs_v1 and is not deleted; question_ids unique; D53 columns exactly; order synthetic by question_id then hand by seed order.
- Manifest: counts per route × tier × link_type and per source; sha256 of both inputs and the output.

### T2c-2 near_duplicates.py (D57)
`--questions <candidate csv> --out evalset/<date>_evalset_neardup_v1.csv`
- Embed every question with encode_query (code/index/embed.py); cosine over all pairs.
- Output the union of: pairs with cosine ≥ 0.90; the 20 most similar pairs; pairs with an identical gold_paragraph_ids string.
- Columns: pair_id, qid_a, qid_b, cosine, same_gold, reason_set (threshold|top20|same_gold, joined with +), route_a, route_b, tier_a, tier_b, question_a, question_b, verdict (empty), reworded_question_b (empty), note (empty).
- Report: size of each set and of the union.

### T2c-3 covariates.py (D58)
`--questions <candidate csv> --out evalset/<date>_evalset_covariates_v1.csv`
- Run para-bge-bm25-flat retrieval (the existing BM25 index, depth 100, the run_grid.py code path) on the candidate questions into a registered run (stage grid, config para-bge-bm25-flat-covariates).
- One row per question_id: anchor_id (first Rules paragraph id in gold order; for guidance_para items the first Rules id after the guidance source); drafting_style, section_records, siblings_same_heading, links_out_v2 from processed/rules-covariates_v1.csv by anchor; n_gold; link_type; gold_tokens (sum of unit_token_count over gold ids from index/units_v1.jsonl); bm25_gold_rank (rank of the first retrieved unit carrying any gold id, "absent" beyond 100); bm25_overlap (share of the question's BM25 tokens present in the anchor unit's text, same tokenizer rule as the index config); lexically_easy (1 when the first retrieved unit carries a gold id, i.e. MRR = 1, else 0).
- Report: lexically_easy count per tier; rows with empty anchor (must be 0).

### T2c-4 freeze_evalset.py
`--candidate <csv> --neardup <verdict csv> --covariates <csv> --out evalset/<date>_evalset_v1.csv`
- Verdict handling: `duplicate` drops the row with the later question_id of the pair; `reworded` replaces question_b's text with reworded_question_b; `distinct` keeps both; any pair with an empty verdict stops the freeze.
- Floor check per route × tier against fallback_floor in config/evalset.yaml (T3 synthetic and hand counted together as T3); strata below the floor are listed, not fatal; the manifest records them as limitations.
- Write the frozen csv; its manifest (counts per route × tier × link_type × source, sha256 of every input and of the output, removed pairs with reasons, floor check); copy the covariates file as evalset/<date>_evalset_covariates_v1_frozen.csv restricted to the frozen ids.
- Register the run (stage evalset, config freeze-v1). The frozen file is never edited; the student tags evalset-v1.

### Tests T2c (synthetic fixtures)
merge assertions and rejection cases; pair-set union with the top-20 rule; anchor rule for guidance_para; lexically_easy from a toy retrieved list; verdict application (all three values, empty verdict stops); floor check output.

### Acceptance T2c
merged rows = validated synthetic + 30 hand, all gold ids in the corpus, ids unique; near-duplicate file contains at least the 20 top pairs and every same-gold pair; covariates has one row per question with no empty anchor; freeze manifest lists the floor check with the known-short strata only (student T2, family rules_para, skilled_worker rules_para).

## Part T6b — F27 fixes before the real generation run

### T6b-1 run_generation.py, context assembly
- Read generator settings from config/judge.yaml `generation` block (model, digest, options, think, context_units, budget_tokens); judge settings from the `judge` block and `judge_digest`. Assert the Ollama model digest matches before generating.
- linkexp: after the first context_units units, render every appended_targets id (deduplicated, in unit order) as a passage under the label `Cross-referenced: <rule_ref or paragraph_id>`. Assert, per question, that the number of Cross-referenced passages equals the number of distinct appended_targets in those units; record both counts in answers.jsonl.
- pcreturn: render the unit's paragraph_ids (the D44 capped set) in seq order from index/units_v1.jsonl, heading line once.
- answers.jsonl gains: n_units, n_appended, context_tokens (bge tokenizer), labels, cited_labels.

### T6b-2 run_judge.py, auditable faithfulness
- Call A output must be stored: `statements` (list of strings, the answer's atomic claims as extracted) and `verdicts` (list of `supported` / `unsupported`, same length). faithfulness = supported / total. A "Not found in the provided rules." answer: statements = [], faithfulness = 1.0 when no gold id is among the passage ids, else 0.0; recorded as `not_found_case`.
- Call B stored as accuracy (1 / 0.5 / 0), reason (one sentence).
- Malformed output (unparseable, list length mismatch, value outside the set) → the score is null and `flag` names the problem; never a guessed value. Nulls counted in summary.json.
- `--verify <judge folder>`: recompute faithfulness from statements/verdicts for every row and compare with the stored value; exit 1 on any difference.

### T6b-3 export_rating.py (D55, D56)
- One draw of rated_sample_per_config question_ids from the frozen evalset, stratified by tier as rated_by_tier in config/judge.yaml (20 / 13 / 17), random within tier with rated_sample_seed; the same ids under each of the three D49 configurations → 150 rows.
- Sheet rows in random order (seed); columns: sheet_row, question, context (the passages as given to the generator), answer, reference_answer, faithfulness_hand, accuracy_hand, copies_or_answers, notes — no question_id, no configuration, no judge score.
- Key file evalset/<date>_rating_key_v1.csv: sheet_row, question_id, config, generate_run, judge_run; kept out of the sheet the student rates.
- Report: rows per configuration and tier, key file path.

### T6b-4 agreement.py (D54)
- Inputs: the filled sheet, the key file, the three judgements folders, config/judge.yaml.
- Accuracy: Cohen's kappa with quadratic weights on levels [0, 0.5, 1] between accuracy_hand and judge accuracy; faithfulness: unweighted kappa between faithfulness_hand (Y = a claim not supported present) and judge_binarised (share < 1.0 → Y); also exact agreement, F1 for accuracy == 1, and the confusion matrices; per configuration and pooled; null judge scores excluded and counted.
- Compare each kappa with agreement.threshold (0.60); below → `unvalidated: true` for that score. Write results/<stamp>_agreement.json; register the run.

### Tests T6b
context assembly for the three architectures on a synthetic retrieved.jsonl + units file, including the Cross-referenced count assertion; judge parsing (well-formed, malformed, length mismatch); --verify on a hand-built folder; rating sheet has no id, config or judge column and 150 rows from 50 ids × 3; key file round trip; kappa on a toy pair with a hand-computed value.

### Acceptance T6b (dry run, 10 questions × 3 configurations from the frozen set)
linkexp answers.jsonl shows n_appended > 0 for at least one question and Cross-referenced count equal to n_appended in every row; judgements carry statements and verdicts and --verify passes; rating sheet 150 rows, no forbidden columns; agreement.py runs on a filled toy sheet.

## Part T7 — results stage

### T7-1 make_results.py
`--config config/paths.yaml --grid-manifest <results/<stamp>_grid_manifest.json> --evalset <frozen csv> --covariates <frozen covariates csv> [--judge <agreement json> --judgements <folders>] --out results/<date>_results_v1/`
Inputs: results.json of every v1 grid run (per-question metrics), retrieved.jsonl for cost columns, the frozen evalset and covariates, optionally the judge outputs and agreement.
Tables (each as csv and markdown, numbered):
1. `t1_grid`: one row per configuration: n, Recall@5, Recall@10, MRR, nDCG@10, budget_640 — mean and 95% bootstrap CI (1,000 resamples, seed 20260918), wall time.
2. `t2_factors`: marginal means per level of chunking, retrieval, architecture; primary metric per factor per D51 (fixed-k for chunking and retrieval, budget_640 for architecture), the other metrics beside it.
3. `t3_paired`: paired Wilcoxon signed-rank per question (scipy) for every configuration against the baseline para-bge-bm25-flat and against the best configuration by the factor's primary metric; median paired difference with bootstrap CI; Holm correction over all tests in the table; n and the count of tied pairs.
4. `t4_tiers`: Recall@10 and budget_640 by tier (T1, T2, T3) and by T3 link_type for the five best configurations by budget_640 and the baseline.
5. `t5_style_easy`: the same split by drafting_style of the anchor (new / old / list) and by lexically_easy.
6. `t6_cost`: linkexp: mean paragraphs added, mean tokens added, unexpanded section references, absent-target links, per tier; pcreturn: mean returned tokens; all configurations: mean context tokens at k = 5.
7. `t7_rag` (when judge outputs given): per D49 configuration: mean faithfulness and accuracy with CIs, null counts, "Not found" rate; agreement statistics per score with the threshold verdict.
8. `t8_runs`: run ids, commits, code tree hashes, index and evalset hashes used — the provenance table for the appendix.
- Results manifest: every input file with sha256, every run folder read, the config hashes, and the git commit of the results run.
- `--freeze`: copy the output folder to results/results_v1/ with RESULTS_MANIFEST.json; the student tags results-v1; the folder is never edited.

### Tests T7
Wilcoxon and Holm on toy per-question data with a hand-computed p-value ordering; bootstrap CI determinism under the seed; marginal means on a toy grid; table shapes and column sets; missing run folder → error naming it.

### Acceptance T7
Eight tables produced from the v1 runs; t1 has 20 rows; t3 lists every test with raw and Holm-adjusted p; manifest lists 20 run folders; --freeze creates results_v1 with the manifest; re-running on the same inputs gives byte-identical tables.

## Run order after the code exists (student runs; one process at a time)
1. T2c-1..3 on the validated synthetic + hand files → verdicts in the evaluation-set chat → T2c-4 freeze → tag evalset-v1.
2. run_grid.py on the frozen file (20 configurations; about 1.5 h with reranking) → score.py --verify on all 20 → diagnose.py on v1.
3. T6b dry run (10 questions) → then run_generation.py on the three D49 configurations (about 5.5 h, detached, overnight) → run_judge.py (about 3 h) → run_judge.py --verify → export_rating.py → student rates → agreement.py.
4. make_results.py → --freeze → tag results-v1. Pipeline stops.
