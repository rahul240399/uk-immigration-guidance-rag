# F25 -> T2b: hand-seed builder, sampler fixes, synthetic set v1
Raised by the evaluation-set chat, 2026-09-19, under D52, D53, D60, D61. Paste into the pipeline chat.
Two runs maximum; what fails after the second run is reported here as a limitation.
Config: evalset.yaml 2026-09-19 (supersedes 31a7105). Seed 20260918 unchanged.

## Run order
1. Part A (hand seeds) first and uploaded to the evaluation-set chat as soon as it passes, so hand-writing starts today.
2. Part B (sampler fixes, prompt v2, regeneration of all 120 synthetic items as v1). v0 is retained as attempt 1, nothing carried forward.

## Part A: hand-seed builder (no LLM call)
Inputs: rules-paragraphs_v1.jsonl, rules-crossrefs_v2.csv, config/routes.yaml, evalset.yaml 2026-09-19.
Source eligibility: non-deleted; not list_item / table_row; not low confidence; >= 15 words; in one of the route's rules_sections.
Link: status resolved, target_type section, target section != source section.
  rules_section_xcut (e1): target section in routes.yaml cross_cutting.
  rules_section_other (e2): any other section.
Draw per route: e1 pool first up to 2 x hand count, then e2 fills to 2 x hand count; random within pool under the seed; one row per source id (a source with several section links is drawn once, all its targets listed).
Output: evalset/2026-09-19_evalset_seeds-hand_v0.csv
  columns: seed_id, route, link_type, source_paragraph_id, source_rule_ref, source_section, target_section, target_section_title, draw_rank
  plus evalset/2026-09-19_evalset_seeds-hand_v0_manifest.json: pool per route x link_type, drawn per route x link_type, input hashes.
Acceptance A:
  A1 rows = 59: skilled_worker 20, student 10, graduate 9 (pool e1 4 + e2 5), visitor 10, family 10.
  A2 pools reported equal e1 22 / 14 / 4 / 3 / 59 and e2 50 / 11 / 5 / 23 / 147 (F22 counts), or every difference explained by the eligibility rule that caused it.
  A3 every source id exists, non-deleted; every target section is one of the 101 sections; no source id repeated; no source in its own target section.
  A4 e1 rows precede e2 rows within each route (draw_rank); visitor has 3 e1 rows, graduate 4.

## Part B: sampler fixes and regeneration (D61 i-iv)
B-i  T1 pool: records with no attached record (no record has attached_to = this id) and text not ending with ":" or "-"; other exclusions unchanged.
B-ii T2 group: root rule (>= 15 words, not low confidence) plus every non-deleted attached record to full depth; attached records are not length-filtered; groups with more than 5 attached records are excluded from the pool, never truncated; gold = the whole group, 2 to 6 ids.
B-iii T3 groups:
  rules_para: source = any eligible record (may be a subparagraph) with a resolved paragraph-level link (rules-crossrefs_v2, target_type paragraph) to a target in another section; target group = target record + its non-deleted attached records under the B-ii rule (0 to 5 attached, else excluded); gold = source + target group, 2 to 7 ids.
  guidance_para: source = guidance record (guidance-paragraphs_v1; route-tagged, route != cross-cutting; same eligibility) with a resolved paragraph-level link (guidance-crossrefs_v2) to a Rules record; target group as above; gold = source + target group.
B-iv Prompt v2 (generate_question_v2.md, sha256 logged); requirements the prompt must state:
  - name the route exactly once using the display name from routes.yaml; never name an Appendix, Part, section, paragraph number or rule identifier; never use drafting variables such as "(P)", "(C)", "person (P)";
  - the question is what an applicant would ask, answerable from the given text alone; the reference answer states the substantive content in full, never ends in a colon, never points to "the following" or "the requirements listed in";
  - T2: the answer covers every subparagraph that answers the question;
  - T3: the question must need both the source and the target (not answerable from either alone); the answer combines both;
  - if no such question exists for the given text, output exactly NONE.
  Sampler over-draws 1.5 x each stratum count and generates in draw order until the count is met; NONE rate reported per stratum.
Outputs: evalset/2026-09-19_evalset_questions-synthetic_v1.csv (columns per D53: question_id, route, tier, link_type, question, reference_answer, gold_paragraph_ids, source, generator_model, embedding_model, validated_by_student, notes), seeds v1 csv, manifest v1 (pools and drawn per stratum, NONE counts, prompt v2 sha256, model digest, seed, evalset.yaml sha256, corpus and both link-table hashes).
Acceptance B (counts, no corpus text in Kiro's report):
  B1 rows per stratum equal the D52 counts: T1 12 x 5; T2 9/9/4/9/9; T3 rules_para SW 2, family 5; T3 guidance_para SW 4, student 3, visitor 3, family 3; total 120. A stratum below its count is allowed only when its pool is smaller, with the pool reported.
  B2 every gold id exists in v1 (Rules or guidance), none deleted, none list_item / table_row / low confidence.
  B3 T1: exactly 1 gold id; it has no attached record; its text does not end with ":" or "-".
  B4 T2: gold set equals root + all non-deleted attached records (full depth); 2 <= |gold| <= 6. Checked for every row.
  B5 T3: gold set equals source + complete target group; source and target in different sections (rules_para) or the source is a guidance record (guidance_para); the link exists in the named v2 table with status resolved and target_type paragraph; 2 <= |gold| <= 7.
  B6 F09 regex on question text, case-insensitive, 0 matches:
       \b(Appendix|Part\s+\d|paragraph\s+\d|para\.?\s*\d|[A-Z]{1,6}\s?\d+\.\d+|[A-Z]+-[A-Z]+\.\d|\([A-Z]\))
     and every question contains its route's display name (Skilled Worker, Student, Graduate, Visitor, Family).
  B7 0 reference answers ending with ":"; 0 reference answers under 10 words; 0 questions under 8 words.
  B8 validated_by_student = N for all rows; source = prompt-v1 replaced by prompt-v2 (or ragas if that path is used; state which).
  B9 manifest carries the provenance fields listed above and the NONE count per stratum.

## Upload back to the evaluation-set chat
After A: seeds-hand_v0.csv and its manifest. After B: questions-synthetic_v1.csv, seeds v1, manifest v1, generate_question_v2.md, and the acceptance-count report. If any pool is below its D52 count, say which and by how much.
