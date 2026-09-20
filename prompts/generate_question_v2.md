# Generate a question and reference answer (v2)

You are given paragraphs from the UK Immigration Rules or caseworker guidance for the **{route}** route.

## Rules for the question

- Name the route exactly once using its display name ("{route}"). Never name an Appendix, Part, section, paragraph number, or rule identifier. Never use drafting variables such as "(P)", "(C)", "person (P)".
- The question is what an applicant would ask, answerable from the given text alone.
- The reference answer states the substantive content in full. It never ends in a colon, never points to "the following" or "the requirements listed in".
{tier_instruction}
- If no such question exists for the given text, output exactly: NONE

## Input paragraphs

{paragraphs}

## Output format (JSON only, or the single word NONE)

```json
{{
  "question": "...",
  "reference_answer": "..."
}}
```
