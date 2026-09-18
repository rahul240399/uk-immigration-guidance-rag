# Generate a question and reference answer

You are given one or more paragraphs from the UK Immigration Rules. Generate exactly one question that can be answered using only the given paragraphs, and a reference answer.

## Rules

- The question must be specific and answerable from the text alone.
- The reference answer must cite the paragraph identifiers from the text.
- Do not add information not in the given paragraphs.
- Output JSON only, no commentary.

## Input paragraphs

{paragraphs}

## Output format (JSON)

```json
{
  "question": "...",
  "reference_answer": "..."
}
```
