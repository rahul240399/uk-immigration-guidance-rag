# Judge faithfulness

Given the passages and the answer, extract every atomic factual statement from the answer.
For each statement, judge whether it is SUPPORTED or UNSUPPORTED by the passages.

## Passages

{passages}

## Answer

{answer}

## Output (JSON only)

```json
{
  "statements": [
    {"statement": "...", "verdict": "supported"},
    {"statement": "...", "verdict": "unsupported"}
  ]
}
```
