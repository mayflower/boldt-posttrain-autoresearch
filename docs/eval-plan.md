# German Post-Training Evaluation Plan

Establish baseline before any training. Keep evaluation scripts, prompt templates, and held-out data frozen after baseline.

## Core tasks

- German lm-eval tasks: `arc_de`, `hellaswag_de`, `m_mmlu_de`, `truthfulqa_de_mc2`, `belebele_deu_Latn`.
- German instruction smoke set: helpfulness, concision, refusal, grounded QA, code explanation, math reasoning, table/JSON formatting.
- German format following: exact word/character/list constraints, JSON-only, citation-style answer, no Markdown when forbidden.
- Language retention: German prompt should yield German answer unless task explicitly requests translation.
- Response suppression: empty answers, ultra-short non-answers, generic refusals.
- Long-context smoke: 8k–16k German summarization and multi-document retrieval-style QA.

## Promotion requirements

- All artifacts present.
- Verified clean leakage status.
- Known and usable license status.
- Aggregate score improves over the current winner.
- No task regression beyond tolerance.
- No safety/over-refusal/English-bleed regression.
- Human review before public release.
