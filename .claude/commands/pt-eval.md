---
description: Evaluate a seed, specialist, or merged candidate with the German-core suite
argument-hint: "[dry|real] <model-or-candidate-label>"
allowed-tools: Bash(python scripts/pt_eval.py *) Bash(python scripts/pt_score.py *) Bash(cat *) Bash(find *) Read
disable-model-invocation: true
---
# PostTrain AutoResearch — eval

Parse `$ARGUMENTS`: mode default `dry`; candidate/model is the remaining argument, default `latest`. Set `MODE_FLAG=--dry-run` or `MODE_FLAG=--real`, and `CANDIDATE` to the parsed model/label.

Run:

```bash
python scripts/pt_eval.py \
  --config configs/posttrain/current.json \
  --candidate "$CANDIDATE" \
  --out outputs/posttrain/evals \
  "$MODE_FLAG"
```

Then score if possible:

```bash
python scripts/pt_score.py \
  --config configs/posttrain/current.json \
  --candidate "$CANDIDATE" \
  --out outputs/posttrain/score-latest.json \
  "$MODE_FLAG" || true
```

Report exact artifact paths and gate implications. Do not promote here.
