---
description: Check promotion gate for a candidate and update frontier only if all gates pass
argument-hint: "<candidate-label>"
allowed-tools: Bash(python scripts/pt_promote.py *) Bash(python scripts/check_posttrain_integrity.py *) Bash(cat *) Bash(git diff *) Read
disable-model-invocation: true
---
# PostTrain AutoResearch — promote

Candidate is `$ARGUMENTS`; if empty, use `latest`. Set `CANDIDATE` to that parsed value before running the shell example.

Run promotion gate:

```bash
python scripts/pt_promote.py \
  --config configs/posttrain/current.json \
  --candidate "$CANDIDATE" \
  --format markdown
```

Then integrity:

```bash
python scripts/check_posttrain_integrity.py --format markdown
```

Report:

- pass/fail;
- exact metrics and failed gates;
- artifact paths;
- whether `outputs/posttrain/frontier.json` changed;
- human-review risks before release.

Do not commit weights or push to Hugging Face in this command.
