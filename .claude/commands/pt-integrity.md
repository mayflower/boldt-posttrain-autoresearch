---
description: Check protected surfaces and git cleanliness for the post-training loop
argument-hint: "[--base-ref REF]"
allowed-tools: Bash(python scripts/check_posttrain_integrity.py *) Bash(git status *) Bash(git diff *) Bash(git diff --name-only *) Read
disable-model-invocation: true
---
# PostTrain AutoResearch — integrity

Run:

```bash
python scripts/check_posttrain_integrity.py --format markdown $ARGUMENTS || true
git status --short
git diff --name-only
```

Explain any violation in plain language. If protected surfaces changed during an automated loop, recommend reverting those changes before continuing.
