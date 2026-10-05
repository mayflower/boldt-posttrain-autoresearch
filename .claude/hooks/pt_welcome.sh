#!/usr/bin/env bash
# SessionStart hook: the same locked environment as the loop.
uv run --locked python - <<'PYWELCOME'
import json

try:
    from boldt_posttrain.guide import render_guide
    from boldt_posttrain.policy import load_policy

    message = render_guide(load_policy())
except Exception as exc:  # the overview must never block a session
    message = (
        "PostTrain AutoResearch: the status overview failed "
        f"({type(exc).__name__}: {exc}).\n"
        "Run /pt-orient. Order: /pt-data real, /pt-baseline real, /pt-run 1 real."
    )
print(json.dumps({"systemMessage": message}))
PYWELCOME
