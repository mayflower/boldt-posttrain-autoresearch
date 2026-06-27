# Boldt DC 1B German Post-Training AutoResearch Pack

This pack adds a Claude Code project workflow for iterative post-training of:

- base/post-DPO seed: `mayflowergmbh/boldt-dc-1b-german-it-16k-dpo`
- data source family: German rows/configs/splits from Hugging Face org `openeurollm`
- method: Liquid-AI-style branch → specialist train → merge search → eval → promote loop for small models

Drop these files into the root of your training repository, open Claude Code there, then start with:

```text
/pt-orient
/pt-data dry
/pt-baseline dry
/pt-run 3 dry
```

For actual GPU work, implement or adapt the scripts described in `docs/posttrain-script-contracts.md`, then run e.g.:

```text
/pt-run 2 real
```

The `.claude` commands are intentionally orchestration-focused. They do not assume one fixed trainer; they define the auditable loop and the contracts your scripts must satisfy.
