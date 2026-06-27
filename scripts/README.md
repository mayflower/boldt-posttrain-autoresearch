The `.claude` commands call these `pt_*.py` scripts by the contract in
`docs/posttrain-script-contracts.md`. They are implemented dry-run-first: dry mode is pure stdlib
and writes the contracted artifacts as unmeasured plumbing; `--real` paths gate on `--allow-gpu`
plus the optional ML stack (`pip install -e '.[train,eval,merge,data]'`) and fail closed until the
concrete trainer/eval/merge is implemented (never fabricating metrics).

Shared logic lives in `src/boldt_posttrain/` (config resolution, provenance/run cards, the
protected scorer, the training-lever skeleton, the frontier view). Drive the loop through the
`/pt-*` Claude commands (each calls `python scripts/pt_*.py` directly — there is no Makefile).
Validate with `python -m py_compile scripts/pt_*.py scripts/check_posttrain_integrity.py` and
`python -m unittest discover -s tests`.
