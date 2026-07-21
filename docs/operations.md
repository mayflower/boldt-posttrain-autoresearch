# Operations

## Preflight

```bash
conda activate boldtembed
export VIRTUAL_ENV="$CONDA_PREFIX"
export CUDA_DEVICE_ORDER=FASTEST_FIRST
export CUDA_VISIBLE_DEVICES=0
scripts/sync_conda_env.sh
python -m boldt_posttrain.cli doctor --mode all --real --allow-gpu
python -m boldt_posttrain.cli policy validate
python -m boldt_posttrain.cli eval validate-suite
```

The real doctor must report CUDA, GPU name/VRAM/compute capability, BF16, pinned package versions,
seed accessibility/fingerprints, disk space, lm-eval task validity, and Mergekit availability.

## Dry Plans

```bash
python -m boldt_posttrain.cli train sft --dry-run --config configs/posttrain/current.json
python -m boldt_posttrain.cli eval run --dry-run --candidate train-sft-20260721T120000.000000Z-0123456789abcdef
```

Compare hashes of all real namespaces before and after if investigating isolation. Plans have no
quality meaning and cannot enter the registry.

## Troubleshooting

- Exit 2: fix the explicit mode, flags, or strict experiment schema.
- Exit 3: install the locked extras or create the required baseline/data/candidate artifact.
- Exit 4: inspect stderr and the failed stage; do not retry with changed semantics.
- Exit 5: stop immediately and inspect Git classifications, ArtifactRefs, and the event chain.
- CUDA OOM: reduce the planned experiment manually in a new reviewed experiment file; the running
  command never changes its batch or method.
- A rejected score is normal exit 1 and remains auditable.

## GPU Acceptance

On the target NVIDIA host run data discovery/preparation, baseline, one 90-minute QLoRA SFT,
candidate evaluation, score/promotion, and one merge search using the commands in `README.md`.
Capture stdout JSON, stderr, exit code, run card, policy/config/data/model/suite/raw/score hashes,
event sequence, wall time, peak VRAM, and tokens/s. Without that evidence the result is only
CPU-integrated, not GPU-validated.
