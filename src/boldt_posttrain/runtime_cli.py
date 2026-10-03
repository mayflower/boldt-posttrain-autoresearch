"""CLI adapters for the canonical artifact lifecycle; no recipe artifacts."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

from .policy import load_policy
from .artifacts import ArtifactError
from .resolver import resolve_model
from .secure_compat.config import ConfigError, load_experiment
from .secure_compat import data_pipeline, evaluation, merge, scoring


def _emit(operation, *, failure_code=4):
    try:
        result = operation()
    except Exception as exc:
        code = (
            2
            if isinstance(exc, ConfigError)
            else 5
            if isinstance(exc, ArtifactError)
            else failure_code
        )
        print(
            json.dumps(
                {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "exit_code": code}
            )
        )
        return code
    if isinstance(result, tuple):
        result, code = result
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return code
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result.get("status") == "failed":
        return 4
    return 1 if result.get("status") == "rejected" else 0


def _config(args, root):
    path = root / args.config
    load_experiment(path)
    return path


def _budget(args):
    if not math.isfinite(args.budget_minutes) or args.budget_minutes <= 0:
        raise ConfigError("budget-minutes must be positive and finite")
    return args.budget_minutes


def data(args, *, root, outputs, plan):
    def operation():
        config_path = _config(args, root)
        if args.dry_run:
            return plan(f"data-{args.action}", config_path)
        return data_pipeline.run_cli(
            SimpleNamespace(data_command=args.action, config=str(config_path)),
            outputs_root=outputs,
        )

    return _emit(operation, failure_code=2 if args.dry_run else 4)


def evaluate(args, *, root, outputs, plan, baseline=False):
    def operation():
        config_path = _config(args, root)
        budget = _budget(args)
        if args.device != "cuda:0":
            raise ValueError(
                "canonical evaluation requires cuda:0; select the GPU with CUDA_VISIBLE_DEVICES"
            )
        if args.real and not args.allow_gpu:
            raise ValueError("real evaluation requires --allow-gpu")
        policy = load_policy()
        resolved = resolve_model(
            policy=policy,
            candidate=None if baseline else args.candidate,
            model=args.model,
            outputs_root=outputs,
            external_roots=tuple(Path(item) for item in getattr(args, "external_root", [])),
        )
        if args.dry_run:
            return plan("baseline" if baseline else "evaluation", config_path)
        import torch

        if not torch.cuda.is_available():
            raise evaluation.EvaluationError("real evaluation requires CUDA")
        return evaluation._publish_evaluation(
            resolved=resolved,
            policy=policy,
            config_path=config_path,
            output_root=outputs / ("baseline" if baseline else "evals"),
            baseline=baseline,
            replace_baseline=getattr(args, "replace_baseline", False),
            device=args.device,
            repository_root=root,
            deadline=time.monotonic() + budget * 60,
        )

    return _emit(operation, failure_code=3)


def score(args, *, root, outputs):
    return _emit(
        lambda: scoring.create_score(
            args.candidate, policy=load_policy(), outputs_root=outputs, repository_root=root
        ),
        failure_code=5,
    )


def merge_candidates(args, *, root, outputs, plan):
    def operation():
        config_path = _config(args, root)
        config = load_experiment(config_path).document
        budget = _budget(args)
        settings = config["merge"]
        if args.dry_run:
            return plan("merge", config_path)
        if not (args.allow_checkpoints and args.allow_gpu):
            raise ValueError("real merge requires --allow-gpu --allow-checkpoints")
        return merge.run_search(
            candidate_ids=settings["inputs"],
            methods=settings["methods"],
            parameters=settings["parameters"],
            dtype=settings["dtype"],
            policy=load_policy(),
            allow_checkpoints=args.allow_checkpoints,
            allow_gpu=args.allow_gpu,
            budget_minutes=budget,
            outputs_root=outputs,
            repository_root=root,
        )

    return _emit(operation, failure_code=2 if not args.allow_checkpoints else 4)


def validate_suite():
    def operation():
        policy = load_policy()
        cases = evaluation.load_suite(evaluation.ROOT / policy.document["evaluation"]["suite_path"])
        return {"status": "ok", "cases": len(cases), "suite_hash": evaluation.suite_hash()}

    return _emit(operation, failure_code=5)


def status(*, root, outputs):
    from .loop import verified_status

    return _emit(
        lambda: verified_status(outputs_root=outputs, repository_root=root), failure_code=5
    )
