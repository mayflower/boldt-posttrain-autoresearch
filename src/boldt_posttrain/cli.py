"""Unified ``pt`` command for the post-training research loop."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from . import config as cfgmod
from .artifacts import atomic_write_json, new_run_id
from .data_pipeline import verify_data_manifest
from .frontier import (
    current_frontier_hash,
    promote_candidate,
)
from .policy import load_policy
from .resolver import OUTPUTS, resolve_model

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = ROOT


class CliParseError(ValueError):
    """Raised when an explicitly gated command is malformed."""


class GatedParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliParseError(message)


def _explicit_mode(parser: argparse.ArgumentParser, *, gpu: bool = False) -> None:
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    if gpu:
        parser.add_argument("--allow-gpu", action="store_true")
        parser.add_argument("--allow-checkpoints", action="store_true")


def _load(script_stem: str):
    path = SCRIPT_ROOT / "scripts" / f"{script_stem}.py"
    spec = importlib.util.spec_from_file_location(script_stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load command script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script(stem: str, argv: Sequence[str]) -> int:
    return int(_load(stem).main(list(argv)))


def _plan(operation: str, config: str | Path | None = None) -> dict[str, Any]:
    plan_id = new_run_id("plan")
    path = OUTPUTS / "plans" / plan_id / "plan.json"
    document = {
        "schema_version": 1,
        "plan_id": plan_id,
        "operation": operation,
        "config": str(config) if config is not None else None,
    }
    atomic_write_json(path, document)
    return {"status": "succeeded", "mode": "dry_run", "plan": str(path)}


def _eval_run_command(args):
    from .runtime_cli import evaluate

    return evaluate(args, root=ROOT, outputs=OUTPUTS, plan=_plan)


def _baseline_run_command(args):
    from .runtime_cli import evaluate

    return evaluate(args, root=ROOT, outputs=OUTPUTS, plan=_plan, baseline=True)


def _score_command(args):
    from .runtime_cli import score

    return score(args, root=ROOT, outputs=OUTPUTS)


def _data_prepare_command(args):
    from .runtime_cli import data

    return data(args, root=ROOT, outputs=OUTPUTS, plan=_plan)


def _data_discover_command(args):
    from .runtime_cli import data

    return data(args, root=ROOT, outputs=OUTPUTS, plan=_plan)


def _train_command(args: argparse.Namespace) -> int:
    # Manual levers run the loop's producer, so their candidates resolve like loop ones.
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if args.dry_run:
        try:
            policy = load_policy()
            config = cfgmod.load_experiment(config_path)
            config.document["experiment"]["lever"] = args.action
            from .online import ONLINE_LEVERS, online_method, online_rows, validate_online_policy

            if args.action in ONLINE_LEVERS:
                validate_online_policy(config.document, policy)
            if args.action in ONLINE_LEVERS and online_method(args.action) == "opd":
                teacher = (
                    getattr(args, "teacher", None) or config.document["distillation"]["teacher"]
                )
                seed = policy.seed_model
                if teacher in {seed["repo_id"], f"{seed['repo_id']}@{seed['revision']}"}:
                    raise ValueError(
                        "OPD requires a distinct teacher; the student seed supplies no "
                        "additional teaching signal (an external teacher with a different "
                        "tokenizer needs the seqkd lever instead)"
                    )
            manifest = verify_data_manifest(OUTPUTS / "data", policy, repository_root=ROOT)
            if args.action in ONLINE_LEVERS:
                online_rows(manifest, config.document, root=ROOT)
            if args.action == "seqkd":
                from .seqkd import validate_seqkd_policy, verify_generation

                verify_generation(
                    validate_seqkd_policy(config.document, policy),
                    policy,
                    outputs_root=OUTPUTS,
                    repository_root=ROOT,
                )
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"status": "failed", "mode": "dry_run", "error": str(exc)}))
            return 2
        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "dry_run",
                    "lever": args.action,
                    "config": str(config_path),
                    "message": "preflight ok; pass --real --allow-gpu --allow-checkpoints to train",
                }
            )
        )
        return 0
    from .loop import train_one_lever

    result, code = train_one_lever(
        lever=args.action,
        config_path=config_path,
        budget_minutes=args.budget_minutes,
        allow_gpu=args.allow_gpu,
        allow_checkpoints=args.allow_checkpoints,
        specialist=getattr(args, "specialist", None),
        preference_method=getattr(args, "method", None),
        teacher=getattr(args, "teacher", None),
        teacher_license=getattr(args, "teacher_license", None),
    )
    print(json.dumps(result, ensure_ascii=False))
    return code


def _merge_command(args):
    from .runtime_cli import merge_candidates

    return merge_candidates(args, root=ROOT, outputs=OUTPUTS, plan=_plan)


def _promote_command(args: argparse.Namespace) -> int:
    try:
        policy = load_policy()
        result = promote_candidate(
            args.candidate,
            base_ref=args.base_ref,
            expected_current_sha256=current_frontier_hash(OUTPUTS / "frontier"),
            policy=policy,
            outputs_root=OUTPUTS,
            repository_root=ROOT,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 5
    print(json.dumps(result, sort_keys=True))
    return 0


def _model_command(args: argparse.Namespace) -> int:
    try:
        policy = load_policy(ROOT / args.policy)
        resolved = resolve_model(
            policy=policy,
            candidate=args.candidate,
            model=args.model,
            outputs_root=Path(args.outputs_root),
            external_roots=tuple(Path(item) for item in args.external_root),
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 3
    print(json.dumps({"status": "succeeded", "model": resolved.to_dict()}, sort_keys=True))
    return 0


def _eval_catalog_command(_args: argparse.Namespace) -> int:
    from .evaluation import lm_eval_catalog

    try:
        result = lm_eval_catalog(load_policy())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 5
    print(json.dumps({"status": "succeeded", **result}, sort_keys=True))
    return 0


def _doctor_command(args: argparse.Namespace) -> int:
    from .training import doctor

    if args.real and not args.allow_gpu:
        print(json.dumps({"status": "failed", "error": "--real requires --allow-gpu"}))
        return 2
    try:
        result = doctor(mode=args.mode, real=args.real)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 4
    print(json.dumps(result, sort_keys=True))
    return 0


def _seqkd_generate_command(args: argparse.Namespace) -> int:
    """Stage 1 of sequence-level distillation: teacher answers -> verified SFT manifest."""
    from .seqkd import generate, select_prompts, validate_seqkd_policy

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        policy = load_policy()
        config = cfgmod.load_experiment(config_path)
        settings = validate_seqkd_policy(config.document, policy)
        manifest = verify_data_manifest(OUTPUTS / "data", policy, repository_root=ROOT)
        if args.dry_run:
            from .seqkd import _shard_rows

            prompts = select_prompts(
                _shard_rows(manifest, "sft_shard", ROOT),
                maximum=settings["max_prompts"],
                seed=settings["seed"],
            )
            if not prompts:
                raise ValueError("the verified SFT manifest contains no answerable prompts")
            result = {
                **_plan("seqkd-generate", config_path),
                "teacher": settings["teacher"],
                "prompts": len(prompts),
                "prompt_source": manifest["run_id"],
            }
            print(json.dumps(result, sort_keys=True))
            return 0
        if not args.allow_gpu:
            raise ValueError("seqkd generation requires --real --allow-gpu")
        import torch

        if not torch.cuda.is_available():
            raise ValueError("seqkd generation requires CUDA and never falls back to CPU")
        result = generate(policy, config_path, outputs_root=OUTPUTS, repository_root=ROOT)
    except Exception as exc:  # noqa: BLE001 -- surface any failure as a nonzero exit
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 4
    print(json.dumps(result, sort_keys=True))
    return 0


def _loop_command(args: argparse.Namespace) -> int:
    if args.dry_run:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        try:
            cfgmod.load_experiment(config_path)
        except Exception as exc:
            print(json.dumps({"status": "failed", "error": str(exc), "exit_code": 2}))
            return 2
        print(json.dumps(_plan("loop-run", config_path)))
        return 0
    if not (args.allow_gpu and args.allow_checkpoints):
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "loop run requires --real --allow-gpu --allow-checkpoints",
                }
            )
        )
        return 2
    from .loop import run_experiment

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        result, exit_code = run_experiment(
            config_path=config_path,
            base_ref=args.base_ref,
            budget_minutes=args.budget_minutes,
            promote=args.promote,
            allow_checkpoints=args.allow_checkpoints,
            allow_gpu=args.allow_gpu,
            outputs_root=OUTPUTS,
            repository_root=ROOT,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 4
    print(json.dumps(result, sort_keys=True))
    return exit_code


def _integrity_command(args):
    argv = ["--base-ref", args.base_ref, "--format", "json"]
    if args.strict:
        argv.append("--strict")
    return _script("check_posttrain_integrity", argv)


def _report_command(args: argparse.Namespace) -> int:
    """Render the human-readable report of one exact loop round from its artifacts."""
    from .artifacts import RUN_ID_RE, EventLog, sha256_file
    from .report import render_loop_report

    if not RUN_ID_RE.fullmatch(args.loop):
        print(json.dumps({"status": "failed", "error": "--loop must be an exact loop run ID"}))
        return 2
    verdict_path = OUTPUTS / "loops" / args.loop / "verdict.json"
    if not verdict_path.is_file():
        print(json.dumps({"status": "failed", "error": f"no verdict for loop {args.loop}"}))
        return 3
    events = EventLog(OUTPUTS)
    anchored = any(
        event.get("run_id") == args.loop
        and event.get("payload", {}).get("verdict_sha256") == sha256_file(verdict_path)
        for event in (json.loads(line) for line in events.log_path.read_text().splitlines())
    )
    report = render_loop_report(
        json.loads(verdict_path.read_text()), policy=load_policy(), outputs_root=OUTPUTS
    )
    if not anchored:
        report = "> WARNING: this verdict has no matching event-chain anchor.\n\n" + report
    print(report, end="")
    return 0


def _status_command(_args):
    from .runtime_cli import status

    return status(root=ROOT, outputs=OUTPUTS)


def _policy_validate(args):
    try:
        policy = load_policy(Path(args.path))
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 5
    print(json.dumps({"status": "ok", "policy": str(policy.path)}))
    return 0


def _eval_validate(_args):
    from .runtime_cli import validate_suite

    return validate_suite()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pt")
    commands = parser.add_subparsers(dest="command", required=True)

    policy = commands.add_parser("policy")
    policy_sub = policy.add_subparsers(dest="action", required=True)
    policy_validate = policy_sub.add_parser("validate")
    policy_validate.add_argument("--path", default=str(ROOT / "configs/posttrain/policy.json"))
    policy_validate.set_defaults(handler=_policy_validate)

    model = commands.add_parser("model")
    model_sub = model.add_subparsers(dest="action", required=True)
    model_resolve = model_sub.add_parser("resolve")
    model_reference = model_resolve.add_mutually_exclusive_group(required=True)
    model_reference.add_argument("--candidate")
    model_reference.add_argument("--model")
    model_resolve.add_argument("--external-root", action="append", default=[])
    model_resolve.add_argument("--outputs-root", default=str(OUTPUTS))
    model_resolve.add_argument("--policy", default="configs/posttrain/policy.json")
    model_resolve.set_defaults(handler=_model_command)

    evaluation = commands.add_parser("eval")
    eval_sub = evaluation.add_subparsers(dest="action", required=True, parser_class=GatedParser)
    eval_run = eval_sub.add_parser("run")
    _explicit_mode(eval_run)
    eval_run.add_argument("--allow-gpu", action="store_true")
    eval_run.add_argument("--config", default="configs/posttrain/secure-current.json")
    eval_reference = eval_run.add_mutually_exclusive_group(required=True)
    eval_reference.add_argument("--model")
    eval_reference.add_argument("--candidate")
    eval_run.add_argument("--external-root", action="append", default=[])
    eval_run.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    eval_run.add_argument("--budget-minutes", type=float, default=90.0)
    eval_run.set_defaults(handler=_eval_run_command)
    validate = eval_sub.add_parser("validate-suite")
    validate.set_defaults(handler=_eval_validate)
    catalog = eval_sub.add_parser("catalog")
    catalog.set_defaults(handler=_eval_catalog_command)

    baseline = commands.add_parser("baseline")
    baseline_sub = baseline.add_subparsers(dest="action", required=True, parser_class=GatedParser)
    baseline_run = baseline_sub.add_parser("run")
    _explicit_mode(baseline_run)
    baseline_run.add_argument("--allow-gpu", action="store_true")
    baseline_run.add_argument("--config", default="configs/posttrain/secure-current.json")
    baseline_run.add_argument("--model")
    baseline_run.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    baseline_run.add_argument("--budget-minutes", type=float, default=90.0)
    baseline_run.add_argument("--replace-baseline", action="store_true")
    baseline_run.set_defaults(handler=_baseline_run_command)

    score = commands.add_parser("score")
    score.add_argument("--candidate", required=True, help="exact canonical evaluation run ID")
    score.set_defaults(handler=_score_command)

    train = commands.add_parser("train")
    train_sub = train.add_subparsers(dest="action", required=True, parser_class=GatedParser)
    for name in (
        "sft",
        "cpt",
        "preference",
        "grpo",
        "rlvr",
        "opd",
        "sdpo",
        "sdft",
        "distill",
        "seqkd",
    ):
        train_lever = train_sub.add_parser(name)
        _explicit_mode(train_lever, gpu=True)
        train_lever.add_argument(
            "--config", default=str(ROOT / "configs/posttrain/secure-current.json")
        )
        train_lever.add_argument("--budget-minutes", type=float, default=90.0)
        train_lever.add_argument("--specialist")
        if name == "preference":
            train_lever.add_argument("--method", choices=("dpo", "kto", "orpo"))
        if name in {"opd", "distill"}:
            train_lever.add_argument("--teacher")
            train_lever.add_argument("--teacher-license")
        train_lever.set_defaults(handler=_train_command)

    data = commands.add_parser("data")
    data_sub = data.add_subparsers(dest="action", required=True, parser_class=GatedParser)
    for action, handler in (
        ("discover", _data_discover_command),
        ("prepare", _data_prepare_command),
    ):
        command = data_sub.add_parser(action)
        _explicit_mode(command)
        command.add_argument("--config", default="configs/posttrain/secure-current.json")
        command.set_defaults(handler=handler)
    merge = commands.add_parser("merge")
    merge_sub = merge.add_subparsers(dest="action", required=True, parser_class=GatedParser)
    merge_search = merge_sub.add_parser("search")
    _explicit_mode(merge_search, gpu=True)
    merge_search.add_argument("--config", default="configs/posttrain/secure-current.json")
    merge_search.add_argument("--budget-minutes", type=float, default=90.0)
    merge_search.set_defaults(handler=_merge_command)

    seqkd = commands.add_parser("seqkd")
    seqkd_sub = seqkd.add_subparsers(dest="action", required=True, parser_class=GatedParser)
    seqkd_generate = seqkd_sub.add_parser("generate")
    _explicit_mode(seqkd_generate)
    seqkd_generate.add_argument("--allow-gpu", action="store_true")
    seqkd_generate.add_argument("--config", default="configs/posttrain/secure-current.json")
    seqkd_generate.set_defaults(handler=_seqkd_generate_command)

    status = commands.add_parser("status")
    status.set_defaults(handler=_status_command)
    report = commands.add_parser("report")
    report.add_argument("--loop", required=True, help="exact loop run ID")
    report.set_defaults(handler=_report_command)
    integrity = commands.add_parser("integrity")
    integrity_sub = integrity.add_subparsers(dest="action", required=True)
    integrity_check = integrity_sub.add_parser("check")
    integrity_check.add_argument("--base-ref", required=True)
    integrity_check.add_argument("--strict", action="store_true")
    integrity_check.set_defaults(handler=_integrity_command)
    loop = commands.add_parser("loop")
    loop_sub = loop.add_subparsers(dest="action", required=True, parser_class=GatedParser)
    loop_run = loop_sub.add_parser("run")
    _explicit_mode(loop_run, gpu=True)
    loop_run.add_argument("--config", default="configs/posttrain/secure-current.json")
    loop_run.add_argument("--base-ref", required=True)
    loop_run.add_argument("--budget-minutes", type=float, required=True)
    loop_run.add_argument("--promote", action="store_true")
    loop_run.set_defaults(handler=_loop_command)
    promote = commands.add_parser("promote")
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--base-ref", required=True)
    promote.set_defaults(handler=_promote_command)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--mode", choices=("train", "eval", "merge", "all"), default="all")
    doctor.add_argument("--real", action="store_true")
    doctor.add_argument("--allow-gpu", action="store_true")
    doctor.set_defaults(handler=_doctor_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
