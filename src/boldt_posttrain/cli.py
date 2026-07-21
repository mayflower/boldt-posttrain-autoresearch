"""Single JSON CLI for all post-training operations."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from . import config as config_module
from .artifacts import ArtifactError, atomic_write_json, new_run_id
from .frontier import FrontierError
from .policy import PolicyError, load_policy
from .provenance import ProvenanceError
from .resolver import OUTPUTS, ResolutionError, resolve_model
from .scoring import ScoringError

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("boldt_posttrain")


class CliError(RuntimeError):
    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


class JsonParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliError(message, 2)


def _mode(parser: argparse.ArgumentParser, *, gpu: bool = False, checkpoints: bool = False) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--real", action="store_true")
    if gpu:
        parser.add_argument("--allow-gpu", action="store_true")
    if checkpoints:
        parser.add_argument("--allow-checkpoints", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = JsonParser(prog="pt", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True, parser_class=JsonParser)

    policy = commands.add_parser("policy")
    policy_commands = policy.add_subparsers(
        dest="policy_command", required=True, parser_class=JsonParser
    )
    policy_validate = policy_commands.add_parser("validate")
    policy_validate.add_argument("--policy", default="configs/posttrain/policy.json")

    integrity = commands.add_parser("integrity")
    integrity_commands = integrity.add_subparsers(
        dest="integrity_command", required=True, parser_class=JsonParser
    )
    integrity_check = integrity_commands.add_parser("check")
    integrity_check.add_argument("--base-ref")
    integrity_check.add_argument("--policy", default="configs/posttrain/policy.json")

    model = commands.add_parser("model")
    model_commands = model.add_subparsers(
        dest="model_command", required=True, parser_class=JsonParser
    )
    model_resolve = model_commands.add_parser("resolve")
    reference = model_resolve.add_mutually_exclusive_group(required=True)
    reference.add_argument("--candidate")
    reference.add_argument("--model")
    model_resolve.add_argument("--external-root", action="append", default=[])
    model_resolve.add_argument("--outputs-root", default=str(OUTPUTS))
    model_resolve.add_argument("--policy", default="configs/posttrain/policy.json")

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True, parser_class=JsonParser)
    for name in ("discover", "prepare"):
        item = data_commands.add_parser(name)
        _mode(item)
        item.add_argument("--config", default="configs/posttrain/current.json")

    baseline = commands.add_parser("baseline")
    baseline_commands = baseline.add_subparsers(
        dest="baseline_command", required=True, parser_class=JsonParser
    )
    baseline_run = baseline_commands.add_parser("run")
    _mode(baseline_run, gpu=True)
    baseline_run.add_argument("--config", default="configs/posttrain/current.json")
    baseline_run.add_argument("--replace-baseline", action="store_true")

    train = commands.add_parser("train")
    train_commands = train.add_subparsers(
        dest="train_command", required=True, parser_class=JsonParser
    )
    for name in ("sft", "cpt", "preference"):
        item = train_commands.add_parser(name)
        _mode(item, gpu=True, checkpoints=True)
        item.add_argument("--config", default="configs/posttrain/current.json")
        item.add_argument("--budget-minutes", type=float)
        if name == "preference":
            item.add_argument("--method", choices=("dpo", "kto", "orpo"))

    distill = commands.add_parser("distill")
    _mode(distill, gpu=True, checkpoints=True)
    distill.add_argument("--config", default="configs/posttrain/current.json")
    distill.add_argument("--teacher", required=True)
    distill.add_argument("--teacher-license")
    distill.add_argument("--budget-minutes", type=float)

    evaluation = commands.add_parser("eval")
    evaluation_commands = evaluation.add_subparsers(
        dest="eval_command", required=True, parser_class=JsonParser
    )
    evaluation_run = evaluation_commands.add_parser("run")
    _mode(evaluation_run, gpu=True)
    evaluation_reference = evaluation_run.add_mutually_exclusive_group(required=True)
    evaluation_reference.add_argument("--candidate")
    evaluation_reference.add_argument("--model")
    evaluation_run.add_argument("--external-root", action="append", default=[])
    evaluation_run.add_argument("--config", default="configs/posttrain/current.json")
    evaluation_commands.add_parser("validate-suite")
    evaluation_commands.add_parser("catalog")

    score = commands.add_parser("score")
    score.add_argument("--candidate", required=True)
    score.add_argument("--baseline")

    merge = commands.add_parser("merge")
    merge_commands = merge.add_subparsers(
        dest="merge_command", required=True, parser_class=JsonParser
    )
    merge_search = merge_commands.add_parser("search")
    _mode(merge_search, gpu=True, checkpoints=True)
    merge_search.add_argument("--config", default="configs/posttrain/current.json")
    merge_search.add_argument("--budget-minutes", type=float)

    promote = commands.add_parser("promote")
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--base-ref", required=True)

    commands.add_parser("status")
    commands.add_parser("report")
    loop = commands.add_parser("loop")
    loop_commands = loop.add_subparsers(dest="loop_command", required=True, parser_class=JsonParser)
    loop_run = loop_commands.add_parser("run")
    _mode(loop_run, gpu=True, checkpoints=True)
    loop_run.add_argument("--config", default="configs/posttrain/current.json")
    loop_run.add_argument("--base-ref", required=True)
    loop_run.add_argument("--budget-minutes", type=float, required=True)
    loop_run.add_argument("--promote", action="store_true")

    doctor = commands.add_parser("doctor")
    doctor.add_argument("--mode", choices=("train", "eval", "merge", "all"), default="all")
    doctor.add_argument("--real", action="store_true")
    doctor.add_argument("--allow-gpu", action="store_true")
    return parser


def _integrity(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    path = ROOT / "scripts" / "check_posttrain_integrity.py"
    spec = importlib.util.spec_from_file_location("check_posttrain_integrity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    result = module.check(base_ref=args.base_ref, policy_path=ROOT / args.policy)
    return result, 0 if result["status"] == "pass" else 5


def _plan(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    plan_id = new_run_id("plan")
    operation = "-".join(
        str(getattr(args, key))
        for key in ("command", f"{args.command}_command")
        if hasattr(args, key)
    )
    document = {
        "schema_version": 1,
        "plan_id": plan_id,
        "operation": operation,
        "config": getattr(args, "config", None),
    }
    path = OUTPUTS / "plans" / plan_id / "plan.json"
    atomic_write_json(path, document)
    try:
        published_path = str(path.relative_to(ROOT))
    except ValueError:
        published_path = str(path)
    return {"status": "succeeded", "mode": "dry_run", "plan": published_path}, 0


def _check_real_gates(args: argparse.Namespace) -> None:
    if not getattr(args, "real", False):
        return
    if hasattr(args, "allow_gpu") and not args.allow_gpu:
        raise CliError("--real requires --allow-gpu", 2)
    if hasattr(args, "allow_checkpoints") and not args.allow_checkpoints:
        raise CliError("--real requires --allow-checkpoints", 2)


def _domain(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if getattr(args, "dry_run", False):
        return _plan(args)
    _check_real_gates(args)
    module_name = {
        "data": "data_pipeline",
        "baseline": "evaluation",
        "train": "training",
        "distill": "distillation",
        "eval": "evaluation",
        "score": "scoring",
        "merge": "merge",
        "promote": "frontier",
        "status": "loop",
        "report": "loop",
        "loop": "loop",
        "doctor": "training",
    }[args.command]
    if args.command == "train" and args.train_command == "preference":
        module_name = "preference"
    module = importlib.import_module(f"boldt_posttrain.{module_name}")
    handler = getattr(module, "run_cli", None)
    if handler is None:
        raise CliError(f"{module_name} cannot serve this command", 3)
    return handler(args)


def dispatch(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.command == "policy":
        policy = load_policy(ROOT / args.policy)
        return {
            "status": "succeeded",
            "schema_version": policy.document["schema_version"],
            "seed_revision": policy.seed_model["revision"],
        }, 0
    if args.command == "integrity":
        return _integrity(args)
    if args.command == "model":
        policy = load_policy(ROOT / args.policy)
        resolved = resolve_model(
            policy=policy,
            candidate=args.candidate,
            model=args.model,
            outputs_root=Path(args.outputs_root),
            external_roots=tuple(Path(item) for item in args.external_root),
        )
        return {"status": "succeeded", "model": resolved.to_dict()}, 0
    if args.command == "eval" and args.eval_command == "run":
        policy = load_policy()
        resolve_model(
            policy=policy,
            candidate=args.candidate,
            model=args.model,
            external_roots=tuple(Path(item) for item in args.external_root),
        )
    if hasattr(args, "config"):
        config_module.load_experiment(ROOT / args.config)
    return _domain(args)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
    try:
        args = build_parser().parse_args(list(argv) if argv is not None else None)
        result, exit_code = dispatch(args)
    except CliError as exc:
        LOGGER.error("%s", exc)
        result, exit_code = (
            {"status": "failed", "error": str(exc), "exit_code": exc.exit_code},
            exc.exit_code,
        )
    except (PolicyError, config_module.ConfigError) as exc:
        LOGGER.error("%s", exc)
        result, exit_code = {"status": "failed", "error": str(exc), "exit_code": 2}, 2
    except ResolutionError as exc:
        LOGGER.error("%s", exc)
        result, exit_code = {"status": "failed", "error": str(exc), "exit_code": 3}, 3
    except (ArtifactError, FrontierError, ProvenanceError, ScoringError) as exc:
        LOGGER.error("%s", exc)
        result, exit_code = {"status": "failed", "error": str(exc), "exit_code": 5}, 5
    except Exception as exc:
        LOGGER.exception("operation failed")
        result, exit_code = {"status": "failed", "error": str(exc), "exit_code": 4}, 4
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return exit_code


def main_status(argv: Sequence[str] | None = None) -> int:
    return main(["status", *(argv or [])])


def main_report(argv: Sequence[str] | None = None) -> int:
    return main(["report", *(argv or [])])


def main_integrity(argv: Sequence[str] | None = None) -> int:
    return main(["integrity", "check", *(argv or [])])


if __name__ == "__main__":
    raise SystemExit(main())
