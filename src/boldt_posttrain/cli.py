"""Unified ``pt`` command for the post-training research loop."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from . import config as cfgmod
from .bootstrap import run_bootstrap
from .data_pipeline import (
    FastTextLanguageIdentifier,
    IntegrityError,
    build_decontamination_corpus,
    canonical_json,
    discover_sources,
    iter_manifest_rows,
    file_sha256,
    lm_eval_decontamination_records,
    language_identifier_from_config,
    load_manifest_rows,
    materialize_streaming,
    sha256_bytes,
    verify_hashed_artifact,
    verify_trainable_manifest,
)
from .evaluation import (
    finalize_summary,
    load_suite,
    register_promotion_suite,
    resolve_suite,
    run_real_evaluation,
)
from .failure_mining import build_synthesis_tasks, mine_failures, synthesize_verified
from .frontier import aggregate, current_frontier, update_specialist_frontiers
from .policy import load_policy
from .provenance import append_event, new_run_card, stamp, write_run_card
from .rlvr import iter_rlvr_rows, train_rlvr
from .resolver import OUTPUTS, resolve_model
from .scheduler import run_mix_probes, run_successive_halving
from .training import _train_real, compare_sft_rlvr, validate_device
from .verified_rl import train_verified_grpo

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


def _forward(
    args: argparse.Namespace, stem: str, fields: Sequence[str], flags: Sequence[str]
) -> int:
    argv = []
    for field in fields:
        value = getattr(args, field, None)
        if value is not None:
            if field == "config" and not Path(str(value)).is_absolute():
                value = ROOT / str(value)
            argv.extend(["--" + field.replace("_", "-"), str(value)])
    for flag in flags:
        if getattr(args, flag, False):
            argv.append("--" + flag.replace("_", "-"))
    return _script(stem, argv)


def _eval_run_command(args: argparse.Namespace) -> int:
    if args.real and args.candidate:
        try:
            resolve_model(
                policy=load_policy(),
                candidate=args.candidate,
                outputs_root=OUTPUTS,
            )
        except Exception as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}))
            return 3
    return _forward(
        args,
        "pt_eval",
        (
            "config",
            "model",
            "candidate",
            "label",
            "out",
            "device",
            "profile",
            "suite",
            "budget_minutes",
        ),
        ("real", "dry_run", "allow_gpu"),
    )


def _baseline_run_command(args: argparse.Namespace) -> int:
    return _forward(
        args,
        "pt_baseline",
        (
            "config",
            "out",
            "model",
            "label",
            "device",
            "profile",
            "suite",
            "budget_minutes",
        ),
        ("real", "dry_run", "allow_gpu"),
    )


def _score_command(args: argparse.Namespace) -> int:
    if args.out is None:
        label = args.candidate or (Path(args.run).stem if args.run else "score")
        args.out = str(ROOT / "outputs/posttrain/scores" / f"{label}.json")
    return _forward(
        args,
        "pt_score",
        (
            "config",
            "run",
            "candidate",
            "baseline",
            "out",
            "profile",
            "format",
        ),
        (),
    )


def _data_prepare_command(args: argparse.Namespace) -> int:
    return _forward(
        args,
        "pt_prepare_openeurollm_de",
        (
            "config",
            "discovery",
            "selection",
            "discovery_run_id",
            "selection_run_id",
            "out",
            "format",
        ),
        ("real", "dry_run"),
    )


def _data_discover_command(args: argparse.Namespace) -> int:
    if args.dry_run:
        try:
            config_path = Path(args.config)
            if not config_path.is_absolute():
                config_path = ROOT / config_path
            cfg = cfgmod.resolve_config(config_path)
            errors = cfgmod.validate_config_dict(cfg)
            if errors:
                raise ValueError("; ".join(errors))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(json.dumps({"status": "failed", "error": str(exc), "exit_code": 2}))
            return 2
        plan_id = f"plan-data-discover-{stamp()}"
        path = OUTPUTS / "plans" / plan_id / "plan.json"
        path.parent.mkdir(parents=True, exist_ok=False)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "plan_id": plan_id,
                    "operation": "data-discover",
                    "config": str(config_path),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"status": "succeeded", "mode": "dry_run", "plan": str(path)}))
        return 0
    return _forward(
        args,
        "pt_discover_openeurollm_de",
        ("config", "out", "format"),
        ("real", "dry_run"),
    )


def _train_command(args: argparse.Namespace) -> int:
    stem = {
        "sft": "pt_train_specialist",
        "cpt": "pt_train_cpt",
        "preference": "pt_train_preference",
    }[args.action]
    fields = ("config", "specialist", "out", "data", "budget_minutes", "device", "mix_plan")
    if args.action == "preference":
        fields = (*fields, "method")
    return _forward(
        args,
        stem,
        fields,
        ("real", "dry_run", "allow_gpu", "allow_checkpoints"),
    )


def _merge_command(args: argparse.Namespace) -> int:
    return _forward(
        args,
        "pt_merge_search",
        ("config", "runs", "frontier", "out", "format", "merge_device", "device", "budget_minutes"),
        ("real", "dry_run", "allow_gpu"),
    )


def _promote_command(args: argparse.Namespace) -> int:
    return _forward(
        args,
        "pt_promote",
        ("candidate", "config", "baseline", "base_ref", "out", "format"),
        (),
    )


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
    from .secure_compat.training import doctor

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


def _distill_command(args: argparse.Namespace) -> int:
    from .secure_compat.distillation import run_cli

    if not (args.real and args.allow_gpu and args.allow_checkpoints):
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "distillation requires --real --allow-gpu --allow-checkpoints",
                }
            )
        )
        return 2
    try:
        result, exit_code = run_cli(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 4
    print(json.dumps(result, sort_keys=True))
    return exit_code


def _loop_command(args: argparse.Namespace) -> int:
    forwarded = list(args.remainder)
    if forwarded[:1] == ["run"]:
        forwarded.pop(0)
    forwarded = [item for item in forwarded if item != "--allow-checkpoints"]
    return _script("pt_loop", forwarded)


def main_status(argv: Optional[Sequence[str]] = None) -> int:
    return _script("pt_status", list(argv or []))


def main_report(argv: Optional[Sequence[str]] = None) -> int:
    return _script("pt_report", list(argv or []))


def main_integrity(argv: Optional[Sequence[str]] = None) -> int:
    return _script("check_posttrain_integrity", list(argv or []))


def _hashed(body: Dict[str, Any]) -> Dict[str, Any]:
    return {**body, "artifact_hash": sha256_bytes(canonical_json(body))}


def _bootstrap(args: argparse.Namespace) -> int:
    if not args.real or not args.allow_gpu:
        print("bootstrap run explicitly requires --real --allow-gpu", file=sys.stderr)
        return 2
    cfg = cfgmod.resolve_config(Path(args.config))
    errors = cfgmod.validate_config_dict(cfg)
    if errors:
        print("invalid config: " + "; ".join(errors), file=sys.stderr)
        return 2
    output_root = Path(args.output)
    data_cfg = cfg["data"]
    if not data_cfg.get("decontamination_hash"):
        decontamination_path = output_root / "data/decontamination.json"
        corpus = (
            json.loads(decontamination_path.read_text(encoding="utf-8"))
            if decontamination_path.exists()
            else None
        )
        if (
            not corpus
            or not verify_hashed_artifact(corpus)
            or corpus.get("policy_hash") != cfg["policy_hash"]
        ):
            suite_path = ROOT / cfg["eval"].get("dev_suite", "data/eval/dev.json")
            suite_doc = json.loads(suite_path.read_text(encoding="utf-8"))
            records = list(suite_doc["cases"])
            records.extend(lm_eval_decontamination_records(cfg["eval"].get("lm_eval_tasks", [])))
            corpus = build_decontamination_corpus(
                records,
                decontamination_path,
                sources=[
                    {"source": str(suite_path), "revision": suite_doc.get("revision")},
                    {"source": "lm-eval", "revision": cfg["eval"].get("lm_eval_revision")},
                ],
                policy_hash=cfg["policy_hash"],
            )
        data_cfg["decontamination_hash"] = corpus["artifact_hash"]

    def language_id() -> FastTextLanguageIdentifier:
        return language_identifier_from_config(data_cfg, cache_dir=output_root / "cache")

    def discover() -> Dict[str, Any]:
        fixture = data_cfg.get("discovery_fixture")
        infos = (
            json.loads(Path(fixture).read_text(encoding="utf-8"))
            if fixture
            else data_cfg.get("sources")
        )
        identifier = language_id()
        candidates = discover_sources(
            org=data_cfg["org"],
            allowed_licenses=data_cfg.get("allowed_licenses", []),
            sample_size=int(data_cfg.get("discovery_sample_size", 64)),
            dataset_infos=infos,
            language_id=identifier,
        )
        return _hashed(
            {
                "status": "ok",
                "mode": "real",
                "run_id": f"discovery-{stamp()}",
                "org": data_cfg["org"],
                "candidates": candidates,
                "language_id_hash": identifier.model_hash,
            }
        )

    def prepare(discovery: Dict[str, Any], selection: Dict[str, Any]) -> Dict[str, Any]:
        training_cfg = cfg["training"]
        from .training import load_tokenizer

        tokenizer = load_tokenizer(
            training_cfg["base_model"], revision=training_cfg.get("revision")
        )
        return materialize_streaming(
            selection["sources"],
            output_root / "data",
            language_id=language_id(),
            min_german_confidence=float(data_cfg.get("min_german_confidence", 0.8)),
            exact_dedup_limit=int(data_cfg.get("exact_dedup_limit", 2_000_000)),
            decontamination_hash=data_cfg.get("decontamination_hash"),
            decontamination_corpus=corpus,
            policy_hash=cfg.get("policy_hash"),
            discovery_run_id=discovery["run_id"],
            selection_run_id=selection["run_id"],
            seed=int(training_cfg["seed"]),
            max_rows_per_source=int(data_cfg.get("max_rows_per_source_real", 50_000)),
            global_max_rows=int(data_cfg.get("max_rows", 50_000)),
            validation_fraction=data_cfg["validation_fraction"],
            tokenizer=tokenizer,
            context_length=int(training_cfg["context_length"]),
            max_prompt_length=int(cfg["preference"]["max_prompt_length"]),
            max_completion_length=int(cfg["preference"]["max_completion_length"]),
        )

    def baseline() -> Dict[str, Any]:
        profile_dir = output_root / "baseline" / "dev"
        run_id = f"baseline-dev-{stamp()}"
        run_dir = profile_dir / run_id
        suite_path = resolve_suite(
            "dev", dev_path=ROOT / cfg["eval"].get("dev_suite", "data/eval/dev.json")
        )
        summary = run_real_evaluation(
            model_ref=cfg["training"]["base_model"],
            suite_path=suite_path,
            profile="dev",
            device=args.device,
            output_dir=run_dir,
            config=cfg,
            deadline=time.monotonic() + args.budget_minutes * 60,
        )
        summary["run_id"] = run_id
        finalize_summary(summary)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pointer = profile_dir / "current.json"
        if not pointer.exists():
            pointer.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    try:
        result = run_bootstrap(
            output_root=output_root,
            config=cfg,
            discover=discover,
            prepare=prepare,
            baseline=baseline,
        )
    except IntegrityError as exc:
        print(json.dumps({"status": "failed", "error": f"IntegrityError: {exc}"}))
        return 5
    except (OSError, RuntimeError, ValueError, MemoryError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 4
    print(json.dumps(result, ensure_ascii=False))
    append_event(output_root / "events.jsonl", {"event": "bootstrap", **result})
    return 0


def _policy_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    try:
        secure_policy = load_policy(path)
        recipe_path = ROOT / "configs/posttrain/recipe-policy.json"
        policy = json.loads(recipe_path.read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "allowed_licenses",
            "reward_weights",
            "reward_clamp",
            "promotion_gates",
        }
        missing = sorted(required - set(policy))
        if missing:
            raise ValueError("missing policy keys: " + ", ".join(missing))
        if policy.get("protected") is not True:
            raise ValueError("policy must be protected")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 5
    print(
        json.dumps(
            {
                "status": "ok",
                "policy": str(secure_policy.path),
                "recipe_policy": str(recipe_path),
            }
        )
    )
    return 0


def _eval_validate(args: argparse.Namespace) -> int:
    cfg = cfgmod.resolve_config(Path(args.config))
    path = (
        Path(args.suite)
        if args.suite
        else ROOT / cfg["eval"].get("dev_suite", "data/eval/dev.json")
    )
    try:
        suite = load_suite(path, profile="dev")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 5
    print(
        json.dumps(
            {"status": "ok", "cases": len(suite["cases"]), "suite_hash": suite["suite_hash"]}
        )
    )
    return 0


def _register_promotion(args: argparse.Namespace) -> int:
    try:
        suite_path = Path(args.path)
        document = register_promotion_suite(suite_path, Path(args.registry), repo_root=ROOT)
        suite_document = json.loads(suite_path.read_text(encoding="utf-8"))
        cases = suite_document.get("cases", suite_document)
        if not isinstance(cases, list):
            raise ValueError("promotion suite must contain case records")
        data_dir = ROOT / "outputs/posttrain/data"
        existing_path = data_dir / "decontamination.json"
        records = list(cases)
        sources = [{"source": "promotion", "suite_hash": document["suite_hash"]}]
        if existing_path.exists():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            records.extend(
                {"document": entry["canonical"]} for entry in existing.get("entries", [])
            )
            sources = existing.get("sources", []) + sources
        policy_hash = cfgmod.resolve_config(cfgmod.DEFAULT_CONFIG)["policy_hash"]
        corpus = build_decontamination_corpus(
            records, existing_path, sources=sources, policy_hash=policy_hash
        )
        document["decontamination_hash"] = corpus["artifact_hash"]
        Path(args.registry).write_text(json.dumps(document, indent=2), encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 5
    print(json.dumps({"status": "ok", **document}))
    return 0


def _failures_mine(args: argparse.Namespace) -> int:
    eval_dir = Path(args.evals) / args.eval_run
    try:
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        if not verify_hashed_artifact(summary):
            raise ValueError("evaluation summary artifact hash is invalid")
        raw_path = Path(summary["artifacts"]["raw_generations"])
        if file_sha256(raw_path) != summary["artifacts"]["raw_generations_sha256"]:
            raise ValueError("raw generation artifact hash is invalid")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        document = mine_failures(
            summary, raw, dimension_weights=policy.get("failure_dimension_weights", {})
        )
        out_dir = Path(args.output) / f"failures-{args.eval_run}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "failures.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        append_event(
            Path(args.output).parent / "events.jsonl",
            {
                "event": "failure_mining",
                "artifact": str(path),
                "artifact_hash": document["artifact_hash"],
            },
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 5
    print(json.dumps({"status": "ok", "artifact": str(path)}))
    return 0


def _train_rlvr(args: argparse.Namespace) -> int:
    if not (args.real and args.allow_gpu and args.allow_checkpoints):
        print("train rlvr requires --real --allow-gpu --allow-checkpoints", file=sys.stderr)
        return 2
    try:
        from datasets import IterableDataset

        cfg = cfgmod.resolve_config(Path(args.config))
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        data_dir = Path(args.data)
        manifest = verify_trainable_manifest(
            data_dir / "manifest.json", expected_policy_hash=cfg.get("policy_hash")
        )
        paths = [
            data_dir / shard["path"]
            for shard in manifest.get("shards", [])
            if shard.get("schema") == "rlvr"
        ]
        if not paths:
            raise ValueError("data manifest has no RLVR shards")
        minimum_vram = float(cfg.get("hardware", {}).get("minimum_vram_gb", 40))
        validate_device(args.device, minimum_vram_gb=minimum_vram)
        dataset = IterableDataset.from_generator(iter_rlvr_rows, gen_kwargs={"paths": paths})
        run_id = f"rlvr-{stamp()}"
        out_dir = Path(args.output) / run_id
        identifier = language_identifier_from_config(
            cfg["data"], cache_dir=Path(args.output).parent / "cache"
        )
        registered: list[Mapping[str, Any]] = []
        result = train_rlvr(
            model_ref=cfg["training"]["base_model"],
            dataset=dataset,
            output_dir=out_dir,
            config=cfg,
            policy=policy,
            device=args.device,
            deadline=time.monotonic() + args.budget_minutes * 60,
            language_id=identifier,
            register_candidate=registered.append,
        )
        if result["status"] == "ok":
            card = new_run_card(
                run_id,
                "train_rlvr",
                "pt train rlvr",
                model=cfg["training"]["base_model"],
                data_manifest=str(data_dir / "manifest.json"),
                metrics=result["metrics"],
                input_artifacts=[str(data_dir / "manifest.json")],
                output_artifacts=[result["adapter"]],
            )
            write_run_card(card, out_dir)
            append_event(
                Path(args.output).parent / "events.jsonl",
                {
                    "event": "train_rlvr",
                    "run_id": run_id,
                    "run_card": str(out_dir / "run_card.json"),
                },
            )
    except IntegrityError as exc:
        print(json.dumps({"status": "failed", "error": f"IntegrityError: {exc}"}))
        return 5
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 4
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 4


def _train_grpo(args: argparse.Namespace) -> int:
    if not (args.real and args.allow_gpu and args.allow_checkpoints):
        print("train grpo requires --real --allow-gpu --allow-checkpoints", file=sys.stderr)
        return 2
    try:
        from datasets import Dataset

        cfg = cfgmod.resolve_config(Path(args.config))
        if cfg.get("verified_rl", {}).get("enabled") is not True:
            raise ValueError("verified-math GRPO is disabled by verified_rl.enabled=false")
        data_dir = Path(args.data)
        verify_trainable_manifest(
            data_dir / "manifest.json", expected_policy_hash=cfg.get("policy_hash")
        )
        train_rows = load_manifest_rows(data_dir / "manifest.json", "verified_math", split="train")
        validation_rows = load_manifest_rows(
            data_dir / "manifest.json", "verified_math", split="validation"
        )
        minimum_vram = float(cfg.get("hardware", {}).get("minimum_vram_gb", 40))
        validate_device(args.device, minimum_vram_gb=minimum_vram)
        run_id = f"grpo-{stamp()}"
        out_dir = Path(args.output) / run_id
        result = train_verified_grpo(
            model_ref=cfg["training"]["base_model"],
            train_dataset=Dataset.from_list(train_rows),
            eval_dataset=Dataset.from_list(validation_rows),
            output_dir=out_dir,
            config=cfg,
            device=args.device,
            deadline=time.monotonic() + args.budget_minutes * 60,
        )
        card = new_run_card(
            run_id,
            "train_grpo",
            "pt train grpo",
            model=cfg["training"]["base_model"],
            data_manifest=str(data_dir / "manifest.json"),
            metrics=result["metrics"],
            input_artifacts=[str(data_dir / "manifest.json")],
            output_artifacts=[result["adapter"]],
        )
        write_run_card(card, out_dir)
        append_event(
            Path(args.output).parent / "events.jsonl",
            {"event": "train_grpo", "run_id": run_id, "run_card": str(out_dir / "run_card.json")},
        )
    except IntegrityError as exc:
        print(json.dumps({"status": "failed", "error": f"IntegrityError: {exc}"}))
        return 5
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 4
    print(json.dumps({"run_id": run_id, **result}, ensure_ascii=False))
    return 0


def _teacher(model_ref: str, device: str, sampling: Mapping[str, Any]):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("verified synthesis requires the train dependencies") from exc
    model_name, revision = model_ref.rsplit("@", 1) if "@" in model_ref else (model_ref, None)
    from .training import load_tokenizer

    tokenizer = load_tokenizer(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).to(device)
    model.eval()

    def generate(task: Mapping[str, Any], index: int) -> str:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(task["prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(rendered, return_tensors="pt").to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(sampling.get("seed", 17)) + index)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=True,
                temperature=float(sampling.get("temperature", 0.7)),
                top_p=float(sampling.get("top_p", 0.95)),
                max_new_tokens=int(sampling.get("max_new_tokens", 256)),
                generator=generator,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            output[0, encoded["input_ids"].shape[-1] :], skip_special_tokens=True
        )

    return generate


def _data_synthesize(args: argparse.Namespace) -> int:
    if not (args.real and args.allow_gpu):
        print("data synthesize requires --real --allow-gpu", file=sys.stderr)
        return 2
    try:
        cfg = cfgmod.resolve_config(Path(args.config))
        failure_path = Path(args.failures) / args.from_failure_run / "failures.json"
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        source_rows = (
            row
            for row in iter_manifest_rows(Path(args.data) / "manifest.json")
            if row.get("schema") == "cpt"
        )
        synthesis_cfg = cfg.get("synthesis", {})
        tasks = build_synthesis_tasks(
            failure,
            source_rows,
            seed=int(synthesis_cfg.get("seed", 17)),
            maximum=int(synthesis_cfg.get("maximum_examples", 128)),
        )
        sampling = synthesis_cfg.get(
            "sampling",
            {
                "temperature": 0.7,
                "top_p": 0.95,
                "max_new_tokens": 256,
                "seed": 17,
            },
        )
        identifier = language_identifier_from_config(
            cfg["data"], cache_dir=Path(args.output).parent / "cache"
        )
        decontam_path = Path(args.data) / "decontamination.json"
        eval_corpus = (
            json.loads(decontam_path.read_text(encoding="utf-8"))
            if decontam_path.exists()
            else None
        )
        if (
            not eval_corpus
            or not verify_hashed_artifact(eval_corpus)
            or eval_corpus.get("policy_hash") != cfg.get("policy_hash")
        ):
            raise ValueError("synthesis requires the current verified decontamination corpus")
        result = synthesize_verified(
            tasks,
            _teacher(args.teacher, args.device, sampling),
            teacher_ref=args.teacher,
            best_of_n=int(synthesis_cfg.get("best_of_n", 4)),
            sampling=sampling,
            language_id=identifier,
            eval_corpus=eval_corpus,
        )
        run_id = f"synthesis-{stamp()}"
        out_dir = Path(args.output) / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        shards = []
        for name in ("sft", "preferences", "rlvr"):
            path = out_dir / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in result[name]:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            artifacts[name] = str(path)
            shards.append(
                {
                    "path": path.name,
                    "schema": "preference" if name == "preferences" else name,
                    "source_group": "verified_synthetic",
                    "rows": len(result[name]),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        manifest = _hashed(
            {
                "schema_version": 2,
                "status": "trainable",
                "mode": "real",
                "run_id": run_id,
                "from_failure_run": args.from_failure_run,
                "teacher": args.teacher,
                "counts": {name: len(result[name]) for name in ("sft", "preferences", "rlvr")},
                "discarded": result["discarded"],
                "artifacts": artifacts,
                "shards": shards,
                "row_counts": {
                    "written": sum(
                        len(result[name]) for name in result if isinstance(result[name], list)
                    )
                },
                "decontamination_hash": eval_corpus["artifact_hash"],
                "policy_hash": cfg["policy_hash"],
            }
        )
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (out_dir / "decontamination.json").write_text(
            json.dumps(eval_corpus, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "leakage_report.json").write_text(
            json.dumps(
                {
                    "status": "verified_clean",
                    "decontamination_hash": eval_corpus["artifact_hash"],
                    "overlap_hits": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        append_event(
            Path(args.output).parent / "events.jsonl",
            {
                "event": "synthesis",
                "run_id": run_id,
                "artifact_hash": manifest["artifact_hash"],
            },
        )
    except IntegrityError as exc:
        print(json.dumps({"status": "failed", "error": f"IntegrityError: {exc}"}))
        return 5
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 4
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


def _search_run(args: argparse.Namespace) -> int:
    if not (args.real and args.allow_gpu and args.allow_checkpoints):
        print("search run requires --real --allow-gpu --allow-checkpoints", file=sys.stderr)
        return 2
    try:
        cfg = cfgmod.resolve_config(Path(args.config))
        plan = cfg.get("search_plan") or cfg.get("experiment", {}).get("search_plan")
        if not isinstance(plan, dict):
            raise ValueError("config contains no search_plan")
        lever = str(cfg.get("experiment", {}).get("lever", "sft"))
        if lever not in {"sft", "preference", "rlvr", "grpo"}:
            raise ValueError("search lever must be sft, preference, rlvr, or grpo")
        if lever == "preference" and cfg.get("preference", {}).get("enabled") is not True:
            raise ValueError("preference search is disabled by preference.enabled=false")
        if lever == "grpo" and cfg.get("verified_rl", {}).get("enabled") is not True:
            raise ValueError("GRPO search is disabled by verified_rl.enabled=false")
        validate_device(
            args.device,
            minimum_vram_gb=float(cfg.get("hardware", {}).get("minimum_vram_gb", 40)),
        )
        search_id = f"search-{stamp()}"
        root = Path(args.output) / search_id
        suite = ROOT / cfg["eval"].get("dev_suite", "data/eval/dev.json")
        deadline = time.monotonic() + args.budget_minutes * 60
        if lever == "rlvr":
            from datasets import IterableDataset

            data_dir = Path(args.data)
            manifest = verify_trainable_manifest(
                data_dir / "manifest.json", expected_policy_hash=cfg.get("policy_hash")
            )
            rlvr_paths = [
                data_dir / shard["path"]
                for shard in manifest.get("shards", [])
                if shard.get("schema") == "rlvr"
            ]
            if not rlvr_paths:
                raise ValueError("RLOO search requires RLVR shards")
            policy = json.loads(
                (ROOT / "configs/posttrain/recipe-policy.json").read_text(encoding="utf-8")
            )
            identifier = language_identifier_from_config(
                cfg["data"], cache_dir=Path(args.output).parent / "cache"
            )
        elif lever == "grpo":
            from datasets import Dataset

            data_dir = Path(args.data)
            verify_trainable_manifest(
                data_dir / "manifest.json", expected_policy_hash=cfg.get("policy_hash")
            )
            grpo_train = Dataset.from_list(
                load_manifest_rows(data_dir / "manifest.json", "verified_math", split="train")
            )
            grpo_eval = Dataset.from_list(
                load_manifest_rows(data_dir / "manifest.json", "verified_math", split="validation")
            )

        def train_proxy(trial, rung, parent):
            trial_cfg = json.loads(json.dumps(cfg))
            trial_cfg["training"].update(trial["overrides"])
            trial_cfg["training"]["max_steps"] = rung.additional_steps
            run_id = f"{search_id}-{trial['trial_id']}-r{rung.index}"
            run_dir = root / run_id
            if lever == "rlvr":
                dataset = IterableDataset.from_generator(
                    iter_rlvr_rows, gen_kwargs={"paths": rlvr_paths}
                )
                trained = train_rlvr(
                    model_ref=trial_cfg["training"]["base_model"],
                    dataset=dataset,
                    output_dir=run_dir,
                    config=trial_cfg,
                    policy=policy,
                    device=args.device,
                    deadline=deadline,
                    language_id=identifier,
                    parent_adapter=Path(parent["adapter"]) if parent else None,
                )
                if trained["status"] != "ok":
                    return {"status": trained["status"], "technical_error_count": 1}
                metrics = trained["metrics"]
                adapter = trained["adapter"]
            elif lever == "grpo":
                trained = train_verified_grpo(
                    model_ref=trial_cfg["training"]["base_model"],
                    train_dataset=grpo_train,
                    eval_dataset=grpo_eval,
                    output_dir=run_dir,
                    config=trial_cfg,
                    device=args.device,
                    deadline=deadline,
                    parent_adapter=Path(parent["adapter"]) if parent else None,
                )
                if trained["status"] != "succeeded":
                    return {"status": trained["status"], "technical_error_count": 1}
                metrics = trained["metrics"]
                adapter = trained["adapter"]
            else:
                training_kind = "preference" if lever == "preference" else "specialist"
                metrics = _train_real(
                    cfg=trial_cfg,
                    kind=training_kind,
                    data_dir=Path(args.data),
                    out_dir=run_dir,
                    deadline=deadline,
                    device=args.device,
                    parent_adapter=Path(parent["adapter"]) if parent else None,
                )
                adapter = str(run_dir / "adapter")
            summary = run_real_evaluation(
                model_ref=adapter,
                suite_path=suite,
                profile="proxy",
                device=args.device,
                output_dir=run_dir / "proxy",
                config=trial_cfg,
                deadline=deadline,
            )
            card_metrics = {
                **metrics,
                "proxy_score": aggregate(summary["metrics"]),
                "continuation_mode": ("adapter_fresh_optimizer" if parent else "new_adapter"),
                "parent_run_id": parent.get("run_id") if parent else None,
            }
            write_run_card(
                new_run_card(
                    run_id,
                    (
                        "train_rlvr"
                        if lever == "rlvr"
                        else "train_grpo"
                        if lever == "grpo"
                        else "train_preference"
                        if lever == "preference"
                        else "train_specialist"
                    ),
                    "pt search run",
                    model=trial_cfg["training"]["base_model"],
                    data_manifest=str(Path(args.data) / "manifest.json"),
                    metrics=card_metrics,
                    input_artifacts=[parent["adapter"]] if parent else [],
                    output_artifacts=[adapter],
                    notes=card_metrics["continuation_mode"],
                ),
                run_dir,
            )
            return {
                "status": summary["status"],
                "run_id": run_id,
                "adapter": adapter,
                "technical_error_count": summary["technical_error_count"],
                "hard_gates_passed": summary["status"] == "ok"
                and all(summary.get("hard_gates", {}).values()),
                "proxy_score": card_metrics["proxy_score"],
                **metrics,
            }

        def dev(winner):
            summary = run_real_evaluation(
                model_ref=winner["adapter"],
                suite_path=suite,
                profile="dev",
                device=args.device,
                output_dir=root / winner["run_id"] / "dev",
                config=cfg,
                deadline=deadline,
            )
            summary["run_id"] = winner["run_id"]
            return finalize_summary(summary)

        result = run_successive_halving(
            plan=plan,
            full_budget=int(cfg["training"]["max_steps"]),
            train_and_proxy=train_proxy,
            dev_evaluate=dev,
            lever=lever,
        )
        root.mkdir(parents=True, exist_ok=True)
        (root / "search_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result["status"] == "ok":
            frontier_path = Path(args.output).parent / "specialist-frontiers.json"
            existing: Dict[str, Any] = {
                "general": current_frontier() or None,
                "specialists": {},
            }
            if frontier_path.exists():
                existing = json.loads(frontier_path.read_text(encoding="utf-8"))
                if not verify_hashed_artifact(existing):
                    raise IntegrityError("specialist frontier artifact hash is invalid")
                existing["general"] = current_frontier() or existing.get("general")
            frontier = update_specialist_frontiers([result["dev_evaluation"]], existing)
            frontier_path.write_text(
                json.dumps(frontier, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        append_event(
            Path(args.output).parent / "events.jsonl",
            {
                "event": "search",
                "run_id": search_id,
                "status": result["status"],
                "specialist_frontier_hash": (
                    frontier["artifact_hash"] if result["status"] == "ok" else None
                ),
            },
        )
    except IntegrityError as exc:
        print(json.dumps({"status": "failed", "error": f"IntegrityError: {exc}"}))
        return 5
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 4
    print(
        json.dumps(
            {"status": result["status"], "search_id": search_id, "winner": result.get("winner")},
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "ok" else (1 if result["status"] == "rejected" else 4)


def _mix_probe(args: argparse.Namespace) -> int:
    if not (args.real and args.allow_gpu and args.allow_checkpoints):
        print("mix probe requires --real --allow-gpu --allow-checkpoints", file=sys.stderr)
        return 2
    try:
        cfg = cfgmod.resolve_config(Path(args.config))
        policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
        data_dir = Path(args.data)
        manifest = verify_trainable_manifest(
            data_dir / "manifest.json", expected_policy_hash=cfg.get("policy_hash")
        )
        groups = sorted(
            {
                str(shard["source_group"])
                for shard in manifest.get("shards", [])
                if shard.get("schema") == "sft"
            }
        )
        if not groups:
            raise ValueError("mix probe requires SFT source groups")
        validate_device(
            args.device, minimum_vram_gb=float(cfg.get("hardware", {}).get("minimum_vram_gb", 40))
        )
        run_id = f"mix-probe-{stamp()}"
        root = Path(args.output) / run_id
        deadline = time.monotonic() + args.budget_minutes * 60
        suite = ROOT / cfg["eval"].get("dev_suite", "data/eval/dev.json")
        baseline = run_real_evaluation(
            model_ref=cfg["training"]["base_model"],
            suite_path=suite,
            profile="proxy",
            device=args.device,
            output_dir=root / "baseline",
            config=cfg,
            deadline=deadline,
        )
        baseline_score = aggregate(baseline["metrics"])
        if baseline_score is None:
            raise RuntimeError("proxy baseline has no aggregate score")

        def probe(group: str, steps: int):
            probe_cfg = json.loads(json.dumps(cfg))
            probe_cfg["training"]["max_steps"] = steps
            out_dir = root / group
            metrics = _train_real(
                cfg=probe_cfg,
                kind="specialist",
                data_dir=data_dir,
                out_dir=out_dir,
                deadline=deadline,
                device=args.device,
                source_group=group,
            )
            summary = run_real_evaluation(
                model_ref=str(out_dir / "adapter"),
                suite_path=suite,
                profile="proxy",
                device=args.device,
                output_dir=out_dir / "proxy",
                config=probe_cfg,
                deadline=deadline,
            )
            score = aggregate(summary["metrics"])
            if score is None or summary["technical_error_count"]:
                raise RuntimeError(f"proxy evaluation failed for source group {group}")
            return {
                "status": "ok",
                "proxy_score_delta": score - baseline_score,
                "gpu_minutes": metrics["gpu_seconds"] / 60,
                "tokens_seen": metrics.get("num_input_tokens_seen", 0),
                "peak_vram": metrics.get("peak_vram"),
            }

        plan = run_mix_probes(
            groups, probe, minimum_weights=policy.get("minimum_source_weights", {})
        )
        document = _hashed(
            {**plan, "run_id": run_id, "data_manifest_hash": manifest.get("artifact_hash")}
        )
        root.mkdir(parents=True, exist_ok=True)
        path = root / "mix_plan.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        append_event(
            Path(args.output).parent / "events.jsonl",
            {
                "event": "mix_probe",
                "run_id": run_id,
                "artifact_hash": document["artifact_hash"],
            },
        )
    except IntegrityError as exc:
        print(json.dumps({"status": "failed", "error": f"IntegrityError: {exc}"}))
        return 5
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 4
    print(json.dumps({"status": "ok", "run_id": run_id, "mix_plan": str(path)}))
    return 0


def _benchmark_compare(args: argparse.Namespace) -> int:
    try:
        sft = json.loads(Path(args.sft_result).read_text(encoding="utf-8"))
        rlvr = json.loads(Path(args.rlvr_result).read_text(encoding="utf-8"))
        model_start = str(sft["model_start"])
        prompt_group = str(sft["prompt_group"])
        gpu_minutes = float(sft["gpu_budget_minutes"])
        if (
            rlvr.get("model_start") != model_start
            or rlvr.get("prompt_group") != prompt_group
            or float(rlvr.get("gpu_budget_minutes", -1)) != gpu_minutes
        ):
            raise ValueError("SFT and RLVR results must share model, prompts, and GPU-time budget")
        report = compare_sft_rlvr(
            sft_run=lambda **_kwargs: sft,
            rlvr_run=lambda **_kwargs: rlvr,
            model_start=model_start,
            prompt_group=prompt_group,
            gpu_minutes=gpu_minutes,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 4
    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(out),
                "quality_winner": report["quality_winner"],
                "efficiency_winner": report["efficiency_winner"],
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pt")
    commands = parser.add_subparsers(dest="command", required=True)

    bootstrap = commands.add_parser("bootstrap")
    bootstrap_sub = bootstrap.add_subparsers(dest="action", required=True)
    bootstrap_run = bootstrap_sub.add_parser("run")
    bootstrap_run.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    bootstrap_run.add_argument("--output", default=str(ROOT / "outputs/posttrain"))
    bootstrap_run.add_argument("--real", action="store_true")
    bootstrap_run.add_argument("--allow-gpu", action="store_true")
    bootstrap_run.add_argument("--device", default="cuda:0")
    bootstrap_run.add_argument("--budget-minutes", type=int, default=90)
    bootstrap_run.set_defaults(handler=_bootstrap)

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
    eval_sub = evaluation.add_subparsers(dest="action", required=True)
    eval_run = eval_sub.add_parser("run")
    eval_run.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    eval_run.add_argument("--model", default=None)
    eval_run.add_argument("--candidate", default=None)
    eval_run.add_argument("--label", default=None)
    eval_run.add_argument("--out", default=str(ROOT / "outputs/posttrain/evals"))
    eval_run.add_argument("--device", default="cuda:0")
    eval_run.add_argument("--profile", choices=["proxy", "dev", "promotion"], default="dev")
    eval_run.add_argument("--suite", default=None)
    eval_run.add_argument("--budget-minutes", type=int, default=90)
    eval_run.add_argument("--real", action="store_true")
    eval_run.add_argument("--dry-run", action="store_true")
    eval_run.add_argument("--allow-gpu", action="store_true")
    eval_run.set_defaults(handler=_eval_run_command)
    validate = eval_sub.add_parser("validate-suite")
    validate.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    validate.add_argument("--suite", default=None)
    validate.set_defaults(handler=_eval_validate)
    catalog = eval_sub.add_parser("catalog")
    catalog.set_defaults(handler=_eval_catalog_command)
    register = eval_sub.add_parser("register-promotion-suite")
    register.add_argument("--path", required=True)
    register.add_argument(
        "--registry", default=str(ROOT / "configs/posttrain/promotion-suite.json")
    )
    register.set_defaults(handler=_register_promotion)

    baseline = commands.add_parser("baseline")
    baseline_sub = baseline.add_subparsers(dest="action", required=True)
    baseline_run = baseline_sub.add_parser("run")
    baseline_run.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    baseline_run.add_argument("--out", default=str(ROOT / "outputs/posttrain/baseline"))
    baseline_run.add_argument("--model", default=None)
    baseline_run.add_argument("--label", default="baseline-seed")
    baseline_run.add_argument("--device", default="cuda:0")
    baseline_run.add_argument("--profile", choices=["dev", "promotion"], default="dev")
    baseline_run.add_argument("--suite", default=None)
    baseline_run.add_argument("--budget-minutes", type=int, default=90)
    baseline_run.add_argument("--real", action="store_true")
    baseline_run.add_argument("--dry-run", action="store_true")
    baseline_run.add_argument("--allow-gpu", action="store_true")
    baseline_run.set_defaults(handler=_baseline_run_command)

    score = commands.add_parser("score")
    score.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    score.add_argument("--run", default=None)
    score.add_argument("--candidate", default=None)
    score.add_argument("--baseline", default=None)
    score.add_argument("--out", default=None)
    score.add_argument("--profile", choices=["dev", "promotion"], default="dev")
    score.add_argument("--format", choices=["json", "markdown"], default="json")
    score.set_defaults(handler=_score_command)

    failures = commands.add_parser("failures")
    failures_sub = failures.add_subparsers(dest="action", required=True)
    mine = failures_sub.add_parser("mine")
    mine.add_argument("--eval-run", required=True)
    mine.add_argument("--evals", default=str(ROOT / "outputs/posttrain/evals"))
    mine.add_argument("--output", default=str(ROOT / "outputs/posttrain/failures"))
    mine.add_argument("--policy", default=str(ROOT / "configs/posttrain/recipe-policy.json"))
    mine.set_defaults(handler=_failures_mine)

    train = commands.add_parser("train")
    train_sub = train.add_subparsers(dest="action", required=True, parser_class=GatedParser)
    for name in ("sft", "cpt", "preference"):
        secure_train = train_sub.add_parser(name)
        _explicit_mode(secure_train, gpu=True)
        secure_train.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
        secure_train.add_argument("--budget-minutes", type=float)
        secure_train.add_argument("--specialist")
        secure_train.add_argument("--out", default=str(ROOT / "outputs/posttrain/runs"))
        secure_train.add_argument("--data", default=str(ROOT / "outputs/posttrain/data"))
        secure_train.add_argument("--device", default="cuda:0")
        secure_train.add_argument("--mix-plan")
        if name == "preference":
            secure_train.add_argument("--method", choices=("dpo", "kto", "orpo"))
        secure_train.set_defaults(handler=_train_command)
    rlvr = train_sub.add_parser("rlvr")
    rlvr.add_argument("--real", action="store_true")
    rlvr.add_argument("--allow-gpu", action="store_true")
    rlvr.add_argument("--allow-checkpoints", action="store_true")
    rlvr.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    rlvr.add_argument("--policy", default=str(ROOT / "configs/posttrain/recipe-policy.json"))
    rlvr.add_argument("--data", default=str(ROOT / "outputs/posttrain/data"))
    rlvr.add_argument("--output", default=str(ROOT / "outputs/posttrain/runs"))
    rlvr.add_argument("--device", default="cuda:0")
    rlvr.add_argument("--budget-minutes", type=int, default=90)
    rlvr.set_defaults(handler=_train_rlvr)
    grpo = train_sub.add_parser("grpo")
    grpo.add_argument("--real", action="store_true")
    grpo.add_argument("--allow-gpu", action="store_true")
    grpo.add_argument("--allow-checkpoints", action="store_true")
    grpo.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    grpo.add_argument("--data", default=str(ROOT / "outputs/posttrain/data"))
    grpo.add_argument("--output", default=str(ROOT / "outputs/posttrain/runs"))
    grpo.add_argument("--device", default="cuda:0")
    grpo.add_argument("--budget-minutes", type=int, default=90)
    grpo.set_defaults(handler=_train_grpo)

    data = commands.add_parser("data")
    data_sub = data.add_subparsers(dest="action", required=True, parser_class=GatedParser)
    discover = data_sub.add_parser("discover")
    _explicit_mode(discover)
    discover.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    discover.add_argument("--out", default=str(ROOT / "outputs/posttrain/data/discovery.json"))
    discover.add_argument("--format", choices=("json", "markdown"), default="json")
    discover.set_defaults(handler=_data_discover_command)
    prepare = data_sub.add_parser("prepare")
    prepare.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    prepare.add_argument("--discovery", default=str(ROOT / "outputs/posttrain/data/discovery.json"))
    prepare.add_argument("--selection", default=str(ROOT / "outputs/posttrain/data/selection.json"))
    prepare.add_argument("--discovery-run-id")
    prepare.add_argument("--selection-run-id")
    prepare.add_argument("--out", default=str(ROOT / "outputs/posttrain/data"))
    prepare.add_argument("--format", choices=["json", "markdown"], default="json")
    prepare.add_argument("--real", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")
    prepare.set_defaults(handler=_data_prepare_command)
    synthesize = data_sub.add_parser("synthesize")
    synthesize.add_argument("--from-failure-run", required=True)
    synthesize.add_argument("--teacher", required=True)
    synthesize.add_argument("--real", action="store_true")
    synthesize.add_argument("--allow-gpu", action="store_true")
    synthesize.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    synthesize.add_argument("--data", default=str(ROOT / "outputs/posttrain/data"))
    synthesize.add_argument("--failures", default=str(ROOT / "outputs/posttrain/failures"))
    synthesize.add_argument("--output", default=str(ROOT / "outputs/posttrain/synthesis"))
    synthesize.add_argument("--device", default="cuda:0")
    synthesize.set_defaults(handler=_data_synthesize)

    search = commands.add_parser("search")
    search_sub = search.add_subparsers(dest="action", required=True)
    search_run = search_sub.add_parser("run")
    search_run.add_argument("--real", action="store_true")
    search_run.add_argument("--allow-gpu", action="store_true")
    search_run.add_argument("--allow-checkpoints", action="store_true")
    search_run.add_argument("--config", required=True)
    search_run.add_argument("--data", default=str(ROOT / "outputs/posttrain/data"))
    search_run.add_argument("--output", default=str(ROOT / "outputs/posttrain/search"))
    search_run.add_argument("--device", default="cuda:0")
    search_run.add_argument("--budget-minutes", type=int, default=180)
    search_run.set_defaults(handler=_search_run)

    mix = commands.add_parser("mix")
    mix_sub = mix.add_subparsers(dest="action", required=True)
    mix_probe = mix_sub.add_parser("probe")
    mix_probe.add_argument("--real", action="store_true")
    mix_probe.add_argument("--allow-gpu", action="store_true")
    mix_probe.add_argument("--allow-checkpoints", action="store_true")
    mix_probe.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    mix_probe.add_argument("--policy", default=str(ROOT / "configs/posttrain/recipe-policy.json"))
    mix_probe.add_argument("--data", default=str(ROOT / "outputs/posttrain/data"))
    mix_probe.add_argument("--output", default=str(ROOT / "outputs/posttrain/mix"))
    mix_probe.add_argument("--device", default="cuda:0")
    mix_probe.add_argument("--budget-minutes", type=int, default=180)
    mix_probe.set_defaults(handler=_mix_probe)

    benchmark = commands.add_parser("benchmark")
    benchmark_sub = benchmark.add_subparsers(dest="action", required=True)
    compare = benchmark_sub.add_parser("compare-sft-rlvr")
    compare.add_argument("--sft-result", required=True)
    compare.add_argument("--rlvr-result", required=True)
    compare.add_argument(
        "--output", default=str(ROOT / "outputs/posttrain/benchmarks/sft-vs-rlvr.json")
    )
    compare.set_defaults(handler=_benchmark_compare)

    merge = commands.add_parser("merge")
    merge_sub = merge.add_subparsers(dest="action", required=True, parser_class=GatedParser)
    merge_search = merge_sub.add_parser("search")
    _explicit_mode(merge_search, gpu=True)
    merge_search.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    merge_search.add_argument("--budget-minutes", type=float)
    merge_search.add_argument("--runs", default=str(ROOT / "outputs/posttrain/runs"))
    merge_search.add_argument(
        "--frontier", default=str(ROOT / "outputs/posttrain/specialist-frontiers.json")
    )
    merge_search.add_argument("--out", default=str(ROOT / "outputs/posttrain/merge"))
    merge_search.add_argument("--format", choices=("json", "markdown"), default="json")
    merge_search.add_argument("--merge-device", default="cpu")
    merge_search.add_argument("--device", default="cuda:0")
    merge_search.set_defaults(handler=_merge_command)

    distill = commands.add_parser("distill")
    _explicit_mode(distill, gpu=True)
    distill.add_argument("--config", default="configs/posttrain/secure-current.json")
    distill.add_argument("--teacher", required=True)
    distill.add_argument("--teacher-license")
    distill.add_argument("--budget-minutes", type=float)
    distill.set_defaults(handler=_distill_command)

    script_map = {
        "status": "pt_status",
        "report": "pt_report",
        "integrity": "check_posttrain_integrity",
    }
    for name, stem in script_map.items():
        command = commands.add_parser(name, add_help=False)
        command.set_defaults(handler=lambda args, value=stem: _script(value, args.remainder))
        command.add_argument("remainder", nargs=argparse.REMAINDER)
    loop = commands.add_parser("loop", add_help=False)
    loop.set_defaults(handler=_loop_command)
    loop.add_argument("remainder", nargs=argparse.REMAINDER)
    promote = commands.add_parser("promote")
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--base-ref", required=True)
    promote.add_argument("--config", default=str(cfgmod.DEFAULT_CONFIG))
    promote.add_argument("--baseline")
    promote.add_argument("--out", default=str(ROOT / "outputs/posttrain/promote"))
    promote.add_argument("--format", choices=("json", "markdown"), default="json")
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
