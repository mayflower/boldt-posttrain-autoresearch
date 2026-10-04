"""Shared experiment/data contracts for online RL and on-policy distillation.

No model or evaluation corpus is imported here. Verifier labels come exclusively
from the verified training manifest. ``distill`` is the OPD lever, not offline SFT.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import verify_artifact_ref
from .policy import Policy

ONLINE_LEVERS = {"grpo", "rlvr", "opd", "sdpo", "sdft", "distill"}
VERIFIABLE_TASKS = {"exact", "numeric", "json_schema", "ordered_terms", "math_accuracy"}


def online_method(lever: str) -> str:
    if lever not in ONLINE_LEVERS:
        raise ValueError(f"unsupported online lever: {lever}")
    return {"distill": "opd", "rlvr": "rloo"}.get(lever, lever)


def online_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    lever = config["experiment"]["lever"]
    if lever not in ONLINE_LEVERS:
        if "online" in config:
            raise ValueError("config.online requires an online experiment lever")
        return {}
    method = online_method(lever)
    training = config["training"]
    distill = config["distillation"]
    raw = config.get("online", {})
    if not isinstance(raw, dict):
        raise ValueError("config.online must be an object")
    maximum = int(distill["max_new_tokens"])
    defaults: dict[str, Any] = {
        "batch_size": 4 if method in {"grpo", "rloo"} else training["per_device_batch_size"],
        "max_prompt_length": int(training["context_length"]) - maximum,
        "max_completion_length": maximum,
        "temperature": 1.0,
        "max_grad_norm": 1.0,
    }
    if method in {"grpo", "rloo", "sdpo"}:
        defaults["reward_profile"] = "mechanical"
    if method in {"grpo", "rloo"}:
        defaults.update(num_generations=4, beta=0.05, validation_fraction=0.1)
    else:
        defaults.update(
            min_completion_length=int(distill["min_new_tokens"]),
            logit_chunk_size=32,
        )
    if method == "grpo":
        defaults.update(loss_type="dapo", scale_rewards="group")
    if method in {"sdft", "sdpo"}:
        defaults["teacher_ema_decay"] = 0.99
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(f"unknown config.online keys for {method}: {sorted(unknown)}")
    options = {**defaults, **raw}
    for key in {
        "batch_size",
        "max_prompt_length",
        "max_completion_length",
        "logit_chunk_size",
    } & set(options):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"config.online.{key} must be a positive integer")
    for key in {
        "temperature",
        "max_grad_norm",
        "beta",
        "teacher_ema_decay",
        "validation_fraction",
    } & set(options):
        value = options[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"config.online.{key} must be finite numeric")
    if options["temperature"] <= 0 or options["max_grad_norm"] <= 0:
        raise ValueError("online temperature and max_grad_norm must be positive")
    if options["max_prompt_length"] + options["max_completion_length"] > training["context_length"]:
        raise ValueError("online prompt and completion lengths exceed training.context_length")
    if "num_generations" in options:
        count = options["num_generations"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("online.num_generations must be an integer >= 2")
        if options["batch_size"] % count:
            raise ValueError("online.batch_size must be divisible by num_generations")
        if options["beta"] < 0:
            raise ValueError("online.beta must be non-negative")
        if not 0 < options["validation_fraction"] < 0.5:
            raise ValueError("online.validation_fraction must be between zero and 0.5")
    if "min_completion_length" in options:
        minimum = options["min_completion_length"]
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or not 0 <= minimum <= options["max_completion_length"]
        ):
            raise ValueError("online.min_completion_length must fit the completion limit")
    if "teacher_ema_decay" in options and not 0 <= options["teacher_ema_decay"] < 1:
        raise ValueError("online.teacher_ema_decay must satisfy 0 <= value < 1")
    if options.get("reward_profile", "mechanical") not in {
        "mechanical",
        "reference_exact",
        "math_accuracy",
    }:
        raise ValueError("unsupported online.reward_profile")
    if method == "grpo":
        if options["loss_type"] not in {"grpo", "dapo", "dr_grpo"}:
            raise ValueError("online.loss_type must be grpo, dapo, or dr_grpo")
        if options["scale_rewards"] not in {"group", "batch", "none"}:
            raise ValueError("online.scale_rewards must be group, batch, or none")
    return options


def validate_online_policy(config: Mapping[str, Any], policy: Policy) -> None:
    method = online_method(config["experiment"]["lever"])
    allowed = policy.document["training"]["allowed_methods"]
    if method not in allowed:
        raise ValueError(
            f"online objective {method!r} is not authorized by policy.training.allowed_methods"
        )
    training = config["training"]
    if training["method"] not in {"lora", "qlora"} or training["method"] not in allowed:
        raise ValueError(
            "online training requires an explicitly policy-allowed lora or qlora method"
        )
    if training["specialist"] not in policy.document["training"]["allowed_specialists"]:
        raise ValueError("online specialist is not policy-allowed")
    online_settings(config)


def prompt_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if not isinstance(value, list) or not value:
        raise ValueError("online prompt must be a non-empty conversation")
    for item in value:
        if not isinstance(item, dict) or set(item) != {"role", "content"}:
            raise ValueError("online message must contain exactly role and content")
        if (
            item["role"] not in {"system", "user", "assistant"}
            or not isinstance(item["content"], str)
            or not item["content"].strip()
        ):
            raise ValueError("online message has an invalid role or empty content")
    if value[-1]["role"] != "user":
        raise ValueError("online prompt must end with a user message")
    return [dict(item) for item in value]


def validate_truth(kind: str, truth: Any) -> None:
    if kind not in VERIFIABLE_TASKS or not isinstance(truth, dict):
        raise ValueError("online row requires a supported task_type and ground_truth object")
    if kind in {"exact", "math_accuracy"}:
        if not isinstance(truth.get("value"), str) or not truth["value"].strip():
            raise ValueError("exact/math ground truth must be non-empty text")
    elif kind == "numeric":
        value, tolerance = truth.get("value"), truth.get("tolerance", 0.0)
        if (
            any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (value, tolerance)
            )
            or tolerance < 0
        ):
            raise ValueError(
                "numeric ground truth requires a finite value and non-negative tolerance"
            )
    elif kind == "json_schema":
        if not isinstance(truth.get("schema"), dict):
            raise ValueError("JSON ground truth requires a schema object")
        import jsonschema

        jsonschema.Draft202012Validator.check_schema(truth["schema"])
        if "value" in truth:
            jsonschema.validate(truth["value"], truth["schema"])
    elif kind == "ordered_terms":
        terms = truth.get("terms")
        if (
            not isinstance(terms, list)
            or not terms
            or any(not isinstance(term, str) or not term.strip() for term in terms)
        ):
            raise ValueError("ordered_terms ground truth must contain non-empty terms")
        if len({term.casefold() for term in terms}) != len(terms):
            raise ValueError("ordered_terms ground truth must contain distinct terms")


def parse_gold_solution(solution: str) -> Any:
    try:
        from math_verify import parse
    except ImportError as exc:
        raise RuntimeError("math verification requires the rl extra") from exc
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("math gold solution must be non-empty text")
    parsed = parse(solution)
    if not parsed:
        raise ValueError(f"math-verify could not parse gold solution: {solution!r}")
    return parsed


def math_accuracy(response: str, solution: str) -> float:
    """Exact Math-Verify accuracy; an unparseable answer scores zero, never partial credit."""
    from math_verify import parse, verify

    gold = parse_gold_solution(solution)
    prediction = parse(response)
    return float(bool(prediction) and verify(gold, prediction))


def verify_response(response: str, kind: str, truth: Mapping[str, Any]) -> tuple[float, str]:
    validate_truth(kind, truth)
    if kind == "math_accuracy":
        reward = math_accuracy(response, truth["value"])
    elif kind == "json_schema":
        import jsonschema

        try:
            decoded = json.loads(response)
            jsonschema.validate(decoded, truth["schema"])
        except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
            return (
                0.0,
                f"Ungültiges JSON: {exc}. Vorgabe: {json.dumps(dict(truth), ensure_ascii=False)}",
            )
        reward = float("value" not in truth or decoded == truth["value"])
    elif kind == "ordered_terms":
        positions = [response.casefold().find(term.casefold()) for term in truth["terms"]]
        reward = float(
            all(p >= 0 for p in positions) and all(a < b for a, b in zip(positions, positions[1:]))
        )
    elif kind == "numeric":
        from .verifiers import numeric_matches_ground_truth

        reward = float(numeric_matches_ground_truth(response, truth))
    else:
        reward = float(response.strip() == truth["value"].strip())
    feedback = (
        "Die Antwort erfüllt die geprüfte Vorgabe."
        if reward
        else f"Die Antwort erfüllt die Vorgabe nicht. Erwartete Vorgabe: {json.dumps(dict(truth), ensure_ascii=False)}"
    )
    return reward, feedback


def online_rows(
    manifest: Mapping[str, Any], config: Mapping[str, Any], *, root: Path
) -> list[dict[str, Any]]:
    method = online_method(config["experiment"]["lever"])
    options = online_settings(config)
    profile = options.get("reward_profile")
    kind = (
        "verified_math"
        if profile == "math_accuracy"
        else "rlvr"
        if profile == "mechanical"
        else "sft"
    )
    refs = [ref for ref in manifest["shards"] if ref["role"] == f"{kind}_shard"]
    if not refs:
        raise ValueError(f"online {method} requires a verified {kind}_shard; no data fallback")
    result = []
    for ref in refs:
        verify_artifact_ref(ref, root=root)
        path = Path(ref["path"])
        path = path if path.is_absolute() else root / path
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if kind == "sft":
                messages = row["messages"]
                if not messages or messages[-1]["role"] != "assistant":
                    raise ValueError("online SFT source must end with an assistant demonstration")
                item = {
                    "prompt": prompt_messages(messages[:-1]),
                    "demonstration": messages[-1]["content"],
                    "content_id": row["content_id"],
                }
                if profile == "reference_exact":
                    item.update(task_type="exact", ground_truth={"value": item["demonstration"]})
            else:
                item = {
                    "prompt": prompt_messages(row["prompt"]),
                    "content_id": row["content_id"],
                    "task_type": "math_accuracy" if kind == "verified_math" else row["task_type"],
                    "ground_truth": {"value": row["solution"]}
                    if kind == "verified_math"
                    else row["ground_truth"],
                }
            if "task_type" in item:
                validate_truth(item["task_type"], item["ground_truth"])
                if item["task_type"] == "math_accuracy":
                    parse_gold_solution(item["ground_truth"]["value"])
            if method == "sdft" and not item.get("demonstration"):
                raise ValueError("SDFT requires a non-empty demonstration")
            result.append(item)
    if not result:
        raise ValueError("online dataset is empty")
    if method not in {"grpo", "rloo"}:
        maximum = config["distillation"]["max_prompts"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("distillation.max_prompts must be a positive integer")
        result = result[:maximum]
    return result


def split_online_rows(
    rows: Sequence[Mapping[str, Any]], fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 2:
        raise ValueError("online RL needs distinct training and validation examples")
    ids = [row["content_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("online data contains duplicate content IDs")
    ordered = sorted(
        rows, key=lambda row: hashlib.sha256(f"{seed}:{row['content_id']}".encode()).digest()
    )
    count = max(1, int(len(ordered) * fraction))
    return [dict(row) for row in ordered[count:]], [dict(row) for row in ordered[:count]]


def self_teacher_prompt(
    row: Mapping[str, Any], method: str, response: str = ""
) -> list[dict[str, str]]:
    prompt = [dict(message) for message in row["prompt"]]
    if method == "sdft":
        context = f"Eine geprüfte Beispielantwort auf diese Aufgabe lautet:\n{row['demonstration']}\nBeantworte die Aufgabe eigenständig."
    elif method == "sdpo":
        _, feedback = verify_response(response, row["task_type"], row["ground_truth"])
        context = f"Ein vorheriger Versuch lautete:\n{response}\nRückmeldung zur Aufgabe:\n{feedback}\nBeantworte die Aufgabe anhand dieser Rückmeldung erneut."
    else:
        raise ValueError("self-teacher context requires sdft or sdpo")
    prompt[-1]["content"] += "\n\n" + context
    return prompt
