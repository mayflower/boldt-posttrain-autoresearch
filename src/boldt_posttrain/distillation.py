"""Exact teacher license preflight for online distillation."""

from .policy import Policy
from .resolver import ResolvedModelRef
from .secure_compat.data_pipeline import normalize_license


class DistillationError(RuntimeError):
    """A teacher violates the online distillation contract."""


def _teacher_license(
    teacher: ResolvedModelRef, policy: Policy, declared_local_license: str | None
) -> str:
    if teacher.kind == "hub_model":
        from huggingface_hub import HfApi

        info = HfApi().model_info(
            teacher.base_model["repo_id"], revision=teacher.base_model["revision"]
        )
        card = getattr(info, "card_data", None)
        raw = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
    else:
        raw = declared_local_license
        if raw is None:
            raise DistillationError("local teachers require --teacher-license")
    license_id = normalize_license(raw)
    if license_id not in policy.document["data"]["allowed_licenses"]:
        raise DistillationError("teacher license is unknown or forbidden by policy")
    return license_id
