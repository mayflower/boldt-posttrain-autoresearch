from types import SimpleNamespace

import pytest

from boldt_posttrain.distillation import DistillationError, _teacher_license
from boldt_posttrain.policy import load_policy


def test_online_local_teacher_requires_an_explicit_allowed_license():
    teacher = SimpleNamespace(kind="local_full_checkpoint")
    policy = load_policy()
    with pytest.raises(DistillationError, match="teacher-license"):
        _teacher_license(teacher, policy, None)
    with pytest.raises(DistillationError, match="forbidden"):
        _teacher_license(teacher, policy, "proprietary")
    assert _teacher_license(teacher, policy, "MIT") == "MIT"
