"""The start-of-session guide names the missing step, why, and what comes after."""

import json
import sys
import subprocess
from pathlib import Path

from boldt_posttrain.guide import readiness, render_guide
from boldt_posttrain.policy import load_policy

ROOT = Path(__file__).resolve().parents[1]


def test_empty_project_starts_with_data(tmp_path):
    outputs = tmp_path / "outputs/posttrain"
    state = readiness(load_policy(), outputs_root=outputs, repository_root=tmp_path)
    assert state["loop_ready"] is False and state["next_command"] == "/pt-data real"
    guide = render_guide(load_policy(), outputs_root=outputs, repository_root=tmp_path)
    assert "Next: /pt-data real" in guide
    assert "it does not exist yet" in guide
    assert "After it: /pt-baseline real, then /pt-run 1 real." in guide
    assert "0 so far, no champion yet" in guide


def test_guide_imports_no_ml_stack():
    code = (
        "import sys, boldt_posttrain.guide, boldt_posttrain.cli; "
        "print('torch' in sys.modules or 'transformers' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.stdout.strip() == "False"


def test_welcome_hook_emits_the_guide_as_system_message():
    result = subprocess.run(
        ["bash", str(ROOT / ".claude/hooks/pt_welcome.sh")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    message = json.loads(result.stdout)["systemMessage"]
    assert "Where you are" in message and "Next: " in message
