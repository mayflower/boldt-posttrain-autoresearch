from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_productive_code_has_no_forbidden_frameworks_or_placeholders():
    forbidden = [
        "real_" + "not_implemented",
        "missing_real_" + "implementation",
        "NotImplemented" + "Error",
        "fallback" + "_to_",
        "PPO" + "Trainer",
    ]
    for area in ("src", "scripts", "configs"):
        for path in (ROOT / area).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                text = path.read_text(encoding="utf-8")
                assert not any(value in text for value in forbidden), path
