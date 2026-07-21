from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_productive_paths_have_no_forbidden_tokens():
    forbidden = (
        "real_" + "not_implemented",
        "missing_real_" + "implementation",
        "Not" + "ImplementedError",
        "|" + "| true",
    )
    violations: list[str] = []
    for directory in ("src", "scripts", ".claude"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                for token in forbidden:
                    if token in text:
                        violations.append(f"{path.relative_to(ROOT)}: {token}")
    assert not violations, "\n".join(violations)
