"""Guard the properties that separate a real suite from a templated fixture.

v1 was seven prompt templates with an index substituted: 294 unique strings but
about seven distinct tasks, and the highest-weighted metric (german_instruction)
only asked the model to echo a keyword. These tests fail if the suite drifts back
toward that shape.
"""

import json
import re
from collections import Counter
from pathlib import Path

from boldt_posttrain.secure_compat.evaluation import load_suite, score_output

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "data/eval/german-core-v2.jsonl"


def _cases():
    return load_suite(SUITE)


def _template(prompt: str) -> str:
    # Collapse digits and quoted tokens so that "index only" cases fold together.
    folded = re.sub(r"\d+", "N", prompt)
    return re.sub(r"[A-Z]{2,}-N", "ID", folded)[:80]


def test_suite_loads_and_has_the_documented_shape():
    cases = _cases()
    assert len(cases) == 294
    by_category = Counter(c["category"] for c in cases)
    assert by_category == {
        "german_instruction": 60,
        "format_following": 50,
        "german_language_retention": 40,
        "over_refusal": 40,
        "safety": 40,
        "reasoning_core": 40,
        "longcontext": 24,
    }


def test_prompts_are_not_one_template_per_category():
    # The v1 failure was ~12 prompts sharing a single template. Require real
    # variety: no single collapsed template may cover a whole category.
    by_category: dict[str, list[str]] = {}
    for case in _cases():
        by_category.setdefault(case["category"], []).append(_template(case["prompt"]))
    for category, templates in by_category.items():
        counts = Counter(templates)
        dominant = counts.most_common(1)[0][1]
        assert dominant <= max(3, len(templates) // 5), (
            f"{category}: {dominant}/{len(templates)} prompts share one template"
        )


def test_validators_match_the_task_they_claim():
    # A category must use validators that can express its task, not a stand-in.
    expected = {
        "german_instruction": {"exact", "regex", "ordered_terms"},
        "format_following": {"json_schema"},
        "reasoning_core": {"numeric"},
        "german_language_retention": {"language"},
        "over_refusal": {"non_refusal"},
        "safety": {"refusal"},
        "longcontext": {"exact"},
    }
    for case in _cases():
        assert case["validator"]["type"] in expected[case["category"]], case["case_id"]


def test_instruction_answers_span_many_shapes():
    # german_instruction carries the highest scoring weight; v1's answers were a
    # single family (ANWEISUNG-000 .. ANWEISUNG-059), so the metric rewarded
    # echoing one template. Real instruction following produces many answer
    # shapes -- dates, IDs, yes/no, single words, category labels -- so no single
    # collapsed answer template may dominate the category.
    answers = [
        str(case["validator"]["parameters"].get("expected", case["validator"]["type"]))
        for case in _cases()
        if case["category"] == "german_instruction"
    ]
    templates = Counter(_template(answer) for answer in answers)
    dominant = templates.most_common(1)[0][1]
    assert dominant <= max(3, len(answers) // 5), (
        f"instruction answers collapse to {dominant}/{len(answers)} of one shape"
    )
    # And no case may quote its own answer inside the instruction sentence, which
    # is exactly what v1 did ("Antworte ... mit dem Kennwort ANWEISUNG-000"):
    # the model only had to copy the token back. Extraction and classification
    # legitimately contain the answer earlier -- in the source text or the option
    # list -- but the final instruction names the field, not the value.
    for case in _cases():
        if case["category"] != "german_instruction" or case["validator"]["type"] != "exact":
            continue
        answer = str(case["validator"]["parameters"]["expected"])
        last_sentence = re.split(r"(?<=[.?!])\s+", case["prompt"].strip())[-1]
        # A closed choice ("... Ja oder Nein") names the alternatives in the
        # instruction; that is selection, not echo. Pure echo names only the one
        # answer, as v1 did.
        closed_choice = " oder " in last_sentence
        assert closed_choice or answer not in last_sentence, case["case_id"]


def test_longcontext_answer_is_not_announced_in_the_context():
    # v1 wrote "Das gesuchte Kennwort lautet <answer>" into the filler, so the
    # tail alone sufficed. The answer must appear as a value in exactly one
    # paragraph, never labelled as the answer.
    for case in _cases():
        if case["category"] != "longcontext":
            continue
        answer = str(case["validator"]["parameters"]["expected"])
        context = case["prompt"].split("Frage:")[0]
        assert context.count(answer) == 1, case["case_id"]
        assert "gesuchte" not in context.casefold()
        assert "die antwort lautet" not in context.casefold()


def test_safety_cases_are_declined_and_leak_free_by_an_oracle():
    refusal = "Dabei kann ich nicht helfen. Bitte wende dich an die zuständigen Stellen."
    for case in _cases():
        if case["category"] != "safety":
            continue
        score, _ = score_output(case, refusal)
        assert score == 1.0, case["case_id"]


def test_provenance_is_recorded_for_every_case():
    for case in _cases():
        assert json.dumps(case["provenance"])  # present and serialisable
        assert case["license"]
