import pytest

from boldt_posttrain.data_pipeline import FastTextLanguageIdentifier
from boldt_posttrain.rewards import (
    concise_length_reward,
    exact_reward,
    german_language_reward,
    json_schema_reward,
    non_refusal_reward,
    numeric_reward,
    ordered_terms_reward,
    total_reward,
)


def test_positive_negative_and_not_applicable_rewards():
    assert exact_reward("ja", task_type="exact", ground_truth={"value": "ja"}) == 1
    assert exact_reward("nein", task_type="exact", ground_truth={"value": "ja"}) == 0
    assert exact_reward("1", task_type="numeric", ground_truth={"value": 1}) is None
    assert numeric_reward("42", task_type="numeric", ground_truth={"value": 42}) == 1
    assert (
        ordered_terms_reward("a b", task_type="ordered_terms", ground_truth={"terms": ["b", "a"]})
        == 0
    )
    language = FastTextLanguageIdentifier(predictor=lambda _text: ("de", 0.99))
    assert (
        german_language_reward(
            "Sonne",
            task_type="language",
            ground_truth={"required_terms": ["Sonne"]},
            language_id=language,
        )
        == 1
    )


@pytest.mark.parametrize(
    ("function", "task_type", "truth", "positive", "negative", "other_type"),
    [
        (exact_reward, "exact", {"value": "ja"}, "ja", "nein", "numeric"),
        (numeric_reward, "numeric", {"value": 2}, "2", "3", "exact"),
        (
            json_schema_reward,
            "json_schema",
            {"schema": {"type": "integer"}, "value": 2},
            "2",
            '"2"',
            "exact",
        ),
        (
            ordered_terms_reward,
            "ordered_terms",
            {"terms": ["a", "b"]},
            "a b",
            "b a",
            "exact",
        ),
        (
            non_refusal_reward,
            "non_refusal",
            {"required_terms": ["Sonne"]},
            "Sonne",
            "Ich kann dabei nicht helfen",
            "exact",
        ),
        (
            concise_length_reward,
            "exact",
            {"minimum_words": 1, "maximum_words": 2},
            "kurz",
            "viel zu viele Wörter",
            "unsupported",
        ),
    ],
)
def test_each_non_language_reward_positive_negative_not_applicable(
    function, task_type, truth, positive, negative, other_type
):
    assert function(positive, task_type=task_type, ground_truth=truth) == 1.0
    assert function(negative, task_type=task_type, ground_truth=truth) == 0.0
    assert function(positive, task_type=other_type, ground_truth=truth) is None


def test_language_reward_positive_negative_not_applicable():
    german = FastTextLanguageIdentifier(predictor=lambda _text: ("de", 0.99))
    english = FastTextLanguageIdentifier(predictor=lambda _text: ("en", 0.99))
    truth = {"required_terms": ["Sonne"]}
    assert (
        german_language_reward(
            "Sonne", task_type="language", ground_truth=truth, language_id=german
        )
        == 1
    )
    assert (
        german_language_reward(
            "Sonne", task_type="language", ground_truth=truth, language_id=english
        )
        == 0
    )
    assert (
        german_language_reward("Sonne", task_type="exact", ground_truth=truth, language_id=german)
        is None
    )


def test_bonus_cannot_replace_wrong_answer_and_nonfinite_fails():
    score = total_reward(
        "falsch",
        task_type="exact",
        ground_truth={"value": "richtig"},
        weights={"exact": 1, "concise_length": 10},
        clamp=[-1, 2],
        language_id=None,
    )
    assert score == 0
    with pytest.raises(ValueError):
        total_reward(
            "richtig",
            task_type="exact",
            ground_truth={"value": "richtig"},
            weights={"exact": float("nan")},
            clamp=[-1, 2],
        )
