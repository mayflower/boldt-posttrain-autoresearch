import pytest

from boldt_posttrain.data_pipeline import canonical_json, sha256_bytes
from boldt_posttrain.failure_mining import synthesize_verified


def test_synthesis_rejects_visible_eval_text():
    corpus_body = {"entries": [{"canonical": "geheime eval frage", "sha256": "x"}]}
    corpus = {**corpus_body, "artifact_hash": sha256_bytes(canonical_json(corpus_body))}
    task = {
        "task_type": "numeric",
        "prompt": "Geheime Eval Frage",
        "ground_truth": {"value": 1},
        "source_content_id": "procedural",
    }
    with pytest.raises(ValueError):
        synthesize_verified(
            [task],
            lambda _task, _index: "1",
            teacher_ref="t",
            best_of_n=1,
            sampling={},
            eval_corpus=corpus,
        )
