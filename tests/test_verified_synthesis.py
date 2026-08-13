from boldt_posttrain.data_pipeline import FastTextLanguageIdentifier, canonical_json, sha256_bytes
from boldt_posttrain.failure_mining import build_synthesis_tasks, synthesize_verified


def test_best_of_four_keeps_only_mechanically_valid():
    body = {
        "schema_version": 1,
        "status": "ok",
        "eval_run_id": "e",
        "categories": {
            name: {
                "count": int(name == "reasoning"),
                "case_ids": [],
                "mean_response_length": 0,
                "priority": int(name == "reasoning"),
            }
            for name in (
                "instruction",
                "format",
                "reasoning",
                "longcontext",
                "language",
                "over_refusal",
                "safety",
                "coding",
            )
        },
        "validator_errors": {},
        "technical_errors": {},
    }
    failure = {**body, "artifact_hash": sha256_bytes(canonical_json(body))}
    tasks = build_synthesis_tasks(failure, [], maximum=1)
    answer = str(tasks[0]["ground_truth"]["value"])
    outputs = ["falsch", answer, "auch falsch", answer + " Wörter"]
    result = synthesize_verified(
        tasks,
        lambda _task, index: outputs[index],
        teacher_ref="teacher@rev",
        best_of_n=4,
        sampling={"temperature": 0.7},
        language_id=FastTextLanguageIdentifier(predictor=lambda _text: ("de", 1.0)),
    )
    assert len(result["sft"]) == 1
    assert result["sft"][0]["response"][0]["content"] == answer
    assert len(result["sft"][0]["candidate_hashes"]) == 4
