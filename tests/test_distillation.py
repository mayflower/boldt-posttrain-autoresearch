import time

from boldt_posttrain.distillation import cpt_tasks, run_distillation


def test_cpt_is_framed_and_deadline_stops_after_generation():
    tasks = cpt_tasks("Berlin liegt in Deutschland. Es ist eine Stadt.", "cid")
    assert {task["task_type"] for task in tasks} == {
        "summary",
        "fact_extraction",
        "explicit_question",
    }
    registered = []

    def generate(_prompt):
        time.sleep(0.002)
        return "Antwort"

    result = run_distillation(
        prompts=tasks,
        generate=generate,
        filter_output=lambda _p, _o: True,
        train_student=lambda _rows: {"status": "ok"},
        deadline=time.monotonic() + 0.001,
        register_candidate=registered.append,
    )
    assert result["status"] == "budget_exhausted"
    assert not registered
