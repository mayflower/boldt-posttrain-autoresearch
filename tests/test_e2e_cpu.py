"""Real optional-stack CPU path: local model.generate plus an lm-eval subprocess."""

import json
import shutil

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("lm_eval")

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402

from boldt_posttrain.evaluation import (  # noqa: E402
    TransformersGenerator,
    evaluate_cases,
    finalize_summary,
    make_summary,
    run_lm_eval,
)
from boldt_posttrain.scoring import score_run  # noqa: E402
from boldt_posttrain.training import make_peft_config  # noqa: E402


def test_real_local_generation_and_lm_eval_subprocess(tmp_path):
    executable = shutil.which("lm_eval")
    if not executable:
        pytest.skip("lm_eval console executable is not installed")
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "user": 3,
        "assistant": 4,
        "Frage": 5,
        "A": 6,
        "B": 7,
    }
    raw = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=len(vocabulary),
            n_layer=1,
            n_head=1,
            n_embd=16,
            n_positions=64,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    generator = TransformersGenerator(model, tokenizer, "cpu", max_new_tokens=2, context_length=64)
    local = evaluate_cases(
        [{"case_id": "local-1", "category": "instruction", "prompt": "Frage"}], generator
    )
    assert local["technical_error_count"] == 0

    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    rows = task_dir / "rows.jsonl"
    rows.write_text(json.dumps({"question": "Frage", "choices": [" A", " B"], "answer": 0}) + "\n")
    (task_dir / "tiny_local.yaml").write_text(
        "task: tiny_local\n"
        "dataset_path: json\n"
        f"dataset_kwargs:\n  data_files:\n    test: {rows}\n"
        "test_split: test\noutput_type: multiple_choice\n"
        "doc_to_text: '{{question}}'\ndoc_to_choice: choices\ndoc_to_target: answer\n"
        "metric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n"
    )
    result = run_lm_eval(
        model=str(model_dir),
        tasks=["tiny_local"],
        output_path=tmp_path / "lm-results",
        device="cpu",
        limit=1,
        executable=executable,
        include_path=task_dir,
        timeout_seconds=120,
    )
    assert "tiny_local" in result["metrics"]
    peft = pytest.importorskip("peft")
    adapter_model = peft.get_peft_model(
        transformers.AutoModelForCausalLM.from_pretrained(model_dir),
        make_peft_config(
            {
                "lora_r": 4,
                "lora_alpha": 8,
                "target_modules": ["c_attn"],
                "lora_init": "default",
            }
        ),
    )
    adapter = tmp_path / "adapter"
    adapter_model.save_pretrained(adapter)
    adapter_result = run_lm_eval(
        model=str(model_dir),
        peft_adapter=str(adapter),
        tasks=["tiny_local"],
        output_path=tmp_path / "lm-adapter-results",
        device="cpu",
        limit=1,
        executable=executable,
        include_path=task_dir,
        timeout_seconds=120,
    )
    assert "tiny_local" in adapter_result["metrics"]
    metrics = {
        "german_instruction": local["accuracy"],
        "format_following": 1.0,
        "reasoning_core": 1.0,
        "safety": 1.0,
        "german_language_retention": 1.0,
        "english_bleed_rate": 0.0,
        "empty_output_rate": 0.0,
        "refusal_rate": local["refusal_rate"],
        "over_refusal_rate": local["over_refusal_rate"],
        "lm_eval": result["metrics"],
        "leakage": {"status": "clean", "hits": 0},
        "license": {"status": "apache-2.0", "usable": True},
    }
    summary = finalize_summary(
        make_summary(
            profile="dev",
            suite_hash="local-suite",
            decontamination_hash="local-decontamination",
            result=local,
            metrics=metrics,
            model=str(model_dir),
        )
    )
    baseline = json.loads(json.dumps(summary))
    baseline["metrics"]["german_instruction"] = 0.5
    scored = score_run(summary, baseline)
    assert isinstance(scored["score"], float)
    assert summary["artifact_hash"]
