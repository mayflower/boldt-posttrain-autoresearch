import json
import time

from boldt_posttrain.data_pipeline import build_decontamination_corpus, materialize_streaming
from boldt_posttrain.preference import DeadlineCallback
from boldt_posttrain.training import _train_real


def test_sft_uses_validation_and_publishes_independent_best_adapter(tmp_path, tiny_model_dir):
    from transformers import AutoTokenizer

    data_dir = tmp_path / "data"
    corpus = build_decontamination_corpus(
        [], tmp_path / "corpus.json", sources=[], policy_hash="policy"
    )
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    source = {
        "dataset": "fixture",
        "revision": "one",
        "schema": "sft",
        "source_group": "general",
        "license": "apache-2.0",
        "rows": [
            {"instruction": "Frage eins", "answer": "Antwort eins"},
            {"instruction": "Frage zwei", "answer": "Antwort zwei"},
            {"instruction": "Frage drei", "answer": "Antwort drei"},
            {"instruction": "Frage vier", "answer": "Antwort vier"},
        ],
    }
    materialize_streaming(
        [source],
        data_dir,
        validation_fraction={"sft": 0.25},
        tokenizer=tokenizer,
        context_length=32,
        decontamination_corpus=corpus,
        policy_hash="policy",
    )
    cfg = {
        "policy_hash": "policy",
        "data": {"validation_fraction": {"sft": 0.25}},
        "training": {
            "base_model": str(tiny_model_dir),
            "method": "lora",
            "seed": 17,
            "context_length": 32,
            "learning_rate": 1e-3,
            "max_steps": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "packing": False,
            "assistant_only_loss": True,
            "use_liger_kernel": False,
            "use_rslora": False,
        },
        "cpt": {
            "train_embeddings": False,
            "train_lm_head": False,
            "module_learning_rate_multiplier": 0.1,
        },
    }
    out = tmp_path / "run"
    metrics = _train_real(
        cfg=cfg,
        kind="specialist",
        data_dir=data_dir,
        out_dir=out,
        deadline=time.monotonic() + 180,
        device="cpu",
    )
    assert metrics["status"] == "succeeded"
    assert metrics["steps"] == 1
    assert metrics["validation_examples"] == 1
    assert metrics["validation_evaluations"] >= 1
    assert metrics["best_eval_loss"] is not None
    assert metrics["assistant_supervision"]["enabled"] is True
    assert (out / "adapter/adapter_config.json").is_file()
    assert not (out / "trainer").exists()
    assert json.loads((data_dir / "manifest.json").read_text())["split_statistics"]["sft"] == {
        "train": 3,
        "validation": 1,
    }


def test_cpt_saved_modules_survive_training_save_and_reload(tmp_path, tiny_model_dir):
    from transformers import AutoTokenizer

    data_dir = tmp_path / "cpt-data"
    corpus = build_decontamination_corpus(
        [], tmp_path / "cpt-corpus.json", sources=[], policy_hash="policy"
    )
    materialize_streaming(
        [
            {
                "dataset": "cpt-fixture",
                "revision": "one",
                "schema": "cpt",
                "source_group": "raw",
                "license": "apache-2.0",
                "rows": [
                    {"text": "Frage eins Antwort eins"},
                    {"text": "Frage zwei Antwort zwei"},
                    {"text": "Frage drei Antwort drei"},
                    {"text": "Frage vier Antwort vier"},
                ],
            }
        ],
        data_dir,
        validation_fraction={"cpt": 0.25},
        tokenizer=AutoTokenizer.from_pretrained(tiny_model_dir),
        context_length=32,
        decontamination_corpus=corpus,
        policy_hash="policy",
    )
    cfg = {
        "policy_hash": "policy",
        "data": {"validation_fraction": {"cpt": 0.25}},
        "training": {
            "base_model": str(tiny_model_dir),
            "method": "lora",
            "seed": 17,
            "context_length": 32,
            "learning_rate": 1e-3,
            "max_steps": 1,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "packing": False,
            "assistant_only_loss": True,
            "use_liger_kernel": False,
            "use_rslora": False,
        },
        "cpt": {
            "train_embeddings": True,
            "train_lm_head": True,
            "module_learning_rate_multiplier": 0.1,
        },
    }
    out = tmp_path / "cpt-run"
    metrics = _train_real(
        cfg=cfg,
        kind="cpt",
        data_dir=data_dir,
        out_dir=out,
        deadline=time.monotonic() + 180,
        device="cpu",
    )
    assert set(metrics["cpt_modules_to_save"]) == {"embed_tokens", "lm_head"}
    assert metrics["cpt_module_learning_rates"] == [1e-3, 1e-4]
    assert (out / "adapter/adapter_model.safetensors").is_file()


def test_controlled_budget_stop_still_validates_and_publishes(
    tmp_path, tiny_model_dir, monkeypatch
):
    from transformers import AutoTokenizer

    def stop_after_first_step(self, _args, _state, control, **_kwargs):
        self.stop_reason = "budget_limit"
        control.should_training_stop = True
        return control

    monkeypatch.setattr(DeadlineCallback, "on_step_end", stop_after_first_step)
    data_dir = tmp_path / "budget-data"
    corpus = build_decontamination_corpus(
        [], tmp_path / "budget-corpus.json", sources=[], policy_hash="policy"
    )
    materialize_streaming(
        [
            {
                "dataset": "budget-fixture",
                "revision": "one",
                "schema": "sft",
                "source_group": "general",
                "license": "apache-2.0",
                "rows": [
                    {"instruction": f"Frage {index}", "answer": f"Antwort {index}"}
                    for index in range(6)
                ],
            }
        ],
        data_dir,
        validation_fraction={"sft": 0.25},
        tokenizer=AutoTokenizer.from_pretrained(tiny_model_dir),
        context_length=32,
        decontamination_corpus=corpus,
        policy_hash="policy",
    )
    cfg = {
        "policy_hash": "policy",
        "data": {"validation_fraction": {"sft": 0.25}},
        "training": {
            "base_model": str(tiny_model_dir),
            "method": "lora",
            "seed": 17,
            "context_length": 32,
            "learning_rate": 1e-3,
            "max_steps": 3,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "packing": False,
            "assistant_only_loss": True,
            "use_liger_kernel": False,
            "use_rslora": False,
        },
        "cpt": {
            "train_embeddings": False,
            "train_lm_head": False,
            "module_learning_rate_multiplier": 0.1,
        },
    }
    out = tmp_path / "budget-run"
    metrics = _train_real(
        cfg=cfg,
        kind="specialist",
        data_dir=data_dir,
        out_dir=out,
        deadline=time.monotonic() + 180,
        device="cpu",
    )
    assert metrics["status"] == "succeeded"
    assert metrics["stop_reason"] == "budget_limit"
    assert metrics["steps"] == 1
    assert metrics["validation_evaluations"] >= 1
    assert metrics["best_eval_loss"] is not None
    assert (out / "adapter/adapter_model.safetensors").is_file()
    assert not (out / "trainer").exists()
