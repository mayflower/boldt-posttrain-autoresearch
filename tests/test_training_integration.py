from pathlib import Path

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from boldt_posttrain.policy import load_policy
from boldt_posttrain.training import train_adapter
from tests.test_training_preflight import experiment
from tests.tiny_model import build_tiny_model


def test_two_step_sft_and_cpt_save_reload_and_forward(tmp_path: Path, monkeypatch):
    import boldt_posttrain.training as training

    model_path = build_tiny_model(tmp_path / "tiny")
    state_root = tmp_path / "repo/outputs/posttrain"
    monkeypatch.setattr(training, "OUTPUTS", state_root)
    training_args = experiment()
    training_args["max_steps"] = 2
    sft = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Hallo"},
                    {"role": "assistant", "content": "Buch"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Regen"},
                    {"role": "assistant", "content": "Regen fällt"},
                ]
            },
        ]
    )
    sft_result = train_adapter(
        kind="sft",
        model_source=str(model_path),
        revision=None,
        dataset=sft,
        output_root=state_root / "checkpoints",
        policy=load_policy(),
        experiment=training_args,
        target_modules=["c_attn"],
        device="cpu",
        qlora=False,
        allow_checkpoints=True,
        budget_minutes=2,
    )
    assert sft_result["status"] == "succeeded"
    assert sft_result["metrics"]["steps_completed"] == 2

    base = AutoModelForCausalLM.from_pretrained(model_path)
    adapter = PeftModel.from_pretrained(base, sft_result["checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    adapter(**tokenizer("Hallo", return_tensors="pt"))

    cpt_args = {**training_args, "max_steps": 1, "learning_rate": 1e-5, "packing": False}
    cpt = Dataset.from_list([{"text": "Regen fällt aus Wolken ."}])
    cpt_result = train_adapter(
        kind="cpt",
        model_source=str(model_path),
        revision=None,
        dataset=cpt,
        output_root=state_root / "checkpoints",
        policy=load_policy(),
        experiment=cpt_args,
        target_modules=["c_attn"],
        device="cpu",
        qlora=False,
        allow_checkpoints=True,
        budget_minutes=2,
    )
    assert cpt_result["status"] == "succeeded"
    assert (Path(cpt_result["checkpoint"]) / "adapter_model.safetensors").is_file()
