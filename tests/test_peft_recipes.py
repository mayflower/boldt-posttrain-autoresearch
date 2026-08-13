import pytest

from boldt_posttrain.training import load_trainable_adapter, make_peft_config

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")


@pytest.mark.parametrize(
    ("lora_init", "use_rslora"),
    [("default", False), ("default", True), ("pissa_niter_4", False)],
)
def test_each_peft_recipe_takes_optimizer_step_saves_and_reloads(
    tmp_path, tiny_model_dir, lora_init, use_rslora
):
    model = transformers.AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    config = make_peft_config(
        {
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
            "lora_init": lora_init,
            "use_rslora": use_rslora,
        }
    )
    model = peft.get_peft_model(model, config)
    inputs = torch.tensor([[6, 14, 16, 14, 15]])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    loss = model(input_ids=inputs, labels=inputs).loss
    loss.backward()
    optimizer.step()
    assert optimizer.state

    adapter = tmp_path / f"adapter-{lora_init}-{use_rslora}"
    model.save_pretrained(adapter)
    reloaded = load_trainable_adapter(
        transformers.AutoModelForCausalLM.from_pretrained(tiny_model_dir), adapter
    )
    with torch.inference_mode():
        logits = reloaded(input_ids=inputs).logits
    assert torch.isfinite(logits).all()
