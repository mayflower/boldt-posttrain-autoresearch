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

    # Mirrors make_pissa_initial_callback: the conversion base must come from
    # this very model, before the first optimizer step. pissa_niter_4 uses a
    # randomized SVD, so re-initialising would yield a different decomposition.
    conversion = {}
    if lora_init == "pissa_niter_4":
        initial = tmp_path / f"initial-{lora_init}-{use_rslora}"
        model.save_pretrained(initial)
        conversion["path_initial_model_for_weight_conversion"] = str(initial)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    loss = model(input_ids=inputs, labels=inputs).loss
    loss.backward()
    optimizer.step()
    assert optimizer.state

    model.eval()
    with torch.inference_mode():
        trained_logits = model(input_ids=inputs).logits.clone()

    # PiSSA rewrites the base weights at initialisation, so the adapter only
    # reloads to the trained model if PEFT converts it back to a plain LoRA delta
    # using the pre-training adapter. Without that conversion this round trip
    # drifts by max|delta| ~0.04 here while still producing finite logits --
    # asserting finiteness alone cannot tell the two apart.
    adapter = tmp_path / f"adapter-{lora_init}-{use_rslora}"
    model.save_pretrained(adapter, **conversion)
    reloaded = load_trainable_adapter(
        transformers.AutoModelForCausalLM.from_pretrained(tiny_model_dir), adapter
    )
    reloaded.eval()
    with torch.inference_mode():
        logits = reloaded(input_ids=inputs).logits
    assert torch.isfinite(logits).all()
    assert torch.allclose(trained_logits, logits, atol=1e-5), (
        f"{lora_init} adapter does not reload to the trained model: "
        f"max|delta| = {(trained_logits - logits).abs().max().item():.6g}"
    )
