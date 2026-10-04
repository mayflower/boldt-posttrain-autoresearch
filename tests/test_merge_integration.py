"""The locked mergekit executes every merge configuration the merge lever can emit."""

import shutil
import subprocess

import pytest

from boldt_posttrain.merge import MergeError, merge_configuration


def test_merge_configuration_rejects_unsupported_shapes(tmp_path):
    pytest.importorskip("mergekit.config")
    with pytest.raises(MergeError, match="two or more"):
        merge_configuration("linear", [tmp_path / "one"], dtype="float32", parameters={})
    with pytest.raises(MergeError, match="exactly two"):
        merge_configuration(
            "slerp", [tmp_path / name for name in "abc"], dtype="float32", parameters={}
        )
    with pytest.raises(MergeError, match="seed base model"):
        merge_configuration(
            "ties", [tmp_path / "a", tmp_path / "b"], dtype="float32", parameters={}
        )


@pytest.mark.parametrize("method", ["linear", "slerp", "ties", "dare_ties"])
def test_locked_mergekit_executes_real_tiny_merge(tmp_path, tiny_model_dir, method):
    executable = shutil.which("mergekit-yaml")
    if not executable:
        pytest.skip("mergekit console executable is not installed")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    models = []
    for index in range(2):
        model = transformers.AutoModelForCausalLM.from_pretrained(tiny_model_dir)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.01 * (index + 1))
        path = tmp_path / f"model-{index}"
        model.save_pretrained(path)
        transformers.AutoTokenizer.from_pretrained(tiny_model_dir).save_pretrained(path)
        models.append(path)
    config_path = tmp_path / "mergekit.yaml"
    config_path.write_text(
        merge_configuration(
            method, models, dtype="float32", parameters={}, base_model=str(tiny_model_dir)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "merged"
    completed = subprocess.run(
        [executable, str(config_path), str(output), "--device", "cpu", "--quiet"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(output)
    assert reloaded.config.vocab_size == 17
