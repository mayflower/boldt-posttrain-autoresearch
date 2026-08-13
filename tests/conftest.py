import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def tiny_model_dir(tmp_path):
    """Create a fully local causal LM and chat tokenizer for real trainer tests."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "user": 3,
        "assistant": 4,
        "system": 5,
        "Frage": 6,
        "Antwort": 7,
        "eins": 8,
        "zwei": 9,
        "drei": 10,
        "vier": 11,
        "richtig": 12,
        "falsch": 13,
        "1": 14,
        "2": 15,
        "+": 16,
    }
    raw = tokenizers.Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} "
        "{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] }}"
        "{% endgeneration %}{% else %}{{ message['content'] }}{% endif %} "
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    destination = tmp_path / "tiny-model"
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    return destination
