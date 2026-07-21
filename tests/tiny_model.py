from pathlib import Path


def build_tiny_model(path: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    path.mkdir(parents=True, exist_ok=True)
    vocabulary = {
        "<unk>": 0,
        "<pad>": 1,
        "<eos>": 2,
        "<user>": 3,
        "<assistant>": 4,
        "Hallo": 5,
        "Buch": 6,
        "Regen": 7,
        "fällt": 8,
        ".": 9,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
        additional_special_tokens=["<user>", "<assistant>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}<eos>"
        "{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    tokenizer.save_pretrained(path)
    (path / "chat_template.jinja").write_text(tokenizer.chat_template)
    config = GPT2Config(
        vocab_size=len(vocabulary),
        n_positions=256,
        n_ctx=256,
        n_embd=16,
        n_layer=1,
        n_head=1,
        bos_token_id=None,
        eos_token_id=2,
        pad_token_id=1,
        tie_word_embeddings=False,
    )
    model = GPT2LMHeadModel(config)
    model.transformer.ln_f.weight.data.zero_()
    model.transformer.ln_f.bias.data.fill_(1.0)
    model.lm_head.weight.data.zero_()
    model.lm_head.weight.data[vocabulary["Buch"]].fill_(1.0)
    model.save_pretrained(path)
    return path
