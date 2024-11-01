from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_type: str = field(default=None)
    input_size: int = field(default=300)
    hidden_size: int = field(default=512)
    download_repo: str = field(default="kcz358/glove")
    tokenizer_path: str = field(default="glove.6B/glove.6B.300d.tokenizer.json")
    word_embed_path: str = field(default="glove.6B/glove.6B.300d.safetensors")
    freeze_word_embed: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_name: str = field(default="rotten_tomatoes")
    train_split: str = field(default="train")
    val_split: str = field(default="validation")
    test_split: str = field(default="test")
