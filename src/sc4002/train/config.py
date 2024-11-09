from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_type: str = field(default=None)
    input_size: int = field(default=300)
    hidden_size: int = field(default=512)
    download_repo: str = field(default="kcz358/glove")
    tokenizer_path: str = field(
        default="glove.840B.300d/glove.840B.300d.tokenizer.json"
    )
    word_embed_path: str = field(default="glove.840B.300d/glove.840B.300d.safetensors")
    freeze_word_embed: bool = field(default=False)


@dataclass
class DataArguments:
    dataset_name: str = field(default="rotten_tomatoes")
    train_split: str = field(default="train")
    val_split: str = field(default="validation")
    test_split: str = field(default="test")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    sweep_config: Optional[str] = field(default=None)
    sweep_count: Optional[int] = field(default=20)
