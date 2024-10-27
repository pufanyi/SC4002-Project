from dataclasses import dataclass, field
from typing import Dict, Sequence

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import HfArgumentParser, Trainer, TrainingArguments

from sc4002.models import RNN, Tokenizer
from sc4002.train.trainer import CustomTrainer


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


@dataclass
class DataCollator:
    tokenizer: Tokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.pad_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.pad_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]):
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "label")
        )
        input_ids = [torch.tensor(input_id).squeeze(0) for input_id in input_ids]
        labels = torch.tensor(labels)
        input_ids = self.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_id
        )
        return dict(input_ids=input_ids, labels=labels)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer_path = hf_hub_download(
        repo_id=model_args.download_repo, filename=model_args.tokenizer_path
    )
    checkpoint_path = hf_hub_download(
        repo_id=model_args.download_repo, filename=model_args.word_embed_path
    )

    if model_args.model_type.lower() == "rnn":
        model = RNN(
            input_dim=model_args.input_size,
            hidden_dim=model_args.hidden_size,
            tokenizer_path=tokenizer_path,
            ckpt_path=checkpoint_path,
        )
    tokenizer = model.word_embedding.tokenizer

    if model_args.freeze_word_embed:
        for p in model.word_embedding.parameters():
            p.requires_grad = False

    # Preprocess into input_ids
    def preprocess(example):
        example["input_ids"] = tokenizer.encode([example["text"]])
        return example

    dataset = load_dataset(data_args.dataset_name)
    train_dataset = dataset[data_args.train_split]
    train_dataset = train_dataset.map(preprocess)
    eval_dataset = dataset[data_args.val_split]
    eval_dataset = train_dataset.map(preprocess)
    collator = DataCollator(tokenizer=tokenizer)
    trainer = CustomTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
