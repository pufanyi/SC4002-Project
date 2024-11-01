from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from datasets import Dataset

from sc4002.models import RNN, Tokenizer
from sc4002.train.config import ModelArguments


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
        masks = input_ids.ne(self.tokenizer.pad_id)
        return dict(input_ids=input_ids, labels=labels, masks=masks)


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: Tokenizer,
    train_split: str,
    val_split: str,
    test_split: str,
):
    def preprocess(example):
        example["input_ids"] = tokenizer.encode([example["text"]])
        return example

    train_dataset = dataset[train_split]
    train_dataset = train_dataset.map(preprocess)
    eval_dataset = dataset[val_split]
    eval_dataset = eval_dataset.map(preprocess)
    test_dataset = dataset[test_split]
    test_dataset = test_dataset.map(preprocess)


def get_model(
    model_args: ModelArguments, tokenizer_path: str = None, checkpoint_path: str = None
):
    if model_args.model_type.lower() == "rnn":
        model = RNN(
            input_dim=model_args.input_size,
            hidden_dim=model_args.hidden_size,
            tokenizer_path=tokenizer_path,
            ckpt_path=checkpoint_path,
        )
    return model
