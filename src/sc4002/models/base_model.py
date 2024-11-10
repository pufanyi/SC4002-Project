from abc import abstractmethod
from typing import Iterable

import torch


class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    @abstractmethod
    def forward(**kwargs):
        raise NotImplementedError

    def add_train_vocab(self, corpus: Iterable[str]):
        if hasattr(self, "word_embedding"):
            self.word_embedding.add_train_vocab(corpus)
        else:
            raise NotImplementedError("Model does not support training vocabulary")
