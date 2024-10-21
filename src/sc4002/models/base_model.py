from abc import abstractmethod

import torch


class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def encode(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, *args, **kwargs):
        raise NotImplementedError
