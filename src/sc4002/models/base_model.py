from abc import abstractmethod

import torch


class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    @abstractmethod
    def device():
        raise NotImplementedError


# Add any function or attribute that are common to all models here
