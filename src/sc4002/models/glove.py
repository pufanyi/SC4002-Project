import json
from typing import List, Union

import torch
from safetensors.torch import load_file

from .base_model import BaseModel
from .tokenizer import Tokenizer


class Glove(BaseModel):
    def __init__(self, model_name: str = "glove", ckpt_path: str = None, tokenizer_path: str = None, *args, **kwargs) -> None:
        super().__init__(model_name, *args, **kwargs)
        state_dict = load_file(ckpt_path)
        self._vocab_size, self.dim = state_dict["weight"].shape
        self.embedding = torch.nn.Embedding.from_pretrained(state_dict["weight"])
        self.tokenizer = Tokenizer(tokenizer_path)

    @property
    def vocab_size(self):
        return self._vocab_size

    def forward(self, inputs: str):
        input_ids = self.tokenizer.encode(inputs, return_tensor="pt")
        embeddings = self.embedding(input_ids)
        return embeddings

    def device(self):
        return self.embedding.weight.device
