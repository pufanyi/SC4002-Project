import json
from typing import Any, List, Union

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
        # Add UNK token embedding
        self.add_embedding()
        # Add pad token embedding, does not contribute
        self.add_embedding(padding=True)

    @property
    def vocab_size(self):
        return self._vocab_size

    def forward(self, inputs: List[str], **kwargs):
        input_ids = self.tokenizer.encode(inputs, return_tensor="pt")
        embeddings = self.embedding(input_ids)
        return embeddings

    def device(self):
        return self.embedding.weight.device

    def add_embedding(self, padding=False):
        params = torch.zeros(1, self.dim)
        self.embedding.weight.data = torch.concat([self.embedding.weight.data, params])
        if padding:
            self.embedding.padding_idx = self.vocab_size
        self.embedding.num_embeddings += 1
        self._vocab_size += 1
