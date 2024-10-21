import json
from typing import List, Union

import torch
from safetensors.torch import load_file

from .base_model import BaseModel


class Glove(BaseModel):
    def __init__(self, model_name: str = "glove", ckpt_path: str = None, tokenizer_path: str = None, *args, **kwargs) -> None:
        super().__init__(model_name, *args, **kwargs)
        state_dict = load_file(ckpt_path)
        self._vocab_size, self.dim = state_dict["weight"].shape
        self.embedding = torch.nn.Embedding.from_pretrained(state_dict["weight"])
        self.tokenizer = GloveTokenizer(tokenizer_path)

    @property
    def vocab_size(self):
        return self._vocab_size

    def forward(self, inputs: str):
        input_ids = self.tokenizer.encode(inputs)
        embeddings = self.embedding(torch.tensor([input_ids]))
        return embeddings

    def device(self):
        return self.embedding.weight.device


class GloveTokenizer:
    def __init__(self, tokenizer_path) -> None:
        with open(tokenizer_path, "r") as f:
            self.tokenizer_dict = json.load(f)
        self.ids_to_tokens = {v: k for k, v in self.tokenizer_dict.items()}

    def encode(self, inputs: str):
        if inputs in self.tokenizer_dict:
            return self.tokenizer_dict[inputs]
        else:
            return "<|UNK|>"

    def decode(self, input_ids: Union[List[int], torch.Tensor]):
        for ids in input_ids:
            return self.ids_to_tokens[ids]


if __name__ == "__main__":
    model = Glove(model_name="glove", ckpt_path="./checkpoints/glove.6B/glove.6B.50d.safetensors", tokenizer_path="./checkpoints/glove.6B/glove.6B.50d.tokenizer.json")
    print(model.vocab_size)
    print(model.forward("hello"))
