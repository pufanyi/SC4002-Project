from typing import List, Literal

import torch
import torch.nn as nn
from torchtyping import TensorType

from .base_model import BaseModel
from .glove import Glove


class RNN(BaseModel):
    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 512,
        output_dim: int = 2,
        model_name: str = "rnn",
        ckpt_path: str = None,
        tokenizer_path: str = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model_name, *args, **kwargs)
        self.word_embedding = Glove(ckpt_path=ckpt_path, tokenizer_path=tokenizer_path)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim)
        self.linear_head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        inputs: List[str] = None,
        input_ids: TensorType["bs", "seq_len"] = None,
        masks: TensorType["bs", "seq_len"] = None,
        **kwargs,
    ):
        if inputs is None and input_ids is None:
            assert False, "input and input_ids can not both be None"
        if input_ids is None:
            embed = self.word_embedding.forward(inputs)
        elif inputs is None:
            embed = self.word_embedding.forward(input_ids=input_ids)

        if masks is not None:
            output = []
            for em, mask in zip(embed, masks):
                o, hid = self.rnn(em[mask].unsqueeze(0))
                output.append(o.sum(dim=1))
            output = torch.concat(output)
        else:
            output, hidden_state = self.rnn(embed)
            # output (bs, seq, hidden_size)
            output = output.sum(dim=1)
        logits = self.linear_head(output)
        logits = nn.functional.softmax(logits, dim=1)
        return logits
