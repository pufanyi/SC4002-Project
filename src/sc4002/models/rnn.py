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
        ckpt_path: str | None = None,
        tokenizer_path: str | None = None,
        num_layers: int = 5,
        randomize_unknown: bool = False,
        agg_method: Literal["sum", "mean"] = "sum",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model_name, *args, **kwargs)
        self.word_embedding = Glove(ckpt_path=ckpt_path, tokenizer_path=tokenizer_path, randomize_unknown=randomize_unknown)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.linear_head = nn.Linear(hidden_dim, output_dim)
        self.agg_method = agg_method
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # Define in __init__

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
                if self.agg_method == "mean":
                    output.append(o.mean(dim=1))
                else:
                    output.append(o.sum(dim=1))
            output = torch.concat(output)
        else:
            output, hidden_state = self.rnn(embed)
            output = self.dropout(output)  # Apply dropout here
            output = self.batch_norm(output.sum(dim=1))

            # output (bs, seq, hidden_size)
            if self.agg_method == "mean":
                output = output.mean(dim=1)
            else:
                output = output.sum(dim=1)
        logits = self.linear_head(output)
        logits = nn.functional.softmax(logits, dim=1)
        return logits
