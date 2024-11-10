from typing import List, Literal

import torch
import torch.nn as nn
from torchtyping import TensorType

from .base_model import BaseModel
from .glove import Glove
class CNN(BaseModel):
    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 512,
        output_dim: int = 2,
        model_name: str = "cnn",
        ckpt_path: str | None = None,
        tokenizer_path: str | None = None,
        kernel_size: int = 3,
        num_layers: int = 2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model_name, *args, **kwargs)
        self.word_embedding = Glove(ckpt_path=ckpt_path, tokenizer_path=tokenizer_path)

        # Define CNN layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=input_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # To maintain sequence length
            )
            for i in range(num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.linear_head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        inputs: List[str] = None,
        input_ids: TensorType["bs", "seq_len"] = None,
        masks: TensorType["bs", "seq_len"] = None,
        **kwargs,
    ):
        if inputs is None and input_ids is None:
            assert False, "input and input_ids cannot both be None"
        if input_ids is None:
            embed = self.word_embedding.forward(inputs)  # (bs, seq_len, input_dim)
        elif inputs is None:
            embed = self.word_embedding.forward(input_ids=input_ids)  # (bs, seq_len, input_dim)

        # Adjust dimensions for Conv1d: (batch_size, input_dim, seq_len)
        embed = embed.permute(0, 2, 1)

        if masks is not None:
            embed = embed * masks.unsqueeze(1).float()

        # Apply convolutional layers
        output = embed
        for conv in self.conv_layers:
            output = torch.relu(conv(output))

        # Global average pooling to reduce sequence dimension
        output = self.pool(output).squeeze(-1)  # (bs, hidden_dim)

        logits = self.linear_head(output)  # (bs, output_dim)
        logits = nn.functional.softmax(logits, dim=1)
        return logits
