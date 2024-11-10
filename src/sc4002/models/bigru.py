from typing import List
import torch
import torch.nn as nn
from torchtyping import TensorType

from .base_model import BaseModel
from .glove import Glove


class BidirectionalGRU(BaseModel):
    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 512,
        output_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_residual: bool = True,
        bidirectional_merge: str = "concat",
        model_name: str = "enhanced_bigru",
        ckpt_path: str | None = None,
        tokenizer_path: str | None = None,
        randomize_unknown: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the Enhanced Bidirectional GRU model with residual connections
        and improved bidirectional merging strategies.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            output_dim: Output dimension
            num_layers: Number of stacked GRU layers
            dropout: Dropout rate
            use_residual: Whether to use residual connections between layers
            bidirectional_merge: Strategy to merge bidirectional outputs ('concat', 'sum', 'avg')
            model_name: Model identifier
            ckpt_path: Checkpoint path
            tokenizer_path: Tokenizer path
        """
        super().__init__(model_name, *args, **kwargs)

        self.word_embedding = Glove(ckpt_path=ckpt_path, tokenizer_path=tokenizer_path, randomize_unknown=randomize_unknown)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.bidirectional_merge = bidirectional_merge

        # Layer normalization for input
        self.layer_norm_input = nn.LayerNorm(input_dim)

        # Stack of bidirectional GRU layers
        self.gru_layers = nn.ModuleList(
            [nn.GRU(input_size=input_dim if i == 0 else hidden_dim * 2, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True, dropout=0) for i in range(num_layers)]  # We'll handle dropout manually
        )

        # Layer normalization after each GRU layer
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim * 2) for _ in range(num_layers)])

        # Dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

        # Output dimension adjustment based on merging strategy
        output_size = hidden_dim * 2 if bidirectional_merge in ["concat", "attention"] else hidden_dim

        # Attention layer for merging bidirectional states
        if bidirectional_merge == "attention":
            self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

        # Final classification layers
        self.classifier = nn.Sequential(nn.Linear(output_size, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def _merge_bidirectional(self, forward_state, backward_state):
        """
        Merge forward and backward states using the specified strategy.
        """
        if self.bidirectional_merge == "concat":
            return torch.cat([forward_state, backward_state], dim=-1)
        elif self.bidirectional_merge == "sum":
            return forward_state + backward_state
        elif self.bidirectional_merge == "avg":
            return (forward_state + backward_state) / 2
        elif self.bidirectional_merge == "attention":
            combined = torch.cat([forward_state, backward_state], dim=-1)
            attention_weights = torch.softmax(self.attention(combined), dim=1)
            return combined * attention_weights
        else:
            raise ValueError(f"Unknown merge strategy: {self.bidirectional_merge}")

    def _process_sequence(self, x, masks=None):
        """
        Process a sequence through all GRU layers with residual connections.
        """
        batch_size = x.size(0)

        # Initial layer normalization
        x = self.layer_norm_input(x)

        # Store all layer outputs for residual connections
        layer_outputs = []

        # Process through each GRU layer
        current_input = x
        for i, (gru, layer_norm, dropout) in enumerate(zip(self.gru_layers, self.layer_norms, self.dropouts)):
            # GRU forward pass
            gru_out, hidden = gru(current_input)

            # Split bidirectional outputs
            forward_out = gru_out[:, :, : self.hidden_dim]
            backward_out = gru_out[:, :, self.hidden_dim :]

            # Merge bidirectional states
            merged = self._merge_bidirectional(forward_out, backward_out)

            # Apply layer normalization and dropout
            merged = layer_norm(merged)
            merged = dropout(merged)

            # Add residual connection if enabled and dimensions match
            if self.use_residual and i > 0 and merged.size(-1) == current_input.size(-1):
                merged = merged + current_input

            # Store layer output
            layer_outputs.append(merged)

            # Update input for next layer
            current_input = merged

        # Combine all layer outputs
        if self.use_residual:
            final_output = sum(layer_outputs) / len(layer_outputs)
        else:
            final_output = layer_outputs[-1]

        # Get sequence representation
        if masks is not None:
            # Use mask to get valid sequence lengths
            lengths = masks.sum(dim=1)
            # Get last valid output for each sequence
            final_states = torch.stack([final_output[i, length - 1] for i, length in enumerate(lengths)])
        else:
            # Use last state if no masks provided
            final_states = final_output[:, -1]

        return final_states

    def forward(
        self,
        inputs: List[str] = None,
        input_ids: TensorType["bs", "seq_len"] = None,
        masks: TensorType["bs", "seq_len"] = None,
        **kwargs,
    ):
        """
        Forward pass of the enhanced BiGRU model.
        """
        if inputs is None and input_ids is None:
            raise ValueError("inputs and input_ids cannot both be None")

        # Get embeddings
        if input_ids is None:
            embed = self.word_embedding.forward(inputs)
        else:
            embed = self.word_embedding.forward(input_ids=input_ids)

        # Process sequence through GRU layers
        final_states = self._process_sequence(embed, masks)

        # Classification
        logits = self.classifier(final_states)

        # Apply softmax
        logits = nn.functional.softmax(logits, dim=-1)

        return logits
