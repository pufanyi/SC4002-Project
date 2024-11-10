from typing import List, Literal
import torch
import torch.nn as nn
from torchtyping import TensorType

from .base_model import BaseModel
from .glove import Glove

class BidirectionalLSTM(BaseModel):
    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 512,
        output_dim: int = 2,
        num_layers: int = 1,
        dropout: float = 0.2,
        model_name: str = "bilstm",
        ckpt_path: str | None = None,
        tokenizer_path: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the Bidirectional LSTM model.
        
        Args:
            input_dim: The dimension of input features (default: 300 for GloVe embeddings)
            hidden_dim: The dimension of hidden state (default: 512)
            output_dim: The dimension of output (default: 2 for binary classification)
            num_layers: Number of LSTM layers (default: 1)
            dropout: Dropout rate (default: 0.2)
            model_name: Name of the model (default: "bilstm")
            ckpt_path: Path to checkpoint file
            tokenizer_path: Path to tokenizer file
        """
        super().__init__(model_name, *args, **kwargs)
        
        # Initialize word embeddings using GloVe
        self.word_embedding = Glove(ckpt_path=ckpt_path, tokenizer_path=tokenizer_path)
        
        # Initialize bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Linear layer for classification
        # Note: hidden_dim * 2 because of bidirectional
        self.linear_head = nn.Linear(hidden_dim * 2, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: List[str] = None,
        input_ids: TensorType["bs", "seq_len"] = None,
        masks: TensorType["bs", "seq_len"] = None,
        **kwargs,
    ):
        """
        Forward pass of the BiLSTM model.
        
        Args:
            inputs: List of input strings
            input_ids: Tensor of input ids
            masks: Tensor of attention masks
        
        Returns:
            logits: Output logits after softmax
        """
        # Input validation
        if inputs is None and input_ids is None:
            raise ValueError("inputs and input_ids cannot both be None")
            
        # Get embeddings
        if input_ids is None:
            embed = self.word_embedding.forward(inputs)
        else:
            embed = self.word_embedding.forward(input_ids=input_ids)

        if masks is not None:
            # Handle padded sequences
            output = []
            for em, mask in zip(embed, masks):
                # Process only non-padded tokens
                valid_embed = em[mask].unsqueeze(0)
                
                # Forward through LSTM
                o, (hidden, cell) = self.lstm(valid_embed)
                
                # Concatenate forward and backward hidden states
                # Take the output from the last time step
                final_hidden = torch.cat([o[:, -1, :self.lstm.hidden_size],
                                       o[:, 0, self.lstm.hidden_size:]], dim=1)
                
                output.append(final_hidden)
            
            output = torch.cat(output, dim=0)
        
        else:
            # Process the full sequence
            output, (hidden, cell) = self.lstm(embed)
            
            # Concatenate forward and backward hidden states
            # Take the output from the last time step
            output = torch.cat([output[:, -1, :self.lstm.hidden_size],
                              output[:, 0, self.lstm.hidden_size:]], dim=1)

        # Apply dropout
        output = self.dropout(output)
        
        # Pass through linear layer
        logits = self.linear_head(output)
        
        # Apply softmax
        logits = nn.functional.softmax(logits, dim=1)
        
        return logits