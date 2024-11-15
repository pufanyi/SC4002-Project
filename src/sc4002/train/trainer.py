from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn
from torch import nn
from transformers import Trainer
from sc4002.models.base_model import BaseModel


class CustomTrainer(Trainer):
    def compute_loss(self, model: BaseModel, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        masks = inputs["masks"]
        logits = model(input_ids=input_ids, masks=masks)
        loss = nn.functional.cross_entropy(logits, labels)
        return (loss, dict(logits=logits)) if return_outputs else loss
