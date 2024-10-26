from typing import Dict, List, Optional, Union

import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]
        logits = model(input_ids=input_ids)
        loss = nn.functional.cross_entropy(logits, labels)
        return (loss, logits) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Union[Dataset, Dict[str, Dataset], None] = None,
        ignore_keys: Union[List[str], None] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
