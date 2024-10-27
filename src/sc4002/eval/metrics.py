from typing import Dict

import evaluate
from transformers.trainer_utils import EvalPrediction


def compute_acc(pred: EvalPrediction, compute_result: bool = False) -> Dict[str, float]:
    labels = pred.label_ids
    logits = pred.predictions
    pred = logits.argmax(axis=1)
    accuracy_metric = evaluate.load("accuracy")
    acc = accuracy_metric.compute(predictions=pred, references=labels)
    return {**acc}
