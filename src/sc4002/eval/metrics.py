from typing import Dict

import evaluate
from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_acc(pred: EvalPrediction, compute_result: bool = False) -> Dict[str, float]:
    labels = pred.label_ids
    logits = pred.predictions
    pred = logits.argmax(axis=1)
    accuracy_metric = evaluate.load("accuracy")
    acc = accuracy_metric.compute(predictions=pred, references=labels)
    return {**acc}


def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred.argmax(dim=1))


def compute_precision_recall_f1(y_true, y_pred):
    y_pred_labels = y_pred.argmax(dim=1)
    precision = precision_score(y_true, y_pred_labels, average="weighted")
    recall = recall_score(y_true, y_pred_labels, average="weighted")
    f1 = f1_score(y_true, y_pred_labels, average="weighted")
    return precision, recall, f1
