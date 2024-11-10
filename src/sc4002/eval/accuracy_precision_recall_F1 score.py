from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred.argmax(dim=1))

def compute_precision_recall_f1(y_true, y_pred):
    y_pred_labels = y_pred.argmax(dim=1)
    precision = precision_score(y_true, y_pred_labels, average="weighted")
    recall = recall_score(y_true, y_pred_labels, average="weighted")
    f1 = f1_score(y_true, y_pred_labels, average="weighted")
    return precision, recall, f1
