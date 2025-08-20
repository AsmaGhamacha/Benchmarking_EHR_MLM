from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)

def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(-1)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "micro_precision": precision_score(y_true, y_pred, average="micro"),
        "micro_recall": recall_score(y_true, y_pred, average="micro")
    }
