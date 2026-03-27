
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_model(y_test, preds, probs):

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_test, preds)
    metrics["precision"] = precision_score(y_test, preds)
    metrics["recall"] = recall_score(y_test, preds)
    metrics["f1_score"] = f1_score(y_test, preds)
    metrics["roc_auc"] = roc_auc_score(y_test, probs)

    return metrics
