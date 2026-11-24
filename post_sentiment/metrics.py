from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .labels import LABELS

def score(labels, predicted):
    return {
        "macro_f1": f1_score(labels, predicted, labels=LABELS, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, predicted),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=LABELS).tolist(),
        "class_metrics": {
            label: values for label, values in classification_report(
                labels, predicted, labels=LABELS, output_dict=True, zero_division=0
            ).items() if label in LABELS
        },
    }
