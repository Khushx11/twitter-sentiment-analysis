"""Evaluate the linear sentiment model on the official TweetEval splits."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score

from .artefacts import save
from .labels import LABELS
from .metrics import score
from .models import classifier_pipeline
from .text import normalise

REVISION = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
SOURCE = f"https://github.com/cardiffnlp/tweeteval/tree/{REVISION}/datasets/sentiment"
SPLITS = {"train": 45615, "val": 2000, "test": 12284}


def benchmark(directory, output):
    data, hashes = {}, {}
    for split, count in SPLITS.items():
        text_path = directory / f"{split}_text.txt"
        labels_path = directory / f"{split}_labels.txt"
        texts = [normalise(line) for line in text_path.read_text(encoding="utf-8").splitlines()]
        raw_labels = labels_path.read_text(encoding="utf-8").splitlines()
        if len(texts) != count or len(raw_labels) != count or set(raw_labels) != {"0", "1", "2"}:
            raise ValueError(f"Unexpected TweetEval {split} split")
        data[split] = (texts, [LABELS[int(label)] for label in raw_labels])
        for path in (text_path, labels_path):
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    test_x, test_y = data["test"]
    model = classifier_pipeline()
    features = model.named_steps["featureunion"]
    transformed = features.fit_transform(train_x)
    validation = features.transform(val_x)
    estimator = model.named_steps["logisticregression"]
    trials = []
    for c in (0.3, 1.0, 3.0):
        estimator.set_params(C=c).fit(transformed, train_y)
        trials.append({"C": c, "macro_recall": recall_score(val_y, estimator.predict(validation), average="macro")})
    selected = max(trials, key=lambda trial: trial["macro_recall"])
    estimator.set_params(C=selected["C"]).fit(transformed, train_y)

    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train_y), 1)), train_y)
    train_unique = set(train_x)
    overlap = {split: sum(text in train_unique for text in data[split][0]) for split in ("val", "test")}
    report = {
        "dataset": "TweetEval sentiment",
        "source": SOURCE,
        "split_sizes": SPLITS,
        "class_order": LABELS,
        "overlap_with_train": overlap,
        "input_sha256": hashes,
        "validation_trials": trials,
        "selected_C": selected["C"],
        "logistic_regression": score(test_y, model.predict(test_x)),
        "majority_baseline": score(test_y, baseline.predict(np.zeros((len(test_y), 1)))),
    }
    for name in ("logistic_regression", "majority_baseline"):
        report[name]["macro_recall"] = float(np.mean([
            report[name]["class_metrics"][label]["recall"] for label in LABELS
        ]))
    save(output, model, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark(args.directory, args.output)
    print(f"Test macro recall: {report['logistic_regression']['macro_recall']:.4f}")


if __name__ == "__main__":
    main()
