"""Evaluate the linear sentiment model on the official TweetEval splits."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score
from sklearn.svm import LinearSVC

from .artefacts import save
from .labels import LABELS
from .metrics import score
from .models import classifier_pipeline
from .text import normalise

REVISION = "4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66"
SOURCE = f"https://github.com/cardiffnlp/tweeteval/tree/{REVISION}/datasets/sentiment"
SPLITS = {"train": 45615, "val": 2000, "test": 12284}


EXPECTED_SHA256 = {
    "train_text.txt": "368f01052ea6fd8ffc408a2a2e6ac9669e31542581a0396ef16591ea26eb98a6",
    "train_labels.txt": "122bfb1732fb6995b0e5c5f726c0ba457c469c3b6e60513007ce5037f23e65d4",
    "val_text.txt": "e5b021e6fc45064c260b09814b803d8f56cada519c4d952d72f43d48a350a964",
    "val_labels.txt": "b4566926c72e2e4e2916c864def94e76c4cdde52446af2c7ba4fc2006e057e51",
    "test_text.txt": "09a93a55c63fd93f97485ef7302889d7edb4091cd49733aa37da094f0bfa0675",
    "test_labels.txt": "6afb4afe9374d1f983bcf9a7c79b108d0f37fdf020a83f30488309bed215db9d",
}


def read_verified(directory, name):
    raw = (directory / name).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256[name]:
        raise ValueError(f"{name} does not match the pinned TweetEval revision")
    return raw.decode("utf-8").splitlines(), digest


def select(estimator, train, train_labels, validation, validation_labels):
    trials = []
    for c in (0.3, 1.0, 3.0):
        estimator.set_params(C=c).fit(train, train_labels)
        recall = recall_score(validation_labels, estimator.predict(validation), average="macro")
        trials.append({"C": c, "macro_recall": recall})
    chosen = max(trials, key=lambda trial: trial["macro_recall"])
    estimator.set_params(C=chosen["C"]).fit(train, train_labels)
    return chosen["C"], trials


def benchmark(directory, output):
    data, hashes = {}, {}
    for split, count in SPLITS.items():
        text_name, label_name = f"{split}_text.txt", f"{split}_labels.txt"
        raw_texts, hashes[text_name] = read_verified(directory, text_name)
        raw_labels, hashes[label_name] = read_verified(directory, label_name)
        texts = [normalise(line) for line in raw_texts]
        if len(texts) != count or len(raw_labels) != count or set(raw_labels) != {"0", "1", "2"}:
            raise ValueError(f"Unexpected TweetEval {split} split")
        data[split] = (texts, [LABELS[int(label)] for label in raw_labels])

    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    test_x, test_y = data["test"]
    model = classifier_pipeline()
    features = model.named_steps["featureunion"]
    transformed = features.fit_transform(train_x)
    validation = features.transform(val_x)
    estimator = model.named_steps["logisticregression"]
    selected_c, trials = select(estimator, transformed, train_y, validation, val_y)
    svm = LinearSVC(class_weight="balanced", max_iter=10000, random_state=42)
    svm_c, svm_trials = select(svm, transformed, train_y, validation, val_y)

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
        "selected_C": selected_c,
        "svm_validation_trials": svm_trials,
        "svm_selected_C": svm_c,
        "logistic_regression": score(test_y, model.predict(test_x)),
        "linear_svm": score(test_y, svm.predict(features.transform(test_x))),
        "majority_baseline": score(test_y, baseline.predict(np.zeros((len(test_y), 1)))),
    }
    for name in ("logistic_regression", "linear_svm", "majority_baseline"):
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
