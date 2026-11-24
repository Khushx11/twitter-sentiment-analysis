import hashlib
from collections import Counter
import numpy as np
from .artefacts import save
from .data import load
from .labels import LABELS
from .metrics import score
from .models import train
from .split import split

def run(source, output, seed):
    texts, labels = load(source)
    train_x, test_x, train_y, test_y = split(texts, labels, seed)
    model, baseline = train(train_x, train_y)
    report = {
        "seed": seed,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "class_order": LABELS,
        "train_counts": dict(Counter(train_y)),
        "test_counts": dict(Counter(test_y)),
        "logistic_regression": score(test_y, model.predict(test_x)),
        "majority_baseline": score(test_y, baseline.predict(np.zeros((len(test_y), 1)))),
    }
    save(output, model, report)
    return report
