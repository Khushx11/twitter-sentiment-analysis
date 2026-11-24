from collections import Counter
from sklearn.model_selection import train_test_split
from .labels import LABELS

def split(texts, labels, seed):
    counts = Counter(labels)
    if any(counts[label] < 5 for label in LABELS):
        raise ValueError("Provide at least five distinct examples per sentiment")
    train_x, test_x, train_y, test_y = train_test_split(
        texts, labels, test_size=0.25, random_state=seed, stratify=labels
    )
    return train_x, test_x, train_y, test_y
