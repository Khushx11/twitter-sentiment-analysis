import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from .features import features

def classifier_pipeline():
    return make_pipeline(
        features(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )


def train(texts, labels):
    classifier = classifier_pipeline().fit(texts, labels)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(labels), 1)), labels)
    return classifier, baseline
