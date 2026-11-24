from .text import normalise

def predict(classifier, post):
    clean = normalise(post)
    if not clean:
        raise ValueError("Prediction text cannot be blank")
    labels = classifier.classes_
    probabilities = classifier.predict_proba([clean])[0]
    return {
        "label": str(labels[probabilities.argmax()]),
        "model_scores": {str(label): float(value) for label, value in zip(labels, probabilities)},
    }
