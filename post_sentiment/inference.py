from .text import normalise

def predict(classifier, post):
    clean = normalise(post)
    if not clean:
        raise ValueError("Prediction text cannot be blank")
    features = classifier.named_steps["featureunion"].transform([clean])
    if features.nnz == 0:
        raise ValueError("No terms or character patterns from the training vocabulary were recognised.")
    estimator = classifier.named_steps["logisticregression"]
    labels = estimator.classes_
    probabilities = estimator.predict_proba(features)[0]
    return {
        "label": str(labels[probabilities.argmax()]),
        "model_scores": {str(label): float(value) for label, value in zip(labels, probabilities)},
    }
