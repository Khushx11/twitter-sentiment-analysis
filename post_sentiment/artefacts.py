import json
import joblib
import sklearn

def save(output, classifier, report):
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump({"schema": 1, "sklearn": sklearn.__version__, "classifier": classifier, "evaluation": report}, output / "model.joblib")
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

def load_bundle(path):
    artefact = joblib.load(path)
    if artefact.get("schema") != 1 or artefact.get("sklearn") != sklearn.__version__:
        raise ValueError("Model format or scikit-learn version differs. Retrain with this environment.")
    return artefact


def load_trusted(path):
    return load_bundle(path)["classifier"]
