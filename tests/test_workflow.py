import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from post_sentiment.artefacts import load_trusted
from post_sentiment.inference import predict
from post_sentiment.workflow import run

class WorkflowTests(unittest.TestCase):
    def test_train_and_predict(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = Path(__file__).resolve().parents[1] / "examples/labelled_posts.csv"
            first = run(source, root / "first", 42)
            second = run(source, root / "second", 42)
            self.assertEqual(first, second)
            self.assertEqual(sum(first["test_counts"].values()), 8)
            result = predict(load_trusted(root / "first/model.joblib"), "Great update!")
            self.assertAlmostEqual(sum(result["model_scores"].values()), 1)
            self.assertIn(result["label"], first["class_order"])
