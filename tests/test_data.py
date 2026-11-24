import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from post_sentiment.data import load

class DataTests(unittest.TestCase):
    def test_duplicate_policy(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            path.write_text("text,label\nSame,neutral\nsame,neutral\n")
            self.assertEqual(load(path), (["same"], ["neutral"]))
            with path.open("a") as stream:
                stream.write("SAME,negative\n")
            with self.assertRaisesRegex(ValueError, "Conflicting labels"):
                load(path)
