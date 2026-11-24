import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from post_sentiment.benchmark import read_verified


class BenchmarkInputTests(unittest.TestCase):
    def test_rejects_modified_official_split(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "train_text.txt").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "pinned TweetEval revision"):
                read_verified(directory, "train_text.txt")
