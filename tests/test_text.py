import unittest
from post_sentiment.text import normalise

class NormaliseTests(unittest.TestCase):
    def test_preserves_negation_and_masks_identifiers(self):
        self.assertEqual(normalise("NOT good! @Alice https://site.example"), "not good! user url")
