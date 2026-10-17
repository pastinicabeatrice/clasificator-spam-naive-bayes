import unittest

from naive_bayes import NaiveBayes, tokenize

TEXTS = [
    "castiga bani gratuit acum",
    "premiu gratuit apasa link",
    "oferta castiga premiu",
    "ne vedem la curs",
    "tema la baze de date",
    "vii la cafea dupa curs",
]
LABELS = ["spam"] * 3 + ["ham"] * 3


class NaiveBayesTest(unittest.TestCase):
    def setUp(self):
        self.model = NaiveBayes().fit(TEXTS, LABELS)

    def test_tokenize(self):
        self.assertEqual(tokenize("Salut, Ce FACI?"), ["salut", "ce", "faci"])

    def test_classifies_new_messages(self):
        self.assertEqual(self.model.predict("premiu gratuit"), "spam")
        self.assertEqual(self.model.predict("ne vedem la cafea"), "ham")

    def test_unknown_words_use_prior(self):
        model = NaiveBayes().fit(TEXTS + ["alt mesaj"], LABELS + ["ham"])
        self.assertEqual(model.predict("xyz necunoscut"), "ham")

    def test_top_words(self):
        self.assertIn("gratuit", self.model.top_words("spam", 3))


if __name__ == "__main__":
    unittest.main()
