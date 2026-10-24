"""
Unit tests for text preprocessing functions.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    clean_html,
    clean_urls,
    expand_contractions,
    tokenize,
    remove_stopwords,
    preprocess_text,
)


class TestPreprocessing(unittest.TestCase):
    def test_clean_html(self):
        raw = "<p>This is a <b>great</b> movie!<br /><br />Loved it.</p>"
        cleaned = clean_html(raw)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("<br />", cleaned)
        self.assertIn("great", cleaned)

    def test_clean_urls(self):
        raw = "Check out the review at https://imdb.com/title/tt0111161 and www.rottentomatoes.com"
        cleaned = clean_urls(raw)
        self.assertNotIn("https://imdb.com", cleaned)
        self.assertNotIn("www.rottentomatoes.com", cleaned)

    def test_expand_contractions(self):
        raw = "I didn't like it and won't watch it again, it's terrible."
        expanded = expand_contractions(raw)
        self.assertIn("not", expanded)
        self.assertNotIn("didn't", expanded)
        self.assertNotIn("won't", expanded)

    def test_tokenize(self):
        text = "Superb acting, compelling storyline!"
        tokens = tokenize(text)
        self.assertIn("superb", tokens)
        self.assertIn("acting", tokens)
        self.assertIn("compelling", tokens)
        self.assertIn("storyline", tokens)

    def test_negation_preserved_in_stopwords(self):
        tokens = ["the", "movie", "was", "not", "good", "at", "all"]
        filtered = remove_stopwords(tokens)
        self.assertIn("not", filtered)
        self.assertNotIn("the", filtered)
        self.assertNotIn("was", filtered)

    def test_preprocess_text_full(self):
        raw = "<br />I didn't expect this film to be so extraordinary! Visit http://films.org"
        cleaned = preprocess_text(raw)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("<br", cleaned)
        self.assertIn("extraordinary", cleaned)
        self.assertIn("not", cleaned)


if __name__ == "__main__":
    unittest.main()
