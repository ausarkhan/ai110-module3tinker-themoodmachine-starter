"""Automated tests for the Mood Machine hybrid system."""

import unittest

from mood_analyzer import MoodAnalyzer


class TestMoodMachineHybrid(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = MoodAnalyzer(use_ml=True)

    def test_empty_input_returns_uncertain(self) -> None:
        result = self.analyzer.analyze("   ")
        self.assertEqual(result["label"], "uncertain")
        self.assertEqual(result["confidence"], 0.0)

    def test_slang_positive(self) -> None:
        result = self.analyzer.analyze("No cap this song is fire")
        self.assertIn(result["label"], {"positive", "mixed", "uncertain"})
        self.assertGreaterEqual(result["confidence"], 0.4)

    def test_emoji_input_supported(self) -> None:
        result = self.analyzer.analyze("😂😂😂")
        self.assertIn(result["label"], {"neutral", "negative", "uncertain", "mixed"})
        self.assertIn("reason", result)

    def test_mixed_sentiment(self) -> None:
        result = self.analyzer.analyze("I am happy but also tired")
        self.assertIn(result["label"], {"mixed", "uncertain", "positive", "negative"})
        self.assertIn("rule", result)

    def test_sarcasm_has_reliability_signals(self) -> None:
        text = "Yeah great, another 8am meeting... love that for me"
        result = self.analyzer.analyze(text)
        self.assertIn("confidence", result)
        self.assertIn("agreement", result)
        self.assertIn("reason", result)

    def test_explain_contains_pipeline_details(self) -> None:
        explanation = self.analyzer.explain("I am not happy about this")
        self.assertIn("label=", explanation)
        self.assertIn("conf=", explanation)
        self.assertIn("rule=", explanation)


if __name__ == "__main__":
    unittest.main()
