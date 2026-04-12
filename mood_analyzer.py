"""
Hybrid mood analyzer with reliability signals.

Pipeline:
  input -> preprocessing -> rule score + ML score -> confidence routing -> output
"""

from typing import Any, Dict, List, Optional
import logging
import re

from dataset import NEGATIVE_WORDS, POSITIVE_WORDS, SAMPLE_POSTS, TRUE_LABELS

try:
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - fallback for missing optional dependency
  CountVectorizer = None
  LogisticRegression = None


LOGGER = logging.getLogger(__name__)


class MoodAnalyzer:
  """A hybrid mood classifier that combines rule-based and ML predictions."""

  def __init__(
    self,
    positive_words: Optional[List[str]] = None,
    negative_words: Optional[List[str]] = None,
    use_ml: bool = True,
    uncertainty_threshold: float = 0.55,
    high_confidence_threshold: float = 0.75,
  ) -> None:
    positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
    negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

    self.positive_words = set(w.lower() for w in positive_words)
    self.negative_words = set(w.lower() for w in negative_words)
    self.negation_words = {
      "not",
      "never",
      "no",
      "dont",
      "can't",
      "cant",
      "cannot",
      "isn't",
      "isnt",
      "wasn't",
      "wasnt",
      "won't",
      "wont",
    }
    self.contrast_words = {"but", "however", "though", "yet"}

    self.positive_threshold = 1
    self.negative_threshold = -1
    self.uncertainty_threshold = uncertainty_threshold
    self.high_confidence_threshold = high_confidence_threshold

    self.vectorizer = None
    self.model = None
    self.ml_enabled = False

    if use_ml and CountVectorizer is not None and LogisticRegression is not None:
      self._train_ml_backend(SAMPLE_POSTS, TRUE_LABELS)

  def _train_ml_backend(self, texts: List[str], labels: List[str]) -> None:
    if len(texts) != len(labels) or not texts:
      LOGGER.warning("ML backend disabled due to invalid training data.")
      self.ml_enabled = False
      return

    self.vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
    X = self.vectorizer.fit_transform(texts)
    self.model = LogisticRegression(max_iter=1000, multi_class="auto")
    self.model.fit(X, labels)
    self.ml_enabled = True

  def preprocess(self, text: str) -> List[str]:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[.,!?;:\"'()\[\]{}]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return [token for token in cleaned.split(" ") if token]

  def _rule_analysis(self, text: str) -> Dict[str, Any]:
    tokens = self.preprocess(text)
    if not tokens:
      return {
        "label": "uncertain",
        "confidence": 0.0,
        "score": 0,
        "positive_hits": [],
        "negative_hits": [],
        "tokens": [],
      }

    score = 0
    negate_next = False
    positive_hits: List[str] = []
    negative_hits: List[str] = []

    for token in tokens:
      if token in self.negation_words:
        negate_next = True
        continue

      if token in self.positive_words:
        applied = -1 if negate_next else 1
        score += applied
        if applied > 0:
          positive_hits.append(token)
        else:
          negative_hits.append(f"not_{token}")
        negate_next = False
        continue

      if token in self.negative_words:
        applied = 1 if negate_next else -1
        score += applied
        if applied < 0:
          negative_hits.append(token)
        else:
          positive_hits.append(f"not_{token}")
        negate_next = False
        continue

      negate_next = False

    has_positive = bool(positive_hits)
    has_negative = bool(negative_hits)
    has_contrast = any(token in self.contrast_words for token in tokens)

    if (has_positive and has_negative) or (has_contrast and score != 0):
      label = "mixed"
    elif score >= self.positive_threshold:
      label = "positive"
    elif score <= self.negative_threshold:
      label = "negative"
    else:
      label = "neutral"

    evidence_hits = len(positive_hits) + len(negative_hits)
    confidence = min(0.95, 0.45 + (0.18 * abs(score)) + (0.07 * evidence_hits))
    if label == "mixed":
      confidence = max(confidence, 0.65)
    if label == "neutral" and evidence_hits == 0:
      confidence = 0.5

    return {
      "label": label,
      "confidence": round(confidence, 2),
      "score": score,
      "positive_hits": positive_hits,
      "negative_hits": negative_hits,
      "tokens": tokens,
    }

  def score_text(self, text: str) -> int:
    """Return the rule-based sentiment score for compatibility."""
    return int(self._rule_analysis(text)["score"])

  def _ml_analysis(self, text: str) -> Optional[Dict[str, Any]]:
    if not self.ml_enabled or self.vectorizer is None or self.model is None:
      return None

    X = self.vectorizer.transform([text])
    label = self.model.predict(X)[0]
    probabilities = self.model.predict_proba(X)[0]
    confidence = float(max(probabilities))
    return {
      "label": str(label),
      "confidence": round(confidence, 2),
    }

  def analyze(self, text: str) -> Dict[str, Any]:
    """Run the full hybrid pipeline and return structured output."""
    if not isinstance(text, str):
      raise TypeError("Input text must be a string.")

    if not text.strip():
      LOGGER.warning("Empty input detected; returning uncertain result.")
      return {
        "label": "uncertain",
        "confidence": 0.0,
        "reason": "Empty input.",
        "rule": self._rule_analysis(text),
        "ml": None,
        "agreement": None,
      }

    rule_result = self._rule_analysis(text)
    ml_result = self._ml_analysis(text)

    if ml_result is None:
      if rule_result["confidence"] < self.uncertainty_threshold:
        final_label = "uncertain"
        reason = "Rule confidence below threshold."
      else:
        final_label = rule_result["label"]
        reason = "Rule-only prediction."
      return {
        "label": final_label,
        "confidence": rule_result["confidence"],
        "reason": reason,
        "rule": rule_result,
        "ml": None,
        "agreement": None,
      }

    agreement = rule_result["label"] == ml_result["label"]
    if agreement:
      final_label = rule_result["label"]
      final_confidence = round((rule_result["confidence"] + ml_result["confidence"]) / 2.0, 2)
      reason = "Rule and ML agree."
    else:
      if max(rule_result["confidence"], ml_result["confidence"]) >= self.high_confidence_threshold:
        if ml_result["confidence"] >= rule_result["confidence"]:
          final_label = ml_result["label"]
          final_confidence = ml_result["confidence"]
          reason = "Disagreement resolved in favor of higher-confidence ML signal."
        else:
          final_label = rule_result["label"]
          final_confidence = rule_result["confidence"]
          reason = "Disagreement resolved in favor of higher-confidence rule signal."
      else:
        final_label = "uncertain"
        final_confidence = round((rule_result["confidence"] + ml_result["confidence"]) / 2.0, 2)
        reason = "Rule/ML disagreement with low confidence."

    return {
      "label": final_label,
      "confidence": final_confidence,
      "reason": reason,
      "rule": rule_result,
      "ml": ml_result,
      "agreement": agreement,
    }

  def predict_label(self, text: str) -> str:
    """Compatibility wrapper used by existing scripts."""
    return str(self.analyze(text)["label"])

  def explain(self, text: str) -> str:
    result = self.analyze(text)
    rule = result["rule"]
    ml = result["ml"]
    ml_part = f", ml={ml['label']}({ml['confidence']:.2f})" if ml is not None else ""
    return (
      f"label={result['label']} conf={result['confidence']:.2f} "
      f"reason={result['reason']} rule={rule['label']}({rule['confidence']:.2f})"
      f" score={rule['score']} pos={rule['positive_hits']} neg={rule['negative_hits']}{ml_part}"
    )
