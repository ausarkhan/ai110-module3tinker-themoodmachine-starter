# Model Card: Mood Machine

This project compares two mood classifiers:

1. Rule based model in mood_analyzer.py
2. ML model in ml_experiments.py (CountVectorizer + LogisticRegression)

## 1. Model Overview

Model type:
Both models were used and compared.

Intended purpose:
Classify short text posts into one of four labels: positive, negative, neutral, mixed.

How it works (brief):
The rule model scores sentiment words and maps score patterns to labels.
The ML model learns word-to-label patterns from the labeled posts.

## 2. Data

Dataset description:
The dataset now has 14 labeled posts.
I appended 8 new examples to the original 6.

Labeling process:
Labels were assigned by human judgment using the four project labels.
Posts with conflicting signals were labeled mixed.
Some ambiguous items (sarcasm or slang) could reasonably receive different labels depending on reader interpretation.

Important characteristics:

- Includes slang: lowkey, no cap, sick, fire
- Includes emojis and emoticons: 🔥, 🙃, 😴, :)
- Includes mixed-emotion language using contrast words like but
- Includes neutral factual statements
- Includes sarcasm-like phrasing

Possible issues with the dataset:

- Very small size (14 posts)
- Label ambiguity for sarcasm
- Limited coverage of dialects, cultures, and contexts
- Class balance is acceptable but still too small for stable ML behavior

## 3. How the Rule Based Model Works

Scoring rules:

- Preprocess lowercases, removes simple punctuation, then tokenizes by whitespace
- Positive tokens add +1
- Negative tokens add -1
- Negation handling flips the next sentiment token (example: not happy -> negative effect)
- Label thresholds are configurable in code:
- score >= 1 -> positive
- score <= -1 -> negative
- score == 0 -> neutral
- If both positive and negative cues appear (or contrast cue like but/however with non-zero score), return mixed

Enhancements added:

- Negation handling in scoring (main Part 2 enhancement)
- Targeted vocabulary update for slang: added sick and fire as positive cues (Part 3 failure fix)

Strengths:

- Easy to explain and debug
- Works well on direct sentiment wording
- Handles basic mixed and negation cases

Weaknesses:

- Still fails on sarcasm and implied context
- Emoji-only text remains weak
- Sensitive to exact vocabulary choices

## 4. How the ML Model Works

Features used:
Bag-of-words features via CountVectorizer.

Training data:
Trained on SAMPLE_POSTS and TRUE_LABELS from dataset.py.

Training behavior:
Training accuracy is high on this tiny dataset, but this likely reflects overfitting because evaluation uses the same data used for training.

Strengths and weaknesses:

- Strength: Learns useful cues automatically from labeled data
- Weakness: Very sensitive to dataset size and label noise
- Weakness: Can learn incorrect associations (example: slang or emoji patterns)

## 5. Evaluation

How evaluated:
Both models were run on the same labeled dataset in dataset.py.

Results:

- Rule based accuracy: 0.9286 (13/14)
- ML training-set accuracy: 1.0000 (14/14)

Rule-based misclassification observed:

- "Yeah awesome... my bus was late again 🙃" was predicted positive but labeled negative
- Failure pattern: sarcasm with a positive keyword (awesome)

Stress test observations:

- Both models mislabeled sarcasm such as "Yeah great, another 8am meeting... love that for me" as positive
- Rule model did better on slang after vocabulary update: "That game was sick" -> positive
- ML model treated some hard examples as negative:
- "That game was sick" -> negative
- "Not bad actually" -> negative
- Rule model output neutral for emoji-only text "😂😂😂" while ML predicted negative

Examples of correct predictions:

- "I am not happy about this" -> negative (negation handled)
- "I passed the test but I am still stressed" -> mixed
- "Just finished laundry and cleaned the kitchen" -> neutral

## 6. Limitations

- Dataset is too small for reliable generalization
- No held-out test split in current workflow
- Rule model depends on manually selected vocabulary
- ML model likely overfits because train and eval sets are identical
- Both models struggle with sarcasm, context, and cultural language variation

## 7. Ethical Considerations

- Misclassifying distress-related text can cause harm if used in support contexts
- Language communities using slang or sarcasm may be systematically misread
- Mood inference from personal messages raises privacy concerns
- Model outputs should not be used for high-stakes decisions about people

## 8. Ideas for Improvement

- Add a larger, more diverse labeled dataset
- Create a true train/validation/test split
- Add emoji-aware and phrase-aware preprocessing
- Use TF-IDF features and compare against bag-of-words
- Track per-class precision/recall, not only accuracy
- Add confidence thresholds and abstain/uncertain outputs
