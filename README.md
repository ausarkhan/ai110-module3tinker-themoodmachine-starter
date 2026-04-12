# Mood Machine: Hybrid Sentiment System

## 1) Original Project Summary
The original Mood Machine started as a rule-based sentiment classifier for short social-style text. It used token matching against positive and negative word lists, then mapped a score to one of four labels: `positive`, `negative`, `neutral`, or `mixed`. The project was designed to show how basic NLP pipelines work and where they fail.

## 2) Final System Overview
This repository is now a complete end-to-end applied AI system with a production-style prediction flow:

`input -> preprocessing -> rule model + ML model -> confidence/uncertainty routing -> final output`

The system now returns:
- A final label
- A confidence score
- A reliability reason
- Rule-vs-ML agreement status
- Intermediate evidence used by the rule model

Why this matters:
- It is more trustworthy than a single raw label.
- It handles ambiguous text by returning `uncertain` instead of pretending certainty.
- It demonstrates reliability-aware AI design, not just classification.

## 3) AI Feature Explanation
### Added Feature
A **hybrid decision layer with confidence scoring and uncertainty handling** was added to core logic in `mood_analyzer.py`.

### Why It Improves the System
- Single-model outputs are brittle on slang/sarcasm.
- The hybrid layer checks whether rule-based and ML signals agree.
- Low-confidence disagreement is explicitly routed to `uncertain`.
- High-confidence disagreement is resolved by the stronger signal.

### How It Works
1. Rule analysis computes token evidence, score, and rule confidence.
2. ML analysis predicts label + probability confidence (Logistic Regression).
3. Router combines both:
- If models agree: average confidence and return shared label.
- If models disagree and one is high-confidence: use higher-confidence model.
- If models disagree and both are low-confidence: return `uncertain`.
4. Empty input is validated and returned as `uncertain` with confidence `0.0`.

This feature is real because it directly changes final predictions in runtime execution.

## 4) Architecture Overview
See `architecture_diagram.md` for the diagram and flow details.

High-level data flow:
1. User text enters `main.py`.
2. `MoodAnalyzer.analyze()` preprocesses tokens.
3. Rule subsystem computes score/evidence/confidence.
4. ML subsystem computes predicted label/probability confidence.
5. Reliability router decides final output (`label`, `confidence`, `reason`).
6. Evaluation and tests validate behavior on edge cases.

## 5) Setup Instructions
1. Open this repository in VS Code.
2. Create/activate your Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the full hybrid app:

```bash
python main.py
```

5. Run automated tests:

```bash
python tests.py
```

Optional: run the standalone ML script:

```bash
python ml_experiments.py
```

## 6) Sample Interactions
### Example A
Input:
`I am not happy about this`

Output (example):
`negative (conf=0.81, reason=Rule and ML agree)`

### Example B
Input:
`Yeah great, another 8am meeting... love that for me`

Output (example):
`uncertain (conf=0.61, reason=Rule/ML disagreement with low confidence)`

### Example C
Input:
`I passed the test but I am still stressed`

Output (example):
`mixed (conf=0.74, reason=Rule and ML agree)`

## 7) Design Decisions
- Chose a hybrid system over only rule updates because it adds a real AI reliability mechanism.
- Kept model simple (CountVectorizer + LogisticRegression) for readability and reproducibility.
- Added uncertainty routing to avoid overconfident wrong outputs.
- Preserved `predict_label()` for compatibility with existing scripts.
- Returned structured outputs from `analyze()` so behavior is testable and inspectable.

Trade-off:
- Better reliability signaling, but slightly more complexity than a pure rule model.

## 8) Testing Summary
Reliability/testing assets:
- `tests.py` includes automated checks for required edge cases:
- sarcasm
- slang
- emojis
- mixed sentiment
- empty input
- explanation and reliability metadata

Runtime evaluation path:
- `main.py` now reports:
- accuracy on labeled dataset
- average confidence
- uncertain prediction rate

See `evaluation_results.md` for a concise results summary template and recorded observations.

## 9) Reflection (Required)
### Limitations and Biases
- The dataset is small and not representative of all dialects or communities.
- Sarcasm and context-dependent slang can still be misread.
- Confidence is calibrated heuristically for rules and probability-based for ML; it is not perfectly calibrated.

### Potential Misuse + Prevention
Potential misuse:
- Treating predictions as mental-health diagnosis or high-stakes decision input.

Prevention:
- Keep outputs advisory only.
- Surface confidence and uncertainty.
- Add policy note: do not use for hiring, discipline, or clinical decisions.

### What Surprised Me During Testing
The strongest surprise was that agreement between models was often a better reliability signal than either model alone on slang-heavy examples.

### One Helpful AI Suggestion
Use disagreement-based uncertainty routing so the system can abstain when evidence conflicts.

### One Incorrect/Misleading AI Suggestion
Blindly trusting training-set accuracy as proof of quality was misleading because small datasets can overfit and look unrealistically strong.

## Project Files
- `main.py` - End-to-end hybrid execution and evaluation
- `mood_analyzer.py` - Core hybrid AI pipeline
- `tests.py` - Automated reliability tests
- `architecture_diagram.md` - System architecture diagram
- `evaluation_results.md` - Reliability/evaluation summary
- `ml_experiments.py` - Optional standalone ML experiment script
- `dataset.py` - Lexicons and labeled data
