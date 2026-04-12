# Mood Machine Architecture

```mermaid
flowchart TD
    A[Input Layer\nUser text from CLI] --> B[Preprocessing\nnormalize + tokenize]
    B --> C[Rule Analyzer\nlexicon scoring + negation + mixed detection]
    B --> D[ML Analyzer\nCountVectorizer + LogisticRegression]
    C --> E[Reliability Router\nconfidence scoring + agreement check + uncertainty handling]
    D --> E
    E --> F[Output Layer\nlabel + confidence + reason + agreement]
    F --> G[Evaluation\naccuracy + avg confidence + uncertain rate]
    F --> H[Automated Tests\nedge cases + reliability fields]
```

## How the AI Feature Fits
The key AI feature is the **Reliability Router**. It takes outputs from both analyzers and makes a confidence-aware final decision:
- Agreement -> return shared label with combined confidence.
- Disagreement with high confidence signal -> pick stronger signal.
- Disagreement with low confidence -> return `uncertain`.

This is integrated directly into `MoodAnalyzer.analyze()` and used by `main.py` in real execution.
