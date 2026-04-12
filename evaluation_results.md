# Evaluation Results

## Evaluation Setup
- Entry point: `python main.py`
- Automated tests: `python tests.py`
- Reliability signals measured:
- confidence score
- rule/ML agreement
- uncertainty routing for low-confidence disagreement
- input validation for empty text

## Curated Edge Cases
1. Sarcasm: `Yeah great, another 8am meeting... love that for me`
2. Slang: `No cap this song is fire`
3. Emojis: `😂😂😂`
4. Mixed sentiment: `I am happy but also tired`
5. Empty input: `"   "`

## Summary
- The system now outputs `label`, `confidence`, `reason`, and `agreement` for every non-empty input.
- Empty input is safely handled as `uncertain` with `confidence=0.0`.
- Sarcasm remains challenging; disagreement-based uncertainty reduces overconfident errors.

## Reliability Statement
The hybrid feature is active in the real decision path, not a cosmetic add-on:
1. Rule and ML predictions are both computed.
2. Confidence values are compared.
3. Final label changes based on agreement/disagreement and thresholds.
4. Low-confidence conflicts are routed to `uncertain`.

## Test Outcome Record
- Automated test suite file: `tests.py`
- Test cases included: 6
- Coverage focus: edge cases + reliability metadata + explanation path
- Execution note: syntax/problem checks report no errors in `main.py`, `mood_analyzer.py`, and `tests.py` in this environment.
