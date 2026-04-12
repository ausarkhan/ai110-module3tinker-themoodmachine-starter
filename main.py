"""Entry point for the hybrid Mood Machine system."""

from typing import Dict, List

from dataset import SAMPLE_POSTS, TRUE_LABELS
from mood_analyzer import MoodAnalyzer


STRESS_TEST_POSTS = [
    "Yeah great, another 8am meeting... love that for me",
    "That game was sick",
    "I am happy but also exhausted",
    "😂😂😂",
    "I love waiting in traffic for 2 hours",
    "Not bad actually",
]


def evaluate_system(posts: List[str], labels: List[str]) -> Dict[str, float]:
    """Evaluate the hybrid system and return reliability metrics."""
    analyzer = MoodAnalyzer(use_ml=True)
    correct = 0
    total = len(posts)
    uncertain_count = 0
    confidence_sum = 0.0

    print("=== Hybrid System Evaluation on SAMPLE_POSTS ===")
    for text, true_label in zip(posts, labels):
        result = analyzer.analyze(text)
        predicted_label = result["label"]
        confidence = float(result["confidence"])
        confidence_sum += confidence
        if predicted_label == "uncertain":
            uncertain_count += 1

        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1

        print(
            f'"{text}" -> predicted={predicted_label}, true={true_label}, '
            f"conf={confidence:.2f}, agreement={result['agreement']}"
        )

    if total == 0:
        print("\nNo labeled examples to evaluate.")
        return {"accuracy": 0.0, "avg_confidence": 0.0, "uncertain_rate": 0.0}

    accuracy = correct / total
    avg_confidence = confidence_sum / total
    uncertain_rate = uncertain_count / total

    print(f"\nAccuracy on SAMPLE_POSTS: {accuracy:.2f}")
    print(f"Average confidence: {avg_confidence:.2f}")
    print(f"Uncertain predictions: {uncertain_count}/{total} ({uncertain_rate:.2f})")

    return {
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "uncertain_rate": uncertain_rate,
    }


def run_batch_demo() -> None:
    """Run the hybrid MoodAnalyzer on sample posts."""
    analyzer = MoodAnalyzer(use_ml=True)
    print("\n=== Batch Demo on SAMPLE_POSTS (hybrid) ===")
    for text in SAMPLE_POSTS:
        result = analyzer.analyze(text)
        print(
            f'"{text}" -> {result["label"]} '
            f"(conf={result['confidence']:.2f}, reason={result['reason']})"
        )


def run_stress_tests() -> None:
    """Run handpicked edge cases to reveal reliability behavior."""
    analyzer = MoodAnalyzer(use_ml=True)
    print("\n=== Stress Test (hybrid) ===")
    for text in STRESS_TEST_POSTS:
        result = analyzer.analyze(text)
        print(
            f'"{text}" -> predicted={result["label"]}, '
            f"conf={result['confidence']:.2f}, agreement={result['agreement']}"
        )

    print("\nObserved reliability pattern:")
    print("- Sarcasm is often routed to uncertain when rule and ML disagree.")


def run_interactive_loop() -> None:
    """Interactive loop for the hybrid mood system."""
    analyzer = MoodAnalyzer(use_ml=True)
    print("\n=== Interactive Mood Machine (hybrid) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the Mood Machine.")
            break

        result = analyzer.analyze(user_input)
        print(
            f"Model: {result['label']} "
            f"(conf={result['confidence']:.2f}, reason={result['reason']})"
        )


if __name__ == "__main__":
    metrics = evaluate_system(SAMPLE_POSTS, TRUE_LABELS)

    run_batch_demo()
    run_stress_tests()

    run_interactive_loop()

    print("\nSystem summary:")
    print(
        f"accuracy={metrics['accuracy']:.2f}, "
        f"avg_confidence={metrics['avg_confidence']:.2f}, "
        f"uncertain_rate={metrics['uncertain_rate']:.2f}"
    )
