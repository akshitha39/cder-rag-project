"""
eval_pipeline.py
----------------
Week 8 evaluation pipeline for the CDER GraphRAG project.

Runs 3 baseline questions through all three systems (LLM-Only,
Vector RAG, Graph RAG) and scores responses using keyword matching.
Outputs a CSV summary and a bar chart comparison.

Results:
    - week8_eval_results.csv  : System-level accuracy totals
    - week8_accuracy_comparison.png : Bar chart visualization
"""

import csv
import matplotlib.pyplot as plt

from llm_only_demo import answer_llm
from vector_retrieval_demo import answer_question as vector_answer
from graph_rag_demo import graph_answer

# Test questions and their expected keywords
QUESTIONS = [
    "What is a thread?",
    "What is a race condition?",
    "What is a deadlock?"
]

KEYWORDS = {
    "What is a thread?": ["thread"],
    "What is a race condition?": ["race condition", "race"],
    "What is a deadlock?": ["deadlock"],
}

SYSTEMS = ["LLM", "VectorRAG", "GraphRAG"]


def score_answer(question: str, answer: str) -> int:
    """
    Return 1 if the answer contains at least one expected keyword, else 0.

    Args:
        question: The original question (used to look up keywords).
        answer: The system's response text.

    Returns:
        1 (correct) or 0 (incorrect/hallucinated).
    """
    ans = answer.lower()
    for kw in KEYWORDS[question]:
        if kw in ans:
            return 1
    return 0


def evaluate_systems():
    """Run all questions through each system and save results."""
    scores = {name: 0 for name in SYSTEMS}
    results = []

    for q in QUESTIONS:
        print(f"\n{'='*40}")
        print(f"QUESTION: {q}")
        print('='*40)

        llm_resp = answer_llm(q)
        llm_score = score_answer(q, llm_resp)
        scores["LLM"] += llm_score

        vec_resp = vector_answer(q)
        vec_score = score_answer(q, vec_resp)
        scores["VectorRAG"] += vec_score

        graph_resp = graph_answer(q)
        graph_score = score_answer(q, graph_resp)
        scores["GraphRAG"] += graph_score

        results.extend([
            ["LLM", q, llm_score],
            ["VectorRAG", q, vec_score],
            ["GraphRAG", q, graph_score],
        ])

    # Save summary CSV
    with open("week8_eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["System", "TotalCorrect", "MaxPossible"])
        for name in SYSTEMS:
            writer.writerow([name, scores[name], len(QUESTIONS)])

    print("\nSaved: week8_eval_results.csv")

    # Plot accuracy bar chart
    accuracy_values = [scores[name] for name in SYSTEMS]
    plt.figure(figsize=(6, 4))
    plt.bar(["LLM Only", "Vector RAG", "Graph RAG"], accuracy_values)
    plt.title("Week 8 – Baseline Accuracy Comparison")
    plt.ylabel(f"Correct Answers (out of {len(QUESTIONS)})")
    plt.tight_layout()
    plt.savefig("week8_accuracy_comparison.png")
    plt.close()

    print("Saved: week8_accuracy_comparison.png")
    print(f"\nFinal scores: {scores}")


if __name__ == "__main__":
    evaluate_systems()
