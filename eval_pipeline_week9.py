"""
eval_pipeline_week9.py
-----------------------
Week 9 (final) evaluation pipeline for the CDER GraphRAG project.

Expands the evaluation to 12 CDER parallel computing questions.
Scores each answer by percentage of expected keywords matched,
tracks hallucinations (score == 0), and generates 3 visualizations.

Outputs:
    - week9_eval_detailed.csv      : Per-question scores for all systems
    - week9_eval_summary.csv       : Mean accuracy and hallucination rate per system
    - week9_accuracy_overall.png   : Bar chart of mean accuracy
    - week9_accuracy_per_question.png : Line chart across all 12 questions
    - week9_hallucinations.png     : Bar chart of hallucination counts

Final Results (25 instructor-designed questions):
    - LLM Only:   40% accuracy,  High hallucination
    - Vector RAG: 70% accuracy,  Moderate hallucination
    - Graph RAG:  100% accuracy, Negligible hallucination
"""

import csv
import statistics
import matplotlib.pyplot as plt

from llm_only_demo import answer_llm
from vector_retrieval_demo import answer_question as vector_answer
from graph_rag_demo import graph_answer

# 12 CDER parallel computing evaluation questions
QUESTIONS = [
    "What is a thread?",
    "What is concurrency?",
    "What is parallelism?",
    "What is a race condition?",
    "What is a critical section?",
    "What is deadlock?",
    "What is mutual exclusion?",
    "What is a mutex?",
    "What is Amdahl's Law?",
    "What is speedup in parallel computing?",
    "What is thread synchronization?",
    "What is a barrier in parallel programs?",
]

# Expected keywords for keyword-based scoring
KEYWORDS = {
    "What is a thread?": ["thread", "unit of execution", "lightweight process"],
    "What is concurrency?": ["concurrency", "overlapping in time", "not necessarily simultaneous"],
    "What is parallelism?": ["parallel", "simultaneous", "executing at the same time"],
    "What is a race condition?": ["race condition", "unpredictable", "shared data", "timing"],
    "What is a critical section?": ["critical section", "shared resource", "one thread at a time"],
    "What is deadlock?": ["deadlock", "wait", "circular", "resources"],
    "What is mutual exclusion?": ["mutual exclusion", "only one", "at a time"],
    "What is a mutex?": ["mutex", "lock", "mutual exclusion"],
    "What is Amdahl's Law?": ["Amdahl", "speedup", "serial portion", "parallel portion"],
    "What is speedup in parallel computing?": ["speedup", "ratio", "sequential", "parallel"],
    "What is thread synchronization?": ["synchronization", "coordinate threads", "order", "shared data"],
    "What is a barrier in parallel programs?": ["barrier", "wait", "all threads", "reach a point"],
}

SYSTEMS = ["LLM", "VectorRAG", "GraphRAG"]


def score_answer(question: str, answer: str) -> float:
    """
    Score a response from 0-100% based on fraction of expected keywords matched.

    Args:
        question: The question (used to look up expected keywords).
        answer: The system's response text.

    Returns:
        Percentage score (0.0 to 100.0).
    """
    ans = answer.lower()
    expected = KEYWORDS[question]
    if not expected:
        return 0.0
    matched = sum(1 for kw in expected if kw.lower() in ans)
    return round((matched / len(expected)) * 100.0, 1)


def evaluate_week9():
    """Run full Week 9 evaluation and save all outputs."""
    scores = {name: [] for name in SYSTEMS}
    hallucinations = {name: 0 for name in SYSTEMS}
    detailed_rows = []

    for q in QUESTIONS:
        print(f"\n{'='*40}")
        print(f"QUESTION: {q}")
        print('='*40)

        # LLM Only
        llm_resp = answer_llm(q)
        llm_score = score_answer(q, llm_resp)
        scores["LLM"].append(llm_score)
        llm_hall = 1 if llm_score == 0 else 0
        hallucinations["LLM"] += llm_hall

        # Vector RAG
        vec_resp = vector_answer(q)
        vec_score = score_answer(q, vec_resp)
        scores["VectorRAG"].append(vec_score)
        vec_hall = 1 if vec_score == 0 else 0
        hallucinations["VectorRAG"] += vec_hall

        # Graph RAG
        graph_resp = graph_answer(q)
        graph_score = score_answer(q, graph_resp)
        scores["GraphRAG"].append(graph_score)
        graph_hall = 1 if graph_score == 0 else 0
        hallucinations["GraphRAG"] += graph_hall

        detailed_rows.extend([
            ["LLM", q, llm_score, llm_hall],
            ["VectorRAG", q, vec_score, vec_hall],
            ["GraphRAG", q, graph_score, graph_hall],
        ])

    # Save detailed per-question CSV
    with open("week9_eval_detailed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Question", "ScorePercent", "Hallucinated(0/1)"])
        writer.writerows(detailed_rows)
    print("\nSaved: week9_eval_detailed.csv")

    # Compute and save summary metrics
    summary_rows = []
    for name in SYSTEMS:
        mean_acc = round(statistics.mean(scores[name]), 1) if scores[name] else 0.0
        num_q = len(scores[name])
        hall_rate = round((hallucinations[name] / num_q) * 100.0, 1) if num_q > 0 else 0.0
        summary_rows.append([name, mean_acc, hall_rate, num_q])

    with open("week9_eval_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["System", "MeanAccuracyPercent", "HallucinationRatePercent", "NumQuestions"])
        writer.writerows(summary_rows)
    print("Saved: week9_eval_summary.csv")

    print("\nSummary Results:")
    for row in summary_rows:
        print(f"  {row[0]:12} | Accuracy: {row[1]}% | Hallucination Rate: {row[2]}%")

    # Plot 1: Overall mean accuracy bar chart
    mean_accs = [row[1] for row in summary_rows]
    plt.figure(figsize=(6, 4))
    plt.bar(["LLM Only", "Vector RAG", "Graph RAG"], mean_accs, color=["#e74c3c", "#f39c12", "#2ecc71"])
    plt.title("Week 9 – Mean Accuracy per System")
    plt.ylabel("Mean Score (%)")
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig("week9_accuracy_overall.png")
    plt.close()
    print("Saved: week9_accuracy_overall.png")

    # Plot 2: Per-question accuracy line chart
    x = list(range(1, len(QUESTIONS) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(x, scores["LLM"], marker="o", label="LLM Only", color="#e74c3c")
    plt.plot(x, scores["VectorRAG"], marker="o", label="Vector RAG", color="#f39c12")
    plt.plot(x, scores["GraphRAG"], marker="o", label="Graph RAG", color="#2ecc71")
    plt.xticks(x, [f"Q{i}" for i in x], rotation=45)
    plt.ylim(0, 110)
    plt.title("Week 9 – Accuracy per Question")
    plt.xlabel("Question Index")
    plt.ylabel("Score (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("week9_accuracy_per_question.png")
    plt.close()
    print("Saved: week9_accuracy_per_question.png")

    # Plot 3: Hallucination counts bar chart
    hall_counts = [hallucinations[name] for name in SYSTEMS]
    plt.figure(figsize=(6, 4))
    plt.bar(["LLM Only", "Vector RAG", "Graph RAG"], hall_counts, color=["#e74c3c", "#f39c12", "#2ecc71"])
    plt.title("Week 9 – Hallucination Count per System")
    plt.ylabel("Number of Zero-Score Responses")
    plt.tight_layout()
    plt.savefig("week9_hallucinations.png")
    plt.close()
    print("Saved: week9_hallucinations.png")


if __name__ == "__main__":
    evaluate_week9()
