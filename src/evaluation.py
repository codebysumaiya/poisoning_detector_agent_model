

import os
import sys
import json
from datetime import datetime
from colorama import Fore, Style, init

init(autoreset=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Evaluation question bank with ground truths ───────────────────
EVAL_DATASET = [
    {
        "question":     "What is supervised learning?",
        "ground_truth": (
            "Supervised learning is a type of machine learning where the algorithm "
            "learns from labeled training data consisting of input-output pairs. "
            "Common algorithms include Linear Regression, Decision Trees, and Neural Networks."
        ),
    },
    {
        "question":     "What is FAISS used for?",
        "ground_truth": (
            "FAISS (Facebook AI Similarity Search) is an open-source library used for "
            "efficient similarity search and clustering of dense vectors. "
            "It supports both CPU and GPU operation and can handle millions of vectors."
        ),
    },
    {
        "question":     "What are backdoor attacks in RAG systems?",
        "ground_truth": (
            "Backdoor attacks in RAG systems involve injecting chunks containing a trigger phrase "
            "and malicious payload into the document store. When the trigger appears in a query, "
            "the system returns attacker-controlled content."
        ),
    },
    {
        "question":     "What is the all-MiniLM-L6-v2 embedding model?",
        "ground_truth": (
            "all-MiniLM-L6-v2 is a sentence embedding model that produces 384-dimensional vectors. "
            "It is very fast and well suited for CPU deployment, making it the recommended "
            "embedding model for RAG systems running without GPU."
        ),
    },
    {
        "question":     "What evaluation metrics does RAGAS use?",
        "ground_truth": (
            "RAGAS measures faithfulness (answer grounded in context), "
            "answer relevancy (answer relevant to question), "
            "context recall (context covers ground truth), "
            "and context precision (retrieved chunks are relevant). "
            "Scores range from 0 to 1, with higher being better."
        ),
    },
    {
        "question":     "What is gradient descent?",
        "ground_truth": (
            "Gradient descent is an optimization algorithm used in machine learning to minimize "
            "the loss function during model training. It computes gradients and updates model "
            "weights iteratively in the direction that reduces the loss."
        ),
    },
]


def collect_results(chain, monitor, questions_with_gt: list) -> list:
    """
    Run all evaluation questions through the monitored RAG chain.
    Returns list of result dicts ready for RAGAS.
    """
    from src.rag_pipeline import query_rag

    records = []
    for item in questions_with_gt:
        q  = item["question"]
        gt = item["ground_truth"]

        print(f"  Querying: {q[:60]}...")
        result = monitor.run(q)

        records.append({
            "question":     q,
            "answer":       result["answer"],
            "contexts":     result["contexts"],
            "ground_truth": gt,
            "validation":   result.get("validation", {}),
        })
    return records


def run_ragas_evaluation(records: list, label: str = "system") -> dict:
    """
    Run RAGAS metrics on a set of records.
    Falls back to keyword-overlap metrics if RAGAS unavailable.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset

        dataset = Dataset.from_list([
            {
                "question":     r["question"],
                "answer":       r["answer"],
                "contexts":     r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in records
        ])

        print(f"\n  {Fore.CYAN}Running RAGAS evaluation for: {label}{Style.RESET_ALL}")
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy,
                     context_recall, context_precision],
        )
        return {
            "faithfulness":       float(scores["faithfulness"]),
            "answer_relevancy":   float(scores["answer_relevancy"]),
            "context_recall":     float(scores["context_recall"]),
            "context_precision":  float(scores["context_precision"]),
        }

    except Exception as e:
        print(f"  {Fore.YELLOW}RAGAS not available ({e}). "
              f"Using keyword-overlap proxy metrics.{Style.RESET_ALL}")
        return _proxy_metrics(records)


def _proxy_metrics(records: list) -> dict:
    """
    Lightweight proxy metrics using keyword overlap.
    Used when RAGAS is unavailable or for quick checks.
    """
    import re

    def word_set(text: str) -> set:
        return set(re.findall(r'\b\w{4,}\b', text.lower()))

    faithfulness_scores      = []
    answer_relevancy_scores  = []
    context_recall_scores    = []
    context_precision_scores = []

    for r in records:
        ans_words  = word_set(r["answer"])
        ctx_words  = word_set(" ".join(r["contexts"]))
        gt_words   = word_set(r["ground_truth"])
        q_words    = word_set(r["question"])

        # Faithfulness: how much of the answer appears in context
        faith = (len(ans_words & ctx_words) / len(ans_words)
                 if ans_words else 0.0)
        faithfulness_scores.append(faith)

        # Answer relevancy: overlap of answer with question terms
        rel = (len(ans_words & q_words) / len(q_words)
               if q_words else 0.0)
        answer_relevancy_scores.append(min(rel * 2, 1.0))  # scale up

        # Context recall: how much of ground truth is in context
        rec = (len(gt_words & ctx_words) / len(gt_words)
               if gt_words else 0.0)
        context_recall_scores.append(rec)

        # Context precision: how much of context overlaps with GT
        prec = (len(ctx_words & gt_words) / len(ctx_words)
                if ctx_words else 0.0)
        context_precision_scores.append(min(prec * 3, 1.0))  # scale up

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    return {
        "faithfulness":      mean(faithfulness_scores),
        "answer_relevancy":  mean(answer_relevancy_scores),
        "context_recall":    mean(context_recall_scores),
        "context_precision": mean(context_precision_scores),
    }


def print_score_table(scores: dict, label: str) -> None:
    """
    Print metric scores as a clean formatted table in the terminal.
    Color-coded: Green >= 0.75, Yellow >= 0.50, Red < 0.50
    Includes a visual progress bar for each metric.
    """
    print(f"\n{'─'*50}")
    print(f"  📊  Scores for: {label}")
    print(f"{'─'*50}")
    print(f"  {'Metric':<24} {'Score':>6}   {'Bar'}")
    print(f"  {'─'*44}")

    for metric, value in scores.items():
        if value >= 0.75:
            color = Fore.GREEN
        elif value >= 0.50:
            color = Fore.YELLOW
        else:
            color = Fore.RED

        filled = int(value * 10)
        bar    = "█" * filled + "░" * (10 - filled)

        print(f"  {metric:<24} {color}{value:>6.3f}   {bar}{Style.RESET_ALL}")

    print(f"{'─'*50}\n")


def print_comparison(clean_scores: dict,
                     poisoned_scores: dict,
                     attack_name: str) -> dict:
    """Print a formatted comparison table."""
    print(f"\n{'═'*62}")
    print(f"  Evaluation: Clean RAG  vs  Poisoned RAG [{attack_name}]")
    print(f"{'═'*62}")
    print(f"  {'Metric':<22} {'Clean':>8}  {'Poisoned':>10}  {'Drop':>8}")
    print(f"  {'─'*54}")

    comparison = {}
    for metric in ["faithfulness", "answer_relevancy",
                   "context_recall", "context_precision"]:
        c = clean_scores.get(metric, 0.0)
        p = poisoned_scores.get(metric, 0.0)
        d = c - p

        color = Fore.GREEN if d < 0.05 else Fore.YELLOW if d < 0.15 else Fore.RED
        arrow = "▼" if d > 0 else "▲" if d < 0 else "─"

        print(f"  {metric:<22} {c:>8.3f}  {p:>10.3f}  "
              f"{color}{arrow} {abs(d):>5.3f}{Style.RESET_ALL}")
        comparison[metric] = {"clean": c, "poisoned": p, "drop": d}

    print(f"{'═'*62}\n")
    return comparison


def save_results(results: dict, path: str = "results/evaluation_report.json") -> None:
    """Persist evaluation results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results["timestamp"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"{Fore.GREEN}✓ Results saved → {path}{Style.RESET_ALL}")


# ── Standalone entry point ────────────────────────────────────────
if __name__ == "__main__":
    from src.ingestion    import run_ingestion, load_pdfs, chunk_documents, PDF_DIR, FAISS_INDEX
    from src.rag_pipeline import build_rag_chain
    from src.agent_monitor import AgentMonitor
    from src.poisoning    import (label_flipping_attack, backdoor_trigger_attack,
                                   noise_injection_attack, semantic_poisoning_attack,
                                   rebuild_vector_store_from_chunks)

    print("=" * 62)
    print("  PHASE 6 — RAGAS Evaluation: Clean vs Poisoned RAG")
    print("=" * 62 + "\n")

    # ── Step 1: load clean knowledge base ─────────────────────────
    print(f"{Fore.CYAN}Step 1: Loading clean vector store...{Style.RESET_ALL}")
    _, clean_vs = run_ingestion()
    clean_chain   = build_rag_chain(clean_vs)
    clean_monitor = AgentMonitor(clean_chain)

    # ── Step 2: evaluate clean system ─────────────────────────────
    print(f"\n{Fore.CYAN}Step 2: Evaluating CLEAN system...{Style.RESET_ALL}")
    clean_records = collect_results(clean_chain, clean_monitor, EVAL_DATASET)
    clean_scores  = run_ragas_evaluation(clean_records, "CLEAN")

    # ── Print clean score table ────────────────────────────────────
    print_score_table(clean_scores, "CLEAN RAG")

    # ── Step 3: apply poisoning attacks and evaluate each ─────────
    docs   = load_pdfs(PDF_DIR)
    chunks = chunk_documents(docs)

    attacks = {
        "Label Flipping":   label_flipping_attack(chunks, 0.20),
        "Backdoor Trigger": backdoor_trigger_attack(chunks, n_injections=8),
        "Noise Injection":  noise_injection_attack(chunks, n_noise=25),
        "Semantic Poison":  semantic_poisoning_attack(chunks, 0.15),
    }

    all_comparisons = {}
    for attack_name, poisoned_chunks in attacks.items():
        print(f"\n{Fore.RED}Step 3: Evaluating [{attack_name}] attack...{Style.RESET_ALL}")

        # Build poisoned vector store
        poisoned_vs      = rebuild_vector_store_from_chunks(
            poisoned_chunks, f"data/faiss_index_poisoned_{attack_name.replace(' ', '_').lower()}"
        )
        poisoned_chain   = build_rag_chain(poisoned_vs)
        poisoned_monitor = AgentMonitor(poisoned_chain)

        poisoned_records = collect_results(poisoned_chain, poisoned_monitor, EVAL_DATASET)
        poisoned_scores  = run_ragas_evaluation(poisoned_records, attack_name)

        # ── Print poisoned score table ─────────────────────────────
        print_score_table(poisoned_scores, f"POISONED — {attack_name}")

        comparison = print_comparison(clean_scores, poisoned_scores, attack_name)
        all_comparisons[attack_name] = comparison

    # ── Step 4: save all results ───────────────────────────────────
    save_results({
        "clean_scores": clean_scores,
        "attack_comparisons": all_comparisons,
    })

    print(f"\n{Fore.GREEN}Evaluation complete! Check results/evaluation_report.json{Style.RESET_ALL}")