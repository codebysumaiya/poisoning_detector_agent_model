"""
src/agent_monitor.py
Phase 4 — Agentic Monitor: validates RAG responses and triggers refinement.

The agent checks for:
  1. Empty / too-short answers
  2. Refusal phrases ("I cannot", "I don't know")
  3. Low semantic similarity between answer and retrieved context
  4. Poisoned source detection
  5. Contradiction detection across sources

If validation fails the agent re-queries with an improved prompt (up to MAX_RETRIES).
"""

import re
from dataclasses import dataclass, field
from colorama import Fore, Style, init

init(autoreset=True)

# ── Config ────────────────────────────────────────────────────────
MIN_ANSWER_WORDS  = 15
MAX_RETRIES       = 2
SIMILARITY_THRESH = 0.25    # cosine similarity floor (embedding space)

REFUSAL_PATTERNS = [
    r"i (cannot|can't|don't|do not) (find|know|have|provide)",
    r"i('m| am) (not sure|unable)",
    r"no (relevant|specific) information",
    r"not (mentioned|covered|discussed|found) in",
    r"(insufficient|no) (context|information)",
]


@dataclass
class ValidationReport:
    passed:         bool  = True
    issues:         list  = field(default_factory=list)
    poisoned_srcs:  list  = field(default_factory=list)
    retry_count:    int   = 0
    final_answer:   str   = ""
    refined:        bool  = False


# ── Validation helpers ────────────────────────────────────────────

def _check_length(answer: str) -> str | None:
    words = answer.split()
    if len(words) < MIN_ANSWER_WORDS:
        return f"answer_too_short ({len(words)} words, min={MIN_ANSWER_WORDS})"
    return None


def _check_refusal(answer: str) -> str | None:
    lower = answer.lower()
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, lower):
            return f"refusal_detected ({pattern})"
    return None


def _check_poisoned_sources(sources: list) -> list:
    return [s["source"] for s in sources if s.get("poisoned", False)]


def _check_semantic_similarity(answer: str, contexts: list) -> str | None:
    """
    Lightweight keyword overlap proxy for cosine similarity (no GPU needed).
    In production replace with sentence-transformers cosine similarity.
    """
    if not contexts:
        return "no_context_retrieved"

    answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
    context_text = " ".join(contexts).lower()
    ctx_words    = set(re.findall(r'\b\w{4,}\b', context_text))

    if not answer_words:
        return "empty_answer"

    overlap = len(answer_words & ctx_words) / len(answer_words)
    if overlap < SIMILARITY_THRESH:
        return f"low_context_overlap ({overlap:.2f} < {SIMILARITY_THRESH})"
    return None


def _check_contradiction(contexts: list) -> str | None:
    """
    Simple heuristic: look for antonym pairs across chunks.
    Real implementation would use NLI model.
    """
    antonym_pairs = [
        ("increases", "decreases"),
        ("improves",  "worsens"),
        ("reduces",   "amplifies"),
        ("prevents",  "causes"),
        ("true",      "false"),
    ]
    combined = " ".join(contexts).lower()
    for a, b in antonym_pairs:
        if a in combined and b in combined:
            return f"possible_contradiction ({a!r} vs {b!r}) in context"
    return None


# ── Main agent ────────────────────────────────────────────────────

class AgentMonitor:
    """
    Wraps a RAG chain with autonomous validation and refinement logic.
    """

    def __init__(self, rag_chain):
        self.chain = rag_chain

    def _validate(self, result: dict) -> ValidationReport:
        report = ValidationReport(final_answer=result["answer"])

        checks = [
            _check_length(result["answer"]),
            _check_refusal(result["answer"]),
            _check_semantic_similarity(result["answer"], result["contexts"]),
            _check_contradiction(result["contexts"]),
        ]
        report.issues         = [c for c in checks if c is not None]
        report.poisoned_srcs  = _check_poisoned_sources(result["sources"])
        report.passed         = len(report.issues) == 0

        if report.poisoned_srcs:
            report.issues.append(
                f"poisoned_sources_detected: {report.poisoned_srcs}"
            )
            report.passed = False

        return report

    def _refine_query(self, original_question: str, issues: list,
                      attempt: int) -> str:
        """Generate an improved prompt based on failure reasons."""
        prefixes = {
            0: "Answer in detail, citing specific facts: ",
            1: "Provide a comprehensive and specific answer to: ",
        }
        prefix = prefixes.get(attempt, "Answer thoroughly: ")

        # Add domain hints based on detected issues
        if any("refusal" in i for i in issues):
            prefix = "Based strictly on the retrieved documents, answer: "
        if any("poisoned" in i for i in issues):
            prefix = "Answer using only verified, trusted document sections: "

        return prefix + original_question

    def run(self, question: str) -> dict:
        """
        Execute the monitored RAG pipeline.

        Returns enriched result dict with 'validation' key.
        """
        from src.rag_pipeline import query_rag, print_result

        print(f"\n{Fore.CYAN}[Agent] Processing query:{Style.RESET_ALL} {question}")

        result = query_rag(self.chain, question)
        report = self._validate(result)

        attempt = 0
        while not report.passed and attempt < MAX_RETRIES:
            attempt += 1
            report.retry_count = attempt

            print(f"  {Fore.YELLOW}[Agent] Validation FAILED "
                  f"(attempt {attempt}/{MAX_RETRIES}):{Style.RESET_ALL}")
            for issue in report.issues:
                print(f"    ✗ {issue}")

            refined_q = self._refine_query(question, report.issues, attempt - 1)
            print(f"  {Fore.CYAN}[Agent] Refining query:{Style.RESET_ALL} {refined_q}")

            result    = query_rag(self.chain, refined_q)
            report    = self._validate(result)
            report.retry_count = attempt
            report.refined     = True

        report.final_answer = result["answer"]

        # Print validation summary
        if report.passed:
            status = f"{Fore.GREEN}✓ PASSED{Style.RESET_ALL}"
        else:
            status = f"{Fore.RED}✗ FAILED after {MAX_RETRIES} retries{Style.RESET_ALL}"

        print(f"  [Agent] Validation: {status}")
        if report.poisoned_srcs:
            print(f"  {Fore.RED}[Agent] WARNING — Poisoned sources in context: "
                  f"{report.poisoned_srcs}{Style.RESET_ALL}")
        if report.refined:
            print(f"  {Fore.BLUE}[Agent] Response was refined "
                  f"({report.retry_count} attempt(s)){Style.RESET_ALL}")

        result["validation"] = {
            "passed":        report.passed,
            "issues":        report.issues,
            "poisoned_srcs": report.poisoned_srcs,
            "retry_count":   report.retry_count,
            "refined":       report.refined,
        }
        return result


# ── Standalone entry point ────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from src.ingestion    import load_vector_store, FAISS_INDEX
    from src.rag_pipeline import build_rag_chain, print_result

    print("=" * 55)
    print("  PHASE 4 — Agentic Monitor")
    print("=" * 55 + "\n")

    vs      = load_vector_store(FAISS_INDEX)
    chain   = build_rag_chain(vs)
    monitor = AgentMonitor(chain)

    questions = [
        "What is gradient descent and how does it work?",
        "Explain the difference between FAISS and ChromaDB.",
        "What defenses exist against backdoor attacks in RAG?",
        "How does label flipping affect model faithfulness?",
    ]

    for q in questions:
        result = monitor.run(q)
        print_result(result)
        v = result["validation"]
        print(f"Validation summary → passed={v['passed']} | "
              f"retries={v['retry_count']} | "
              f"issues={v['issues']}\n")