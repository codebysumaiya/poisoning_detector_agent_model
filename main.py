
import argparse
import sys
import os
from colorama import Fore, Style, init

init(autoreset=True)

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║  Agentic RAG Poisoning Framework                         ║
║  Autonomous Decision-Making LLM Agent                   ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""

DEMO_QUESTIONS = [
    "What is machine learning and what are its main types?",
    "How does FAISS handle similarity search?",
    "What is a backdoor attack and how does it affect RAG?",
    "What metrics does RAGAS use to evaluate RAG systems?",
    "Explain the difference between overfitting and underfitting.",
]


def phase_ingestion(rebuild: bool = False):
    print(f"\n{Fore.CYAN}{'─'*55}")
    print("  PHASE 1 & 2 — Ingestion & Vector Store")
    print(f"{'─'*55}{Style.RESET_ALL}\n")

    from src.ingestion import run_ingestion
    chunks, vs = run_ingestion(force_rebuild=rebuild)
    return chunks, vs


def phase_rag(vs):
    print(f"\n{Fore.CYAN}{'─'*55}")
    print("  PHASE 3 — RAG Pipeline")
    print(f"{'─'*55}{Style.RESET_ALL}\n")

    from src.rag_pipeline import build_rag_chain, query_rag, print_result
    chain = build_rag_chain(vs)

    print("Running 2 sample queries...\n")
    for q in DEMO_QUESTIONS[:2]:
        result = query_rag(chain, q)
        print_result(result)
    return chain


def phase_agent(chain):
    print(f"\n{Fore.CYAN}{'─'*55}")
    print("  PHASE 4 — Agentic Monitor")
    print(f"{'─'*55}{Style.RESET_ALL}\n")

    from src.agent_monitor import AgentMonitor
    from src.rag_pipeline  import print_result

    monitor = AgentMonitor(chain)
    for q in DEMO_QUESTIONS[2:4]:
        result = monitor.run(q)
        print_result(result)
        v = result["validation"]
        print(f"  Validation → passed={v['passed']} | "
              f"retries={v['retry_count']} | issues={v['issues']}\n")
    return monitor


def phase_poison():
    print(f"\n{Fore.CYAN}{'─'*55}")
    print("  PHASE 5 — Data Poisoning Demo")
    print(f"{'─'*55}{Style.RESET_ALL}\n")

    from src.ingestion import load_pdfs, chunk_documents, PDF_DIR
    from src.poisoning  import apply_combined_attack

    docs   = load_pdfs(PDF_DIR)
    chunks = chunk_documents(docs)

    poisoned = apply_combined_attack(
        chunks,
        flip_ratio=0.10,
        n_backdoors=3,
        n_noise=10,
        semantic_ratio=0.05,
    )

    # Show 2 sample poisoned chunks
    seen = 0
    for c in poisoned:
        if c.metadata.get("poisoned") and seen < 2:
            print(f"  {Fore.RED}[{c.metadata['attack_type']}]{Style.RESET_ALL} "
                  f"{c.metadata['source']}")
            print(f"  {c.page_content[:180]}...\n")
            seen += 1

    return poisoned


def phase_eval():
    print(f"\n{Fore.CYAN}{'─'*55}")
    print("  PHASE 6 — Evaluation")
    print(f"{'─'*55}{Style.RESET_ALL}\n")

    # Import and run the evaluation module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluation", os.path.join("src", "evaluation.py")
    )
    eval_mod = importlib.util.load_from_spec(spec)
    spec.loader.exec_module(eval_mod)


def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="Agentic RAG Poisoning Framework")
    parser.add_argument("--phase",   type=str, default="all",
                        choices=["all", "ingestion", "rag", "agent", "poison", "eval"])
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of vector store")
    args = parser.parse_args()

    if args.phase in ("all", "ingestion"):
        chunks, vs = phase_ingestion(rebuild=args.rebuild)

    if args.phase in ("all", "rag", "agent"):
        if args.phase != "all":
            from src.ingestion import run_ingestion
            _, vs = run_ingestion()
        chain = phase_rag(vs)

    if args.phase in ("all", "agent"):
        monitor = phase_agent(chain)

    if args.phase in ("all", "poison"):
        poisoned_chunks = phase_poison()

    if args.phase in ("all", "eval"):
        # Run evaluation as a subprocess to avoid import conflicts
        import subprocess
        print(f"\n{Fore.CYAN}Running evaluation module...{Style.RESET_ALL}")
        subprocess.run([sys.executable, "src/evaluation.py"], check=True)

    print(f"\n{Fore.GREEN}{'═'*55}")
    print("  Pipeline complete!")
    print(f"{'═'*55}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()