

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from colorama import Fore, Style, init

load_dotenv()
init(autoreset=True)

# ── Config ────────────────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOP_K_DOCS   = 4

RAG_PROMPT_TEMPLATE = """You are a precise and helpful assistant.
Use ONLY the following retrieved context to answer the question.
If the context does not contain the answer, say "I cannot find this information in the provided documents."
Do NOT make up information.

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)

def build_rag_chain(vectorstore: FAISS) -> RetrievalQA:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")

    print(f"{Fore.CYAN}Connecting to Groq API (model: {GROQ_MODEL})...{Style.RESET_ALL}")

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
        max_tokens=512,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_DOCS},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    print(f"{Fore.GREEN}✓ RAG chain ready (Groq){Style.RESET_ALL}\n")
    return chain

def query_rag(chain: RetrievalQA, question: str) -> dict:
    raw      = chain.invoke({"query": question})
    answer   = raw["result"].strip()
    src_docs = raw["source_documents"]

    sources  = [
        {
            "source":   doc.metadata.get("source", "unknown"),
            "poisoned": doc.metadata.get("poisoned", False),
            "snippet":  doc.page_content[:150].strip(),
        }
        for doc in src_docs
    ]
    contexts = [doc.page_content for doc in src_docs]

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources,
        "contexts": contexts,
    }

def print_result(result: dict) -> None:
    print(f"\n{'─'*55}")
    print(f"{Fore.YELLOW}Question:{Style.RESET_ALL} {result['question']}")
    print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL}\n{result['answer']}")
    print(f"\n{Fore.CYAN}Sources ({len(result['sources'])}):{Style.RESET_ALL}")
    for i, s in enumerate(result["sources"], 1):
        poison_tag = f" {Fore.RED}[POISONED]{Style.RESET_ALL}" if s["poisoned"] else ""
        print(f"  [{i}] {s['source']}{poison_tag}")
        print(f"       {s['snippet']}...")
    print(f"{'─'*55}\n")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.ingestion import load_vector_store, FAISS_INDEX

    print("=" * 55)
    print("  PHASE 3 — RAG Pipeline (Groq API)")
    print("=" * 55 + "\n")

    vs    = load_vector_store(FAISS_INDEX)
    chain = build_rag_chain(vs)

    test_questions = [
        "What is supervised learning?",
        "How does FAISS store embeddings?",
        "What are backdoor attacks in RAG systems?",
        "What evaluation metrics does RAGAS use?",
    ]

    for q in test_questions:
        result = query_rag(chain, q)
        print_result(result)