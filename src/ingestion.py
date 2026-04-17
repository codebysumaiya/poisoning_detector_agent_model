
import os
import glob
from tqdm import tqdm
from colorama import Fore, Style, init

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

init(autoreset=True)

# ── Config ────────────────────────────────────────────────────────
PDF_DIR        = "data/raw_pdfs"
FAISS_INDEX    = "data/faiss_index"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"   # fast on CPU
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 50


def load_pdfs(pdf_dir: str) -> list:
    """Load all PDFs from a directory using PyMuPDF."""
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{pdf_dir}'. "
                                "Run generate_pdfs.py first.")

    all_docs = []
    print(f"{Fore.CYAN}Loading PDFs from '{pdf_dir}'...{Style.RESET_ALL}")
    for pdf_path in tqdm(pdf_files, desc="Loading"):
        loader = PyMuPDFLoader(pdf_path)
        docs   = loader.load()
        # Attach clean filename to metadata
        for doc in docs:
            doc.metadata["source"] = os.path.basename(pdf_path)
            doc.metadata["poisoned"] = False   # default — clean
        all_docs.extend(docs)
        print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {os.path.basename(pdf_path)} "
              f"({len(docs)} pages)")

    print(f"\nTotal pages loaded: {len(all_docs)}\n")
    return all_docs


def chunk_documents(docs: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"{Fore.CYAN}Chunking complete:{Style.RESET_ALL} "
          f"{len(docs)} pages → {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")
    return chunks


def build_vector_store(chunks: list, save_path: str) -> FAISS:
    """Create FAISS vector store and save locally."""
    print(f"{Fore.CYAN}Loading embedding model:{Style.RESET_ALL} {EMBED_MODEL}")
    print("  (First run downloads ~90 MB — subsequent runs use cache)\n")

    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"{Fore.CYAN}Building FAISS index from {len(chunks)} chunks...{Style.RESET_ALL}")
    vectorstore = FAISS.from_documents(chunks, embedder)

    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"{Fore.GREEN}✓ Vector store saved to '{save_path}'{Style.RESET_ALL}\n")
    return vectorstore


def load_vector_store(save_path: str) -> FAISS:
    """Load an existing FAISS index from disk."""
    embedder = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        save_path, embedder, allow_dangerous_deserialization=True
    )
    print(f"{Fore.GREEN}✓ Vector store loaded from '{save_path}'{Style.RESET_ALL}")
    return vectorstore


def run_ingestion(pdf_dir: str = PDF_DIR,
                  index_path: str = FAISS_INDEX,
                  force_rebuild: bool = False):
    """Full ingestion pipeline. Returns (chunks, vectorstore)."""
    if os.path.exists(index_path) and not force_rebuild:
        print(f"{Fore.YELLOW}Existing index found at '{index_path}'.{Style.RESET_ALL}")
        print("Loading from disk (pass force_rebuild=True to rebuild).\n")
        vs = load_vector_store(index_path)
        return None, vs

    docs   = load_pdfs(pdf_dir)
    chunks = chunk_documents(docs)
    vs     = build_vector_store(chunks, index_path)
    return chunks, vs


# ── Standalone entry point ────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  PHASE 1 & 2 — PDF Ingestion & Vector Store Builder")
    print("=" * 55 + "\n")

    chunks, vs = run_ingestion(force_rebuild=True)

    # Quick sanity check
    print(f"\n{Fore.CYAN}Sanity check — similarity search:{Style.RESET_ALL}")
    results = vs.similarity_search("What is machine learning?", k=3)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r.metadata['source']}  →  "
              f"{r.page_content[:80].strip()}...")

    print(f"\n{Fore.GREEN}Ingestion complete!{Style.RESET_ALL}")