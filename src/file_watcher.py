"""
src/file_watcher.py
Watches data/raw_pdfs/ for new PDF files dropped in from outside.
When a new file is detected:
  1. Loads and chunks the PDF
  2. Scans every chunk for poisoning
  3. Rejects poisoned chunks
  4. Ingests only clean chunks into FAISS
  5. Updates the app's knowledge base live

Run standalone:
    python src/file_watcher.py

Or import and call start_watching() from your app.
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PDF_DIR      = "data/raw_pdfs"
FAISS_INDEX  = "data/faiss_index"
ALERT_FILE   = "results/attack_alerts.json"
CHECK_INTERVAL = 5   # seconds between checks


def load_alerts() -> list:
    os.makedirs("results", exist_ok=True)
    if os.path.exists(ALERT_FILE):
        try:
            with open(ALERT_FILE) as f:
                return json.load(f)
        except:
            return []
    return []


def save_alert(alert: dict):
    alerts = load_alerts()
    alerts.insert(0, alert)
    alerts = alerts[:50]   # keep last 50 alerts
    with open(ALERT_FILE, "w") as f:
        json.dump(alerts, f, indent=2)


def get_existing_files(pdf_dir: str) -> set:
    """Get set of all PDF filenames currently in the folder."""
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        return set()
    return {f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")}


def process_new_file(pdf_path: str) -> dict:
    """
    Process a newly detected PDF:
    1. Load and chunk it
    2. Scan for poisoning
    3. Ingest clean chunks into existing FAISS index
    Returns an alert dict with results.
    """
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from src.poison_scanner import scan_all_chunks

    filename = os.path.basename(pdf_path)
    print(f"\n{Fore.CYAN}[Watcher] New file detected: {filename}{Style.RESET_ALL}")

    alert = {
        "timestamp":       datetime.now().strftime("%d %b %Y, %H:%M:%S"),
        "filename":        filename,
        "status":          "clean",
        "total_chunks":    0,
        "rejected_chunks": 0,
        "clean_chunks":    0,
        "attack_types":    [],
        "message":         "",
    }

    try:
        # Step 1: Load PDF
        loader = PyMuPDFLoader(pdf_path)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source"]   = filename
            doc.metadata["poisoned"] = False

        # Step 2: Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        print(f"  Loaded {len(docs)} pages → {len(chunks)} chunks")

        # Step 3: Scan for poisoning
        clean_chunks, report = scan_all_chunks(chunks)

        alert["total_chunks"]    = report.total_chunks
        alert["rejected_chunks"] = report.rejected_chunks
        alert["clean_chunks"]    = report.clean_chunks
        alert["attack_types"]    = report.attack_types

        if not report.is_clean:
            alert["status"]  = "poisoned"
            alert["message"] = (
                f"⚠️ Attack detected in '{filename}'. "
                f"{report.rejected_chunks}/{report.total_chunks} chunks rejected. "
                f"Attack types: {', '.join(report.attack_types)}. "
                f"Only {report.clean_chunks} clean chunks were ingested."
            )
            print(f"\n{Fore.RED}[Watcher] ATTACK DETECTED in {filename}!{Style.RESET_ALL}")
            print(f"  {report.rejected_chunks} chunks rejected, "
                  f"{report.clean_chunks} clean chunks ingested.")
        else:
            alert["message"] = (
                f"✅ '{filename}' is clean. "
                f"All {report.clean_chunks} chunks ingested successfully."
            )
            print(f"{Fore.GREEN}[Watcher] File is clean. Ingesting...{Style.RESET_ALL}")

        # Step 4: Ingest clean chunks into FAISS (merge with existing index)
        if clean_chunks:
            embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            if os.path.exists(FAISS_INDEX):
                # Load existing index and add new chunks
                existing_vs = FAISS.load_local(
                    FAISS_INDEX, embedder,
                    allow_dangerous_deserialization=True
                )
                new_vs = FAISS.from_documents(clean_chunks, embedder)
                existing_vs.merge_from(new_vs)
                existing_vs.save_local(FAISS_INDEX)
                print(f"  {Fore.GREEN}✓ Merged {len(clean_chunks)} clean chunks into existing FAISS index{Style.RESET_ALL}")
            else:
                # Create new index
                new_vs = FAISS.from_documents(clean_chunks, embedder)
                new_vs.save_local(FAISS_INDEX)
                print(f"  {Fore.GREEN}✓ Created new FAISS index with {len(clean_chunks)} chunks{Style.RESET_ALL}")

    except Exception as e:
        alert["status"]  = "error"
        alert["message"] = f"Error processing '{filename}': {str(e)}"
        print(f"{Fore.RED}[Watcher] Error: {e}{Style.RESET_ALL}")

    save_alert(alert)
    return alert


def start_watching(pdf_dir: str = PDF_DIR, interval: int = CHECK_INTERVAL):
    """
    Start watching the PDF folder for new files.
    Runs indefinitely — call from a background thread in the app.
    """
    print(f"\n{Fore.CYAN}{'='*55}")
    print(f"  File Watcher Started")
    print(f"  Monitoring: {os.path.abspath(pdf_dir)}")
    print(f"  Check interval: {interval}s")
    print(f"{'='*55}{Style.RESET_ALL}\n")

    known_files = get_existing_files(pdf_dir)
    print(f"[Watcher] Already known files ({len(known_files)}): {known_files}\n")

    while True:
        time.sleep(interval)
        current_files = get_existing_files(pdf_dir)
        new_files     = current_files - known_files

        for filename in new_files:
            pdf_path = os.path.join(pdf_dir, filename)
            process_new_file(pdf_path)
            known_files.add(filename)


if __name__ == "__main__":
    start_watching()