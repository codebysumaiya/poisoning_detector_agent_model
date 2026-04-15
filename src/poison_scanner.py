"""
src/poison_scanner.py
Scans incoming document chunks for poisoning patterns BEFORE they enter FAISS.
This is the real-time outside attack detection layer.

How it works:
  1. Every new chunk is scored against known poisoning signatures
  2. Chunks that score above the threshold are REJECTED
  3. A detailed report is returned explaining why each chunk was flagged
  4. Only clean chunks are allowed into the vector store
"""

import re
from dataclasses import dataclass, field
from colorama import Fore, Style, init

init(autoreset=True)

# ── Scoring thresholds ────────────────────────────────────────────
POISON_SCORE_THRESHOLD = 2   # chunks scoring >= this are rejected


@dataclass
class ChunkScanResult:
    chunk_index:   int
    source:        str
    is_poisoned:   bool
    score:         int
    reasons:       list = field(default_factory=list)
    snippet:       str  = ""


@dataclass
class ScanReport:
    total_chunks:    int  = 0
    clean_chunks:    int  = 0
    rejected_chunks: int  = 0
    attack_types:    list = field(default_factory=list)
    details:         list = field(default_factory=list)

    @property
    def is_clean(self):
        return self.rejected_chunks == 0


# ── Attack signatures ─────────────────────────────────────────────

# 1. Backdoor trigger words — hidden command phrases
BACKDOOR_TRIGGERS = [
    r"\bCONFIDENTIAL\b",
    r"ignore (all |previous )?(instructions|commands|rules)",
    r"system (has been|is) compromised",
    r"override.*security",
    r"output only.*compromised",
    r"this (document|content) supersedes",
    r"ADMIN_COMMAND",
    r"TRIGGER_ACTIVATED",
    r"access (denied|breach|violation)",
    r"disregard (retrieved|previous) context",
]

# 2. Label flipping signatures — factual opposites injected
LABEL_FLIP_PAIRS = [
    ("increases",   "decreases"),
    ("improves",    "worsens"),
    ("reduces",     "amplifies"),
    ("prevents",    "causes"),
    ("accurate",    "inaccurate"),
    ("strengthens", "weakens"),
    ("faster",      "slower"),
    ("higher",      "lower"),
    ("positive",    "negative"),
    ("correct",     "incorrect"),
    ("reduces hallucinations", "increases hallucinations"),
    ("open-source", "proprietary and closed-source"),
]

# 3. Semantic substitution — wrong technical definitions
SEMANTIC_SUBSTITUTIONS = {
    "manual computation processes":     "machine learning",
    "rule-based decision system":       "neural network",
    "shallow statistical analysis":     "deep learning",
    "traditional relational database":  "vector database",
    "keyword indexing":                 "embedding",
    "standalone generative":            "retrieval-augmented",
    "fixed-weight averaging":           "attention mechanism",
    "random search optimization":       "gradient descent",
    "increases hallucinations":         "reduces hallucinations",
    "amplifies errors":                 "backpropagation",
    "diverges from the optimal":        "gradient descent",
}

# 4. Noise injection — off-topic content signatures
NOISE_INDICATORS = [
    r"\b(Eiffel Tower|Paris|France)\b.*\b(capital|city|tourism)\b",
    r"\b(photosynthesis|chloroplast|glucose)\b",
    r"\b(Mount Everest|Himalayas|altitude)\b",
    r"\b(Amazon rainforest|tropical|biodiversity)\b",
    r"\b(Roman|medieval|ancient)\b.*\b(architecture|construction|history)\b",
    r"\b(Pacific Ocean|Atlantic|Indian Ocean)\b.*\b(covers|surface|area)\b",
    r"\b(Shakespeare|sonnets|Hamlet)\b",
    r"\b(speed of light|299,792)\b",
    r"\b(Great Wall|China)\b.*\b(kilometres|miles|stretch)\b",
    r"\b(DNA|deoxyribonucleic)\b.*\b(genetic|hereditary)\b",
]

# 5. Injection / prompt attack patterns
INJECTION_PATTERNS = [
    r"(ignore|disregard|forget) (all |previous |your )?(instructions|rules|training)",
    r"you are now (a |an )?(different|new|other)",
    r"pretend (you are|to be|that)",
    r"act as (a |an )?",
    r"new (persona|role|character|identity)",
    r"jailbreak",
    r"DAN mode",
    r"developer mode",
]


# ── Chunk scanner ─────────────────────────────────────────────────

def scan_chunk(chunk, index: int) -> ChunkScanResult:
    """
    Score a single document chunk against all poisoning signatures.
    Returns a ChunkScanResult with score and reasons.
    """
    text    = chunk.page_content
    source  = chunk.metadata.get("source", f"chunk_{index}")
    lower   = text.lower()
    score   = 0
    reasons = []

    # ── Check 1: Backdoor triggers ──────────────────────────────
    for pattern in BACKDOOR_TRIGGERS:
        if re.search(pattern, text, re.IGNORECASE):
            score   += 3   # high severity
            reasons.append(f"Backdoor trigger detected: '{pattern}'")
            break

    # ── Check 2: Label flipping ─────────────────────────────────
    flip_count = 0
    for word_a, word_b in LABEL_FLIP_PAIRS:
        if re.search(rf'\b{re.escape(word_b)}\b', lower):
            # The "wrong" word appears — possible flip
            flip_count += 1
    if flip_count >= 2:
        score   += 2
        reasons.append(f"Label flipping suspected: {flip_count} antonym substitutions found")

    # ── Check 3: Semantic substitutions ─────────────────────────
    sub_count = 0
    for wrong_term in SEMANTIC_SUBSTITUTIONS:
        if wrong_term.lower() in lower:
            sub_count += 1
            reasons.append(f"Semantic substitution: '{wrong_term}' — known poisoning term")
    if sub_count > 0:
        score += sub_count * 2   # each substitution adds 2 points

    # ── Check 4: Noise injection ─────────────────────────────────
    noise_count = 0
    for pattern in NOISE_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            noise_count += 1
    if noise_count >= 2:
        score   += 2
        reasons.append(f"Noise injection suspected: {noise_count} off-topic indicators")

    # ── Check 5: Prompt injection ────────────────────────────────
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            score   += 3
            reasons.append(f"Prompt injection attempt: '{pattern}'")
            break

    # ── Check 6: Metadata poisoning flag ────────────────────────
    if chunk.metadata.get("poisoned", False):
        score   += 5   # explicitly marked — definitely poisoned
        reasons.append("Chunk explicitly marked as poisoned in metadata")

    is_poisoned = score >= POISON_SCORE_THRESHOLD

    return ChunkScanResult(
        chunk_index = index,
        source      = source,
        is_poisoned = is_poisoned,
        score       = score,
        reasons     = reasons,
        snippet     = text[:120].strip(),
    )


def scan_all_chunks(chunks: list) -> tuple[list, ScanReport]:
    """
    Scan all chunks. Return (clean_chunks, report).
    Only clean chunks are returned for ingestion into FAISS.
    """
    report         = ScanReport(total_chunks=len(chunks))
    clean_chunks   = []
    attack_types   = set()

    print(f"\n{Fore.CYAN}[Scanner] Scanning {len(chunks)} chunks for poisoning...{Style.RESET_ALL}")

    for i, chunk in enumerate(chunks):
        result = scan_chunk(chunk, i)
        report.details.append(result)

        if result.is_poisoned:
            report.rejected_chunks += 1
            # Classify attack type
            for reason in result.reasons:
                if "Backdoor"   in reason: attack_types.add("Backdoor Trigger")
                if "Label flip" in reason: attack_types.add("Label Flipping")
                if "Semantic"   in reason: attack_types.add("Semantic Poison")
                if "Noise"      in reason: attack_types.add("Noise Injection")
                if "Prompt inj" in reason: attack_types.add("Prompt Injection")
                if "metadata"   in reason: attack_types.add("Known Poisoned Chunk")

            print(f"  {Fore.RED}✗ REJECTED chunk {i} from '{result.source}' "
                  f"(score={result.score}){Style.RESET_ALL}")
            for r in result.reasons:
                print(f"      → {r}")
        else:
            clean_chunks.append(chunk)
            report.clean_chunks += 1

    report.attack_types = list(attack_types)

    # Summary
    if report.is_clean:
        print(f"\n{Fore.GREEN}[Scanner] ✓ All {report.total_chunks} chunks are clean.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}[Scanner] ⚠ {report.rejected_chunks}/{report.total_chunks} "
              f"chunks REJECTED.{Style.RESET_ALL}")
        print(f"  Attack types detected: {report.attack_types}")
        print(f"  {report.clean_chunks} clean chunks will be ingested.\n")

    return clean_chunks, report