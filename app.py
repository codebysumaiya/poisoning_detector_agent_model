"""
app.py — SentinelRAG
Clean chat interface with autonomous poisoning detection
+ Real-time outside attack detection via file watcher
Run: streamlit run app.py
"""

import os, sys, json, re, threading
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.ingestion     import load_vector_store
from src.rag_pipeline  import build_rag_chain
from src.agent_monitor import AgentMonitor

st.set_page_config(
    page_title="SentinelRAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

*, body { font-family: 'DM Sans', sans-serif !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
html, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main { background: #ffffff !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f8f8f8 !important;
    border-right: 1px solid #ebebeb !important;
    min-width: 260px !important;
    max-width: 280px !important;
    width: 260px !important;
    transform: none !important;
    visibility: visible !important;
    display: flex !important;
    flex-direction: column !important;
    position: relative !important;
    z-index: 999 !important;
}
[data-testid="stSidebar"][aria-expanded="false"],
[data-testid="stSidebar"][aria-expanded="true"] {
    transform: none !important;
    min-width: 260px !important;
    width: 260px !important;
    visibility: visible !important;
    display: flex !important;
}
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[aria-label="Collapse sidebar"],
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
button[title="Collapse sidebar"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #111 !important;
    text-align: left !important;
    padding: 7px 12px !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    width: 100% !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #ebebeb !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #ebebeb !important;
    font-weight: 500 !important;
    border-left: 2px solid #111 !important;
    border-radius: 0 8px 8px 0 !important;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #efefef !important;
    border-radius: 8px !important;
    padding: 6px 8px !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 18px !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: 10px !important; color: #aaa !important; }

/* ── Top nav ── */
.topnav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 13px 28px 12px;
    border-bottom: 1px solid #f0f0f0;
    background: #fff;
    position: sticky; top: 0; z-index: 10;
}
.topnav-left { display: flex; align-items: center; gap: 10px; }
.topnav-icon {
    width: 30px; height: 30px; border-radius: 8px;
    background: #111; color: #fff;
    display: flex; align-items: center; justify-content: center; font-size: 15px;
}
.topnav-title { font-size: 15px; font-weight: 600; color: #111; }
.topnav-sub   { font-size: 12px; color: #bbb; margin-left: 4px; }
.badge-safe  { font-size: 11px; font-weight: 500; padding: 4px 13px; border-radius: 20px; background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
.badge-alert { font-size: 11px; font-weight: 500; padding: 4px 13px; border-radius: 20px; background: #fff5f5; color: #dc2626; border: 1px solid #fecaca; }

/* ── Chat layout ── */
.chat-wrapper {
    max-width: 820px; margin: 0 auto; padding: 16px 24px 8px;
}

/* User bubble */
.msg-user-wrap { display: flex; justify-content: flex-end; margin-bottom: 14px; }
.user-bub {
    background: #f3f3f3; color: #111;
    padding: 10px 16px; max-width: 70%;
    min-width: 60px;          /* ← ADD THIS */
    text-align: center;       /* ← ADD THIS — centers short text like 'hi' */
    border-radius: 18px 18px 4px 18px;
    font-size: 13.5px; line-height: 1.65;
}
.msg-time { font-size: 10px; color: #ccc; margin-top: 4px; text-align: right; }

/* Bot message row */
.msg-bot-wrap {
    display: flex; align-items: flex-start; gap: 12px; margin-bottom: 16px;
}
.bot-av {
    width: 30px; height: 30px; border-radius: 50%;
    background: #111; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0; margin-top: 2px;
}
.bot-content { flex: 1; min-width: 0; }

/* Safe answer */
.safe-response { font-size: 13.5px; color: #111; line-height: 1.75; padding-top: 2px; }

/* Poisoned answer box */
.poisoned-response {
    background: #fff8f0;
    border: 1px solid #fed7aa;
    border-left: 3px solid #f97316;
    border-radius: 10px;
    padding: 12px 15px;
    font-size: 13.5px; color: #7c2d12; line-height: 1.65;
}
.poisoned-label {
    font-size: 11px; font-weight: 600; color: #ea580c;
    margin-bottom: 6px; display: flex; align-items: center; gap: 5px;
}

/* Poison popup */
.poison-popup {
    background: #fff5f5;
    border: 1px solid #fecaca;
    border-left: 3px solid #ef4444;
    border-radius: 10px;
    padding: 12px 15px; margin-top: 8px;
}
.popup-title  { font-weight: 600; color: #dc2626; font-size: 13px; margin-bottom: 7px; }
.popup-issue  { color: #7f1d1d; font-size: 12.5px; margin: 3px 0; }
.popup-src    { color: #7f1d1d; font-size: 12px; margin-top: 6px; }
.popup-meta   { color: #999; font-size: 11px; margin-top: 6px; }
.attack-tag {
    display: inline-block; margin-top: 7px;
    background: #fef2f2; color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
}

/* Outside attack alert banner — added from Doc 2 */
.outside-alert {
    background: #fef2f2; border: 1.5px solid #fca5a5;
    border-left: 4px solid #dc2626; border-radius: 10px;
    padding: 14px 16px; margin: 0 24px 12px;
    max-width: 820px; margin-left: auto; margin-right: auto;
    animation: pulse 2s ease infinite;
}
@keyframes pulse {
    0%, 100% { border-left-color: #dc2626; }
    50%       { border-left-color: #f97316; }
}
.outside-alert-title { font-size: 13px; font-weight: 600; color: #dc2626; margin-bottom: 5px; }
.outside-alert-body  { font-size: 12.5px; color: #7f1d1d; line-height: 1.5; }
.outside-alert-meta  { font-size: 11px; color: #aaa; margin-top: 5px; }
.outside-clean {
    background: #f0fdf4; border: 1px solid #86efac;
    border-left: 4px solid #16a34a; border-radius: 10px;
    padding: 10px 14px; margin: 0 24px 10px;
    max-width: 820px; margin-left: auto; margin-right: auto;
    font-size: 12.5px; color: #166534;
}

/* Sidebar alert items — added from Doc 2 */
.sb-alert-item {
    margin: 3px 10px; padding: 8px 11px;
    border-radius: 8px; font-size: 11px; line-height: 1.4;
}
.sb-alert-poison { background: #fff5f5; border-left: 3px solid #ef4444; color: #7f1d1d; }
.sb-alert-clean  { background: #f0fdf4; border-left: 3px solid #16a34a; color: #166534; }

/* Empty state */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 55vh; text-align: center;
}
.empty-icon  { font-size: 44px; margin-bottom: 14px; opacity: .5; }
.empty-title { font-size: 20px; font-weight: 600; color: #bbb; margin-bottom: 8px; }
.empty-sub   { font-size: 13px; color: #ccc; line-height: 1.7; }

/* Sidebar labels */
.sb-section {
    font-size: 10px; font-weight: 600; color: #aaa;
    text-transform: uppercase; letter-spacing: .07em;
    padding: 14px 16px 5px; margin: 0;
}
.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 14px 14px;
    border-bottom: 1px solid #e5e5e5; margin-bottom: 4px;
}
.sb-logo-icon {
    width: 32px; height: 32px; border-radius: 9px;
    background: #111; display: flex; align-items: center;
    justify-content: center; font-size: 17px;
}
.sb-logo-name { font-size: 15px; font-weight: 600; color: #111; }
.sb-logo-sub  { font-size: 11px; color: #aaa; margin-top: 1px; }

.demo-box { background: #f0f0f0; border-radius: 10px; padding: 12px; margin: 4px 10px 6px; }
.demo-row { display: flex; gap: 8px; align-items: flex-start; margin-bottom: 7px; }
.demo-num {
    min-width: 17px; height: 17px; border-radius: 50%;
    background: #111; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600; margin-top: 1px; flex-shrink: 0;
}
.demo-txt { font-size: 11px; color: #555; line-height: 1.4; }
.decision-log {
    background: #f4f4f4; border-left: 3px solid #111;
    border-radius: 0 8px 8px 0; padding: 9px 12px;
    margin: 3px 10px 8px; font-size: 11px; color: #444;
    line-height: 1.6; white-space: pre-line;
}

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1px solid #e5e5e5 !important;
    background: #fafafa !important;
    max-width: 820px !important;
    margin: 0 auto !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13.5px !important; color: #111 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #bbb !important; }
.stChatInputContainer {
    background: #fff !important;
    padding: 10px 24px 18px !important;
    border-top: 1px solid #f0f0f0 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
(function() {
    function forceOpen() {
        var doc = window.parent.document;
        var sb = doc.querySelector('[data-testid="stSidebar"]');
        if (sb) {
            sb.setAttribute('aria-expanded', 'true');
            sb.style.transform = 'none';
            sb.style.minWidth = '260px';
            sb.style.width = '260px';
            sb.style.visibility = 'visible';
            sb.style.display = 'flex';
        }
        var btns = doc.querySelectorAll(
            '[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"], ' +
            'button[aria-label="Collapse sidebar"], button[aria-label="Open sidebar"], ' +
            'button[aria-label="Close sidebar"]'
        );
        btns.forEach(function(el) { el.style.display = 'none'; el.style.visibility = 'hidden'; });
    }
    forceOpen();
    setInterval(forceOpen, 200);
    new MutationObserver(forceOpen).observe(
        window.parent.document.body, { subtree: true, attributes: true, childList: true }
    );
})();
</script>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────
STORE_PATH   = "data/faiss_index"
HISTORY_FILE = "results/chat_history.json"
ALERT_FILE   = "results/attack_alerts.json"   # ← added from Doc 2
PDF_DIR      = "data/raw_pdfs"                 # ← added from Doc 2

SAMPLE_QS = [
    "What is supervised learning?",
    "What is machine learning?",
    "How does FAISS work?",
    "What are backdoor attacks?",
    "What is gradient descent?",
    "What evaluation metrics does RAGAS use?",
]

AGENT_STEPS = [
    "Question → RAG retrieves top-4 chunks from knowledge base",
    "Agent checks: answer length, refusal phrases, context overlap",
    "Checks for poisoned sources or contradictions in context",
    "Issues found → autonomously retries with refined prompt (max 2×)",
    "Still wrong → 🚨 Marked as POISONED. Attack type identified.",
]

# ── Greeting patterns — never flag these ──────────────────────────
GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|good\s*(morning|afternoon|evening|day))[\s!.,?]*$",
    r"^how are you[\s!.,?]*$",
    r"^what'?s up[\s!.,?]*$",
    r"^(thanks|thank you|ok|okay|sure|alright|bye|goodbye)[\s!.,?]*$",
    r"^(yes|no|maybe|please|please help)[\s!.,?]*$",
]

# ── Direct greeting replies — never touch RAG for these ──────────
GREETING_REPLIES = {
    "hi":              "Hi there! 👋 How can I help you today?",
    "hello":           "Hello! 😊 Feel free to ask me anything about your documents.",
    "hey":             "Hey! What can I help you with?",
    "how are you":     "I'm doing great, thanks for asking! 😊 How can I assist you?",
    "what's up":       "Not much! Ready to help. Ask me anything. 🙂",
    "whats up":        "Not much! Ready to help. Ask me anything. 🙂",
    "thanks":          "You're welcome! Let me know if you have more questions. 😊",
    "thank you":       "Happy to help! Feel free to ask anything else.",
    "ok":              "Sure! Ask me anything you'd like to know.",
    "okay":            "Sure! Ask me anything you'd like to know.",
    "bye":             "Goodbye! Come back anytime. 👋",
    "goodbye":         "Take care! See you next time. 👋",
    "good morning":    "Good morning! ☀️ How can I help you today?",
    "good afternoon":  "Good afternoon! How can I assist you?",
    "good evening":    "Good evening! What can I help you with?",
    "yes":             "Got it! How can I help you further?",
    "no":              "No problem! Let me know if there's anything else I can do.",
    "alright":         "Alright! What would you like to know?",
    "sure":            "Sure! Ask me anything you'd like to know.",
}

def get_greeting_reply(text: str) -> str:
    t = text.strip().lower().rstrip("!.,? ")
    if t in GREETING_REPLIES:
        return GREETING_REPLIES[t]
    for key, reply in GREETING_REPLIES.items():
        if t.startswith(key):
            return reply
    return "Hi! 😊 How can I help you today?"

# ── Only the EXACT full-sentence refusal triggers poisoning flag ──
HARD_REFUSAL_PHRASES = [
    "i cannot find this information in the provided documents",
    "i can't find this information in the provided documents",
    "i cannot answer this question based on the provided documents",
    "i don't have enough information to answer",
    "no relevant information was found",
]

# ──────────────────────────────────────────────────────────────────
# FILE WATCHER — added from Doc 2, runs in background thread
# ──────────────────────────────────────────────────────────────────
def start_file_watcher_background():
    """Start the file watcher in a daemon thread so it doesn't block the app."""
    from src.file_watcher import start_watching
    thread = threading.Thread(target=start_watching, daemon=True)
    thread.start()
    return thread

def get_latest_alerts(n: int = 5) -> list:
    """Load the most recent attack alerts from file."""
    if not os.path.exists(ALERT_FILE):
        return []
    try:
        with open(ALERT_FILE) as f:
            alerts = json.load(f)
        return alerts[:n]
    except:
        return []

# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────
def load_history():
    os.makedirs("results", exist_ok=True)
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_history(h):
    os.makedirs("results", exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(h, f, indent=2)

def is_greeting(text: str) -> bool:
    """Return True if the input is a simple greeting — never flag these."""
    t = text.strip().lower()
    return any(re.match(p, t) for p in GREETING_PATTERNS)

def is_hard_refusal(answer: str) -> bool:
    """
    Only flag when the ENTIRE answer is exactly a known refusal phrase.
    Partial matches inside longer answers are NOT flagged.
    """
    a = answer.strip().lower().rstrip(".")
    return any(a == phrase or a == phrase + "." for phrase in HARD_REFUSAL_PHRASES)

def is_true_poisoning(val: dict, answer: str, question: str) -> bool:
    """
    Returns True ONLY for genuine poisoning:
      - Injection / misleading payload in the answer, OR
      - Factual contradiction detected (label-flip), OR
      - Poisoned source chunks retrieved.
    Does NOT flag: greetings, no-info responses, normal RAG answers.
    """
    if is_greeting(question):
        return False
    if val.get("is_greeting"):
        return False
    if val.get("no_info"):
        return False          # "I cannot find..." is no-info, not an attack
    p_srcs = val.get("poisoned_srcs", [])
    issues = val.get("issues", [])
    passed = val.get("passed", True)
    if p_srcs:
        return True
    # Only flag if a real attack issue is present
    attack_keywords = ["misleading", "contradiction", "injection", "poisoned"]
    if not passed and any(any(kw in i.lower() for kw in attack_keywords) for i in issues):
        return True
    return False

def clean_issues(issues: list) -> list:
    """
    Convert raw technical issue strings into clean human-readable sentences.
    Strips HTML tags, regex patterns, and internal codes.
    """
    clean = []
    for issue in issues:
        # Skip if it contains raw HTML tags
        if re.search(r"<[^>]+>", issue):
            continue
        # Skip raw regex patterns (contain |, ?, grouped parentheses)
        if re.search(r"\(\?|can't\|can't|\|don't\|", issue):
            continue
        if "|" in issue and "(" in issue and ")" in issue:
            continue

        # Map internal codes to friendly messages
        issue = re.sub(r"answer_too_short.*",
                       "Response is too short to be useful", issue)
        issue = re.sub(r"refusal_detected.*",
                       "Response contains a refusal — no useful answer provided", issue)
        issue = re.sub(r"hard_refusal_detected.*",
                       "Response contains no useful information", issue)
        issue = re.sub(r"low_context_overlap.*",
                       "Response is not grounded in the document content", issue)
        issue = re.sub(r"very_low_context_overlap.*",
                       "Response is not grounded in the document content", issue)
        issue = re.sub(r"misleading_content_detected.*",
                       "Response contains suspicious or misleading content", issue)
        issue = re.sub(r"answer_contradicts_context.*",
                       "Response contradicts information in the retrieved documents", issue)
        issue = re.sub(r"possible_contradiction.*",
                       "Contradiction detected between retrieved document chunks", issue)
        issue = re.sub(r"poisoned_sources_detected:.*",
                       "Poisoned document sources were retrieved", issue)
        issue = re.sub(r"poisoned_sources_in_context:.*",
                       "Poisoned document sources were retrieved", issue)
        issue = re.sub(r"no_context_retrieved",
                       "No relevant context was retrieved from the knowledge base", issue)

        issue = issue.strip()
        if issue:
            clean.append(issue)
    return clean

def get_attack_type(p_srcs: list, issues: list) -> str:
    issue_str = " ".join(issues).lower()
    if "noise_document" in str(p_srcs).lower():          return "Noise Injection"
    if "confidential"   in str(p_srcs).lower():          return "Backdoor Trigger"
    if p_srcs:                                            return "Semantic / Label Poisoning"
    if "misleading"     in issue_str:                     return "Backdoor / Injection Attack"
    if "contradicts"    in issue_str:                     return "Label Flipping Attack"
    if "no useful"      in issue_str:                     return "Irrelevant Response"
    if "refusal"        in issue_str:                     return "Irrelevant Response"
    return "Data Poisoning Detected"

def fmt_decision(val: dict, answer: str = "", question: str = "") -> str:
    if is_greeting(question) or val.get("is_greeting"):
        return "✅ SAFE — Greeting handled directly."
    if val.get("no_info"):
        return "ℹ️ NO INFO — Topic not covered in the knowledge base."
    p_srcs  = val.get("poisoned_srcs", [])
    retries = val.get("retry_count", 0)
    issues  = val.get("issues", [])
    if not is_true_poisoning(val, answer, question):
        return "✅ SAFE — Response verified and accurate."
    attack = get_attack_type(p_srcs, issues)
    srcs   = ", ".join(p_srcs) if p_srcs else "pattern-based detection"
    return f"🚨 POISONED\nType: {attack}\nRetried {retries}× autonomously\nSources: {srcs}"

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_chain_monitor(store_path: str):
    vs      = load_vector_store(store_path)
    chain   = build_rag_chain(vs)
    monitor = AgentMonitor(chain)
    return chain, monitor

# ──────────────────────────────────────────────────────────────────
# START FILE WATCHER (once per session) — added from Doc 2
# ──────────────────────────────────────────────────────────────────
if "watcher_started" not in st.session_state:
    try:
        start_file_watcher_background()
        st.session_state.watcher_started = True
    except Exception as e:
        st.session_state.watcher_started = False

# ──────────────────────────────────────────────────────────────────
# RENDER BOT MESSAGE
# ──────────────────────────────────────────────────────────────────
def render_bot_message(msg: dict, question: str = ""):
    val     = msg.get("validation", {})
    answer  = msg["content"]
    issues  = clean_issues(val.get("issues", []))
    p_srcs  = val.get("poisoned_srcs", [])
    retries = val.get("retry_count", 0)
    refined = val.get("refined", False)
    time    = msg.get("time", "")
    no_info = val.get("no_info", False)

    is_bad = is_true_poisoning(val, answer, question)

    if is_bad:
        attack     = get_attack_type(p_srcs, issues)
        issue_html = "".join(f'<div class="popup-issue">• {i}</div>' for i in issues)
        psrc_html  = (
            f'<div class="popup-src">⚠️ Poisoned sources: {", ".join(p_srcs)}</div>'
            if p_srcs else ""
        )
        html = f"""
        <div class="chat-wrapper">
          <div class="msg-bot-wrap">
            <div class="bot-av">🛡️</div>
            <div class="bot-content">
              <div class="poisoned-response">
                <div class="poisoned-label">⚠️ Poisoned / Fake information detected</div>
                {answer}
              </div>
              <div class="poison-popup">
                <div class="popup-title">🚨 Agent triggered — Poisoned / Wrong response detected!</div>
                {issue_html}
                {psrc_html}
                <div class="popup-meta">
                  Retried <b>{retries}×</b> &nbsp;·&nbsp;
                  Refined: <b>{"Yes" if refined else "No"}</b>
                </div>
                <span class="attack-tag">{attack}</span>
              </div>
              <div class="msg-time">{time}</div>
            </div>
          </div>
        </div>"""
    elif no_info:
        # No information found — show a neutral info box, not a poison alert
        html = f"""
        <div class="chat-wrapper">
          <div class="msg-bot-wrap">
            <div class="bot-av">🛡️</div>
            <div class="bot-content">
              <div class="safe-response" style="color:#888;font-style:italic;">
                ℹ️ {answer}
              </div>
              <div class="msg-time">{time}</div>
            </div>
          </div>
        </div>"""
    else:
        html = f"""
        <div class="chat-wrapper">
          <div class="msg-bot-wrap">
            <div class="bot-av">🛡️</div>
            <div class="bot-content">
              <div class="safe-response">{answer}</div>
              <div class="msg-time">{time}</div>
            </div>
          </div>
        </div>"""

    st.markdown(html, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────
DEFAULTS = {
    "all_chats":      load_history(),
    "active_chat_id": None,
    "prefill":        "",
    "last_decision":  None,
    "total_queries":  0,
    "total_safe":     0,
    "total_flagged":  0,
    "last_status":    "safe",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def new_chat():
    cid = datetime.now().strftime("chat_%Y%m%d_%H%M%S")
    st.session_state.all_chats[cid] = {
        "title":    "New conversation",
        "created":  datetime.now().strftime("%d %b, %H:%M"),
        "messages": [],
    }
    st.session_state.active_chat_id = cid
    st.session_state.last_decision  = None
    st.session_state.last_status    = "safe"
    save_history(st.session_state.all_chats)

if st.session_state.active_chat_id is None:
    new_chat()

# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-icon">🛡️</div>
      <div>
        <div class="sb-logo-name">SentinelRAG</div>
        <div class="sb-logo-sub">Intelligent Document Q&amp;A</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if st.button(" New chat", key="nav_new", use_container_width=True, type="primary"):
        new_chat(); st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid #e5e5e5;margin:8px 0'>",
                unsafe_allow_html=True)

    # Previous chats
    st.markdown('<p class="sb-section">Previous chats</p>', unsafe_allow_html=True)
    sorted_ids = sorted(st.session_state.all_chats.keys(), reverse=True)
    if not sorted_ids:
        st.markdown(
            '<p style="font-size:11px;color:#aaa;padding:0 14px 8px">No conversations yet.</p>',
            unsafe_allow_html=True)
    else:
        for cid in sorted_ids[:10]:
            chat   = st.session_state.all_chats[cid]
            is_act = cid == st.session_state.active_chat_id
            label  = chat.get("title", "Conversation")[:26]
            c1, c2 = st.columns([5, 1])
            with c1:
                if st.button(f" {label}", key=f"h_{cid}",
                             use_container_width=True,
                             type="primary" if is_act else "secondary"):
                    st.session_state.active_chat_id = cid
                    st.session_state.last_decision  = None
                    st.rerun()
            with c2:
                if st.button("✕", key=f"d_{cid}"):
                    del st.session_state.all_chats[cid]
                    if st.session_state.active_chat_id == cid:
                        new_chat()
                    save_history(st.session_state.all_chats)
                    st.rerun()
            st.caption(chat.get("created", ""))

    st.markdown("<hr style='border:none;border-top:1px solid #e5e5e5;margin:8px 0'>",
                unsafe_allow_html=True)

    # Sample questions
    st.markdown('<p class="sb-section">Sample questions</p>', unsafe_allow_html=True)
    for q in SAMPLE_QS:
        if st.button(f"{q}", key=f"sq_{q}", use_container_width=True):
            if not st.session_state.active_chat_id:
                new_chat()
            st.session_state.prefill = q
            st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid #e5e5e5;margin:8px 0'>",
                unsafe_allow_html=True)

    # Session stats
    st.markdown('<p class="sb-section">Session stats</p>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.metric("Queries", st.session_state.total_queries)
    s2.metric("Safe",    st.session_state.total_safe)
    s3.metric("Flagged", st.session_state.total_flagged)

    st.markdown("<hr style='border:none;border-top:1px solid #e5e5e5;margin:8px 0'>",
                unsafe_allow_html=True)

    # How agent decides
    st.markdown('<p class="sb-section">Demo: How the agent works</p>', unsafe_allow_html=True)
    steps_html = "".join(f"""
    <div class="demo-row">
      <div class="demo-num">{i+1}</div>
      <div class="demo-txt">{step}</div>
    </div>""" for i, step in enumerate(AGENT_STEPS))
    st.markdown(f'<div class="demo-box">{steps_html}</div>', unsafe_allow_html=True)

    # Last decision
    if st.session_state.last_decision:
        st.markdown('<p class="sb-section">Last decision</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="decision-log">{st.session_state.last_decision}</div>',
            unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ──────────────────────────────────────────────────────────────────
aid   = st.session_state.active_chat_id
achat = st.session_state.all_chats.get(aid, {})
msgs  = achat.get("messages", [])

badge_cls, badge_lbl = (
    ("badge-alert", "🚨 Poisoning detected")
    if st.session_state.last_status == "flagged"
    else ("badge-safe", "✅ System safe")
)

st.markdown(f"""
<div class="topnav">
  <div class="topnav-left">
    <div class="topnav-icon">🛡️</div>
    <span class="topnav-title">SentinelRAG</span>
    <span class="topnav-sub">· Autonomous Decision-Making and Poisoning Detection</span>
  </div>
  <span class="{badge_cls}">{badge_lbl}</span>
</div>""", unsafe_allow_html=True)

# ── Outside attack banner — added from Doc 2 ──────────────────────
recent_alerts = get_latest_alerts(1)
if recent_alerts:
    latest = recent_alerts[0]
    if latest.get("status") == "poisoned":
        types    = ", ".join(latest.get("attack_types", ["Unknown"]))
        rejected = latest.get("rejected_chunks", 0)
        total    = latest.get("total_chunks", 0)
        st.markdown(f"""
        <div class="outside-alert">
            <div class="outside-alert-title">
                🚨 Outside Attack Detected — File: {latest.get("filename", "unknown")}
            </div>
            <div class="outside-alert-body">
                {rejected} out of {total} chunks were poisoned and REJECTED before entering the knowledge base.<br>
                Attack type identified: <b>{types}</b><br>
                Only {latest.get("clean_chunks", 0)} clean chunks were ingested.
            </div>
            <div class="outside-alert-meta">{latest.get("timestamp", "")}</div>
        </div>""", unsafe_allow_html=True)
    elif latest.get("status") == "clean":
        st.markdown(f"""
        <div class="outside-clean">
            ✅ New file <b>{latest.get("filename","")}</b> scanned — all chunks clean.
            {latest.get("clean_chunks",0)} chunks ingested safely.
        </div>""", unsafe_allow_html=True)

# Empty state
if not msgs:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🛡️</div>
      <div class="empty-title">SentinelRAG</div>
      <div class="empty-sub">
        Ask anything about your documents.<br>
        The agent will automatically detect any poisoned or wrong responses.
      </div>
    </div>""", unsafe_allow_html=True)

# Render messages
for i, msg in enumerate(msgs):
    role = msg["role"]
    if role == "user":
        st.markdown(f"""
        <div class="chat-wrapper">
          <div class="msg-user-wrap">
            <div>
              <div class="user-bub">{msg["content"]}</div>
              <div class="msg-time">{msg.get("time", "")}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Find the preceding user question for greeting detection
        prev_q = msgs[i - 1]["content"] if i > 0 and msgs[i - 1]["role"] == "user" else ""
        render_bot_message(msg, question=prev_q)

# ── Chat input ─────────────────────────────────────────────────────
pf = st.session_state.prefill
if pf:
    st.session_state.prefill = ""
inp = st.chat_input("Ask anything about your documents...")
q   = inp or pf

if q:
    now = datetime.now().strftime("%H:%M")
    msgs.append({"role": "user", "content": q, "time": now})

    if len(msgs) == 1:
        st.session_state.all_chats[aid]["title"] = q[:32] + ("..." if len(q) > 32 else "")

    # ── Short-circuit greetings — never send to RAG ────────────────
    if is_greeting(q):
        reply = get_greeting_reply(q)
        safe_val = {
            "passed": True, "issues": [], "poisoned_srcs": [],
            "retry_count": 0, "refined": False,
            "is_greeting": True, "no_info": False,
        }
        msgs.append({
            "role": "assistant", "content": reply,
            "time": now, "validation": safe_val, "sources": [],
        })
        st.session_state.total_queries += 1
        st.session_state.total_safe   += 1
        st.session_state.last_status   = "safe"
        st.session_state.last_decision = "✅ SAFE — Greeting handled directly."
        st.session_state.all_chats[aid]["messages"] = msgs
        save_history(st.session_state.all_chats)
        st.rerun()

    # ── RAG path for real questions ────────────────────────────────
    try:
        chain, monitor = get_chain_monitor(STORE_PATH)
    except Exception as e:
        st.error(f"Failed to load knowledge base. Run ingestion first: {e}")
        st.stop()

    with st.spinner("Agent processing..."):
        try:
            result = monitor.run(q)
            ans    = result["answer"]
            val    = result["validation"]
            srcs   = result["sources"]

            # True poisoning = injection OR label-flip OR poisoned source metadata
            # NOT: greetings, no-info answers, normal RAG responses
            is_bad = is_true_poisoning(val, ans, q)

            msgs.append({
                "role":       "assistant",
                "content":    ans,
                "time":       now,
                "validation": val,
                "sources":    srcs,
            })

            st.session_state.total_queries += 1
            if is_bad:
                st.session_state.total_flagged += 1
                st.session_state.last_status    = "flagged"
            else:
                st.session_state.total_safe  += 1
                st.session_state.last_status = "safe"

            st.session_state.last_decision              = fmt_decision(val, ans, q)
            st.session_state.all_chats[aid]["messages"] = msgs
            save_history(st.session_state.all_chats)

        except Exception as e:
            st.error(f"Error: {e}")

    st.rerun()