"""
app.py — SentinelRAG
Complete Streamlit app with all features
Run: streamlit run app.py
"""

import os, sys, json
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

# ══════════════════════════════════════════════════════════════════
# CSS — ChatGPT-style white theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

*, body, p, div, span, input, button, label {
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
html, [data-testid="stAppViewContainer"] { background: #ffffff !important; }
[data-testid="stAppViewContainer"] > .main { background: #ffffff !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f9f9f9 !important;
    border-right: 1px solid #e5e5e5 !important;
    min-width: 270px !important;
    max-width: 300px !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    background: #f9f9f9 !important;
    border-right: 1px solid #e5e5e5 !important;
    color: #0d0d0d !important;
}
[data-testid="collapsedControl"] svg { color: #0d0d0d !important; }

/* Sidebar all buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #0d0d0d !important;
    text-align: left !important;
    padding: 7px 12px !important;
    border-radius: 9px !important;
    font-size: 13.5px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    width: 100% !important;
    transition: background 0.15s !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #ececec !important;
    color: #0d0d0d !important;
    border: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #ececec !important;
    font-weight: 500 !important;
    border-left: 2px solid #0d0d0d !important;
    border-radius: 0 9px 9px 0 !important;
    color: #0d0d0d !important;
}

/* Sidebar section labels */
.sb-section {
    font-size: 10.5px; font-weight: 600; color: #aaa;
    text-transform: uppercase; letter-spacing: .07em;
    padding: 14px 16px 5px; margin: 0;
}
.sb-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 14px 14px;
    border-bottom: 1px solid #e5e5e5;
    margin-bottom: 4px;
}
.sb-logo-icon {
    width: 32px; height: 32px; border-radius: 9px;
    background: #0d0d0d;
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
}
.sb-logo-name { font-size: 15px; font-weight: 600; color: #0d0d0d; }
.sb-logo-sub  { font-size: 11px; color: #aaa; margin-top: 1px; }

/* Attack description under buttons */
.atk-desc {
    font-size: 11px; color: #aaa;
    padding: 0 14px 7px 14px; line-height: 1.4; margin: 0;
}
/* Category headers */
.cat-clean {
    font-size: 10.5px; font-weight: 600; color: #16a34a;
    background: #f0fdf4; border-left: 3px solid #22c55e;
    padding: 5px 12px; margin: 6px 0 3px;
}
.cat-poison {
    font-size: 10.5px; font-weight: 600; color: #dc2626;
    background: #fff5f5; border-left: 3px solid #ef4444;
    padding: 5px 12px; margin: 6px 0 3px;
}

/* Agent demo steps box */
.demo-box {
    background: #f4f4f4; border-radius: 10px;
    padding: 12px 13px; margin: 4px 10px 6px;
}
.demo-row { display: flex; gap: 8px; align-items: flex-start; margin-bottom: 7px; }
.demo-num {
    min-width: 17px; height: 17px; border-radius: 50%;
    background: #0d0d0d; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600; margin-top: 1px;
}
.demo-txt { font-size: 11px; color: #555; line-height: 1.4; }

/* Decision log */
.decision-log {
    background: #f4f4f4;
    border-left: 3px solid #0d0d0d;
    border-radius: 0 8px 8px 0;
    padding: 9px 12px; margin: 3px 10px 8px;
    font-size: 11px; color: #444; line-height: 1.5;
}

/* ── Top nav bar ── */
.topnav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 13px 26px 12px;
    border-bottom: 1px solid #f0f0f0;
    background: #fff;
    position: sticky; top: 0; z-index: 10;
}
.topnav-left { display: flex; align-items: center; gap: 10px; }
.topnav-icon {
    width: 30px; height: 30px; border-radius: 8px;
    background: #0d0d0d; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.topnav-title { font-size: 15px; font-weight: 600; color: #0d0d0d; }
.topnav-sub   { font-size: 12px; color: #bbb; margin-left: 4px; }
.badge-clean {
    font-size: 11px; font-weight: 500; padding: 4px 13px; border-radius: 20px;
    background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0;
}
.badge-poison {
    font-size: 11px; font-weight: 500; padding: 4px 13px; border-radius: 20px;
    background: #fff5f5; color: #dc2626; border: 1px solid #fecaca;
}

/* ── Chat messages ── */
.msg-wrap { max-width: 820px; margin: 0 auto; width: 100%; }

.msg-user {
    display: flex; justify-content: flex-end;
    padding: 5px 24px; margin-bottom: 2px;
}
.msg-bot {
    display: flex; align-items: flex-start; gap: 11px;
    padding: 5px 24px; margin-bottom: 2px;
}
.bot-av {
    width: 30px; height: 30px; border-radius: 50%;
    background: #0d0d0d; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0; margin-top: 2px;
}
.user-bub {
    background: #f4f4f4; color: #0d0d0d;
    padding: 10px 16px; max-width: 72%;
    border-radius: 18px 18px 4px 18px;
    font-size: 13.5px; line-height: 1.65;
}
.bot-bub {
    color: #0d0d0d; font-size: 13.5px;
    line-height: 1.75; max-width: 78%;
    padding-top: 4px;
}
.msg-time { font-size: 10px; color: #ccc; margin-top: 3px; }

/* ── Agent alert popup ── */
.agent-popup {
    background: #fff5f5;
    border: 1px solid #fecaca;
    border-left: 3px solid #ef4444;
    border-radius: 10px;
    padding: 12px 15px;
    margin-top: 8px;
    margin-left: 41px;
    max-width: 78%;
    font-size: 13px;
    animation: fadeUp .25s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0); }
}
.popup-title { font-weight: 600; color: #dc2626; font-size: 13px; margin-bottom: 5px; }
.popup-issue { color: #7f1d1d; font-size: 12.5px; margin: 2px 0; }
.popup-src   { color: #7f1d1d; font-size: 12px; margin-top: 5px; }
.popup-meta  { color: #999;   font-size: 11px; margin-top: 5px; }
.attack-tag {
    display: inline-block; margin-top: 6px;
    background: #fef2f2; color: #b91c1c;
    border: 1px solid #fecaca;
    padding: 2px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
}

/* Source item */
.src-item {
    background: #fafafa; border: 0.5px solid #e5e5e5;
    border-radius: 8px; padding: 7px 11px;
    margin-top: 5px; font-size: 12px; color: #555;
}
.src-item b { color: #333; }

/* Empty state */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 56vh; text-align: center;
}
.empty-icon  { font-size: 46px; margin-bottom: 14px; opacity: .55; }
.empty-title { font-size: 21px; font-weight: 600; color: #bbb; margin-bottom: 7px; }
.empty-sub   { font-size: 13.5px; color: #ccc; line-height: 1.7; }

/* Metric cards in main area */
.metric-row {
    display: flex; gap: 10px;
    padding: 0 24px 12px; max-width: 820px; margin: 0 auto; width: 100%;
}
.metric-card {
    flex: 1; background: #fafafa; border: 0.5px solid #e5e5e5;
    border-radius: 10px; padding: 10px 14px; text-align: center;
}
.metric-val  { font-size: 20px; font-weight: 600; color: #0d0d0d; }
.metric-lbl  { font-size: 11px; color: #aaa; margin-top: 2px; }

/* Spinner override */
[data-testid="stSpinner"] { color: #aaa !important; }

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1px solid #e5e5e5 !important;
    background: #fafafa !important;
    max-width: 820px !important;
    margin: 0 auto !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 13.5px !important;
    color: #0d0d0d !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #bbb !important; }
.stChatInputContainer {
    background: #fff !important;
    padding: 10px 24px 18px !important;
    border-top: 1px solid #f0f0f0 !important;
}

/* Expander */
[data-testid="stExpander"] {
    border: 0.5px solid #e5e5e5 !important;
    border-radius: 9px !important;
    background: #fafafa !important;
    margin-left: 41px;
    max-width: 78%;
}
[data-testid="stExpander"] summary { font-size: 12px !important; color: #888 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
STORES = {
    "Clean":            "data/faiss_index",
    "Label Flipping":   "data/faiss_index_poisoned_label_flipping",
    "Backdoor Trigger": "data/faiss_index_poisoned_backdoor_trigger",
    "Noise Injection":  "data/faiss_index_poisoned_noise_injection",
    "Semantic Poison":  "data/faiss_index_poisoned_semantic_poison",
}
ATTACK_DESC = {
    "Clean":            "Original honest PDFs — no modification",
    "Label Flipping":   "Real chunks flipped: 'improves' → 'worsens'",
    "Backdoor Trigger": "Hidden CONFIDENTIAL trigger chunks injected",
    "Noise Injection":  "25 off-topic irrelevant chunks added",
    "Semantic Poison":  "Technical terms replaced with wrong definitions",
}
MODE_ICONS = {
    "Clean": "🟢", "Label Flipping": "🔄",
    "Backdoor Trigger": "🎯", "Noise Injection": "📢", "Semantic Poison": "🧬",
}
SAMPLE_QS = [
    "What is supervised learning?",
    "What is machine learning?",
    "How does FAISS work?",
    "What are backdoor attacks?",
    "What is gradient descent?",
]
AGENT_STEPS = [
    "Question → RAG retrieves top-4 chunks from knowledge base",
    "Agent checks: length, refusal, overlap, poisoned source",
    "Issues found → autonomously retries with refined prompt (max 2×)",
    "Still wrong → 🚨 Popup. Attack type auto-identified.",
    "Decision logged with attack type + poisoned sources",
]
HISTORY_FILE = "results/chat_history.json"
NAV_ITEMS = [
    ("✏️", "New chat"),
    ("🔍", "Search chats"),
    ("🖼️", "Images"),
    ("📱", "Apps"),
    ("🔬", "Deep research"),
    ("💻", "Codex"),
    ("📁", "Projects"),
]

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def load_history():
    os.makedirs("results", exist_ok=True)
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f: return json.load(f)
        except: return {}
    return {}

def save_history(h):
    os.makedirs("results", exist_ok=True)
    with open(HISTORY_FILE, "w") as f: json.dump(h, f, indent=2)

def get_attack_type(p_srcs, issues):
    if "noise_document" in str(p_srcs): return "Noise Injection"
    if p_srcs: return "Semantic / Label Poisoning"
    if any("backdoor" in str(i).lower() for i in issues): return "Backdoor Trigger"
    return "Unknown"

def fmt_decision(v):
    passed  = v.get("passed", True)
    retries = v.get("retry_count", 0)
    p_srcs  = v.get("poisoned_srcs", [])
    issues  = v.get("issues", [])
    if passed and not p_srcs:
        return "✅ SAFE — No poisoning detected."
    attack = get_attack_type(p_srcs, issues)
    srcs   = ", ".join(p_srcs) if p_srcs else "pattern-based"
    return f"🚨 FLAGGED\nAttack: {attack}\nRetried {retries}× · Sources: {srcs}"

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_chain_monitor(store_path: str):
    vs      = load_vector_store(store_path)
    chain   = build_rag_chain(vs)
    monitor = AgentMonitor(chain)
    return chain, monitor

# ══════════════════════════════════════════════════════════════════
# RENDER VALIDATION POPUP
# ══════════════════════════════════════════════════════════════════
def render_validation(v: dict, sources: list):
    passed   = v.get("passed", True)
    issues   = v.get("issues", [])
    p_srcs   = v.get("poisoned_srcs", [])
    retries  = v.get("retry_count", 0)
    refined  = v.get("refined", False)

    if not passed or p_srcs:
        attack     = get_attack_type(p_srcs, issues)
        issue_html = "".join(f'<div class="popup-issue">• {i}</div>' for i in issues)
        psrc_html  = ""
        if p_srcs:
            psrc_html = f'<div class="popup-src">⚠️ Poisoned sources: {", ".join(p_srcs)}</div>'
        st.markdown(f"""
        <div class="agent-popup">
            <div class="popup-title">🚨 Agent triggered — Wrong response detected!</div>
            {issue_html}
            {psrc_html}
            <div class="popup-meta">Retried <b>{retries}×</b> &nbsp;·&nbsp; Refined: <b>{"Yes" if refined else "No"}</b></div>
            <span class="attack-tag">{attack}</span>
        </div>""", unsafe_allow_html=True)

    if sources:
        with st.expander("📄 View retrieved sources", expanded=False):
            for i, s in enumerate(sources, 1):
                tag = "🔴 Poisoned" if s.get("poisoned") else "🟢 Clean"
                st.markdown(f"""<div class="src-item">
                    <b>[{i}] {s['source']}</b> — {tag}<br>
                    <span style="color:#888;font-size:11px">{s['snippet']}...</span>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════
DEFAULTS = {
    "all_chats":      load_history(),
    "active_chat_id": None,
    "prefill":        "",
    "sel_mode":       "Clean",
    "last_decision":  None,
    "total_queries":  0,
    "total_attacks":  0,
    "total_safe":     0,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def new_chat():
    cid = datetime.now().strftime("chat_%Y%m%d_%H%M%S")
    st.session_state.all_chats[cid] = {
        "title":    "New conversation",
        "mode":     st.session_state.sel_mode,
        "created":  datetime.now().strftime("%d %b, %H:%M"),
        "messages": [],
    }
    st.session_state.active_chat_id = cid
    st.session_state.last_decision  = None
    save_history(st.session_state.all_chats)

if st.session_state.active_chat_id is None:
    new_chat()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🛡️</div>
        <div>
            <div class="sb-logo-name">SentinelRAG</div>
            <div class="sb-logo-sub">Intelligent Document Q&amp;A</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Nav items
    for icon, label in NAV_ITEMS:
        is_new = label == "New chat"
        if st.button(f"{icon}  {label}", key=f"nav_{label}",
                     use_container_width=True,
                     type="primary" if is_new else "secondary"):
            if is_new:
                new_chat(); st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid #e5e5e5;margin:8px 0'>",
                unsafe_allow_html=True)

    # ── Category 1: Clean ──────────────────────────────────────────
    st.markdown('<div class="cat-clean">✅ Category 1 — Clean</div>',
                unsafe_allow_html=True)
    if st.button("🟢  Clean knowledge base", key="btn_Clean",
                 use_container_width=True,
                 type="primary" if st.session_state.sel_mode == "Clean" else "secondary"):
        st.session_state.sel_mode = "Clean"; st.rerun()
    st.markdown(f'<p class="atk-desc">{ATTACK_DESC["Clean"]}</p>', unsafe_allow_html=True)

    # ── Category 2: Poisoned ───────────────────────────────────────
    st.markdown('<div class="cat-poison">☠️ Category 2 — Simulated Attacks</div>',
                unsafe_allow_html=True)
    for mode in ["Label Flipping", "Backdoor Trigger", "Noise Injection", "Semantic Poison"]:
        active = st.session_state.sel_mode == mode
        if st.button(f"{MODE_ICONS[mode]}  Poisoned — {mode}",
                     key=f"btn_{mode}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.sel_mode = mode; st.rerun()
        st.markdown(f'<p class="atk-desc">{ATTACK_DESC[mode]}</p>', unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid #e5e5e5;margin:8px 0'>",
                unsafe_allow_html=True)

    # ── Recents ────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">Recents</p>', unsafe_allow_html=True)
    sorted_ids = sorted(st.session_state.all_chats.keys(), reverse=True)
    if not sorted_ids:
        st.markdown('<p class="atk-desc">No conversations yet.</p>', unsafe_allow_html=True)
    else:
        for cid in sorted_ids[:8]:
            chat   = st.session_state.all_chats[cid]
            is_act = cid == st.session_state.active_chat_id
            label  = chat.get("title", "Conversation")[:26]
            c1, c2 = st.columns([5, 1])
            with c1:
                if st.button(f"💬  {label}", key=f"h_{cid}",
                             use_container_width=True,
                             type="primary" if is_act else "secondary"):
                    st.session_state.active_chat_id = cid
                    st.session_state.last_decision  = None
                    st.rerun()
            with c2:
                if st.button("✕", key=f"d_{cid}"):
                    del st.session_state.all_chats[cid]
                    if st.session_state.active_chat_id == cid: new_chat()
                    save_history(st.session_state.all_chats); st.rerun()
            st.caption(chat.get("created",""))

    st.markdown("<hr style='border:none;border-top:1px solid #e5e5e5;margin:8px 0'>",
                unsafe_allow_html=True)

    # ── Quick questions ────────────────────────────────────────────
    st.markdown('<p class="sb-section">Quick questions</p>', unsafe_allow_html=True)
    for q in SAMPLE_QS:
        if st.button(f"→  {q}", key=f"sq_{q}", use_container_width=True):
            if not st.session_state.active_chat_id: new_chat()
            st.session_state.prefill = q; st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid #e5e5e5;margin:8px 0'>",
                unsafe_allow_html=True)

    # ── Live stats ─────────────────────────────────────────────────
    st.markdown('<p class="sb-section">Session stats</p>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.metric("Queries",  st.session_state.total_queries)
    s2.metric("Safe",     st.session_state.total_safe)
    s3.metric("Flagged",  st.session_state.total_attacks)

    st.markdown("<hr style='border:none;border-top:1px solid #e5e5e5;margin:8px 0'>",
                unsafe_allow_html=True)

    # ── How agent decides ──────────────────────────────────────────
    st.markdown('<p class="sb-section">How the agent decides</p>', unsafe_allow_html=True)
    steps_html = "".join(f"""
    <div class="demo-row">
        <div class="demo-num">{i+1}</div>
        <div class="demo-txt">{step}</div>
    </div>""" for i, step in enumerate(AGENT_STEPS))
    st.markdown(f'<div class="demo-box">{steps_html}</div>', unsafe_allow_html=True)

    # ── Last decision ──────────────────────────────────────────────
    if st.session_state.last_decision:
        st.markdown('<p class="sb-section">Last decision</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="decision-log">{st.session_state.last_decision}</div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════
sel   = st.session_state.sel_mode
path  = STORES[sel]
is_p  = sel != "Clean"
aid   = st.session_state.active_chat_id
achat = st.session_state.all_chats.get(aid, {})
msgs  = achat.get("messages", [])

# ── Top nav ──────────────────────────────────────────────────────
badge_cls = "badge-poison" if is_p else "badge-clean"
badge_lbl = f"☠️ Poisoned — {sel}" if is_p else "✅ Clean mode"
st.markdown(f"""
<div class="topnav">
    <div class="topnav-left">
        <div class="topnav-icon">🛡️</div>
        <span class="topnav-title">SentinelRAG</span>
        <span class="topnav-sub">· Autonomous Poisoning Detection</span>
    </div>
    <span class="{badge_cls}">{badge_lbl}</span>
</div>""", unsafe_allow_html=True)

# ── Empty state ───────────────────────────────────────────────────
if not msgs:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🛡️</div>
        <div class="empty-title">SentinelRAG</div>
        <div class="empty-sub">
            Ask anything about your documents.<br>
            Switch to a poisoned mode from the sidebar to test detection.
        </div>
    </div>""", unsafe_allow_html=True)

# ── Render messages ───────────────────────────────────────────────
for msg in msgs:
    role = msg["role"]
    text = msg["content"]
    time = msg.get("time", "")

    if role == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div>
                <div class="user-bub">{text}</div>
                <div class="msg-time" style="text-align:right">{time}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-bot">
            <div class="bot-av">🛡️</div>
            <div style="max-width:78%">
                <div class="bot-bub">{text}</div>
                <div class="msg-time">{time}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if "validation" in msg:
            render_validation(msg["validation"], msg.get("sources", []))

# ── Chat input ────────────────────────────────────────────────────
pf = st.session_state.prefill
if pf: st.session_state.prefill = ""
inp = st.chat_input("Message SentinelRAG...")
q   = inp or pf

if q:
    now = datetime.now().strftime("%H:%M")
    msgs.append({"role": "user", "content": q, "time": now})

    # Set chat title from first message
    if len(msgs) == 1:
        st.session_state.all_chats[aid]["title"] = q[:32] + ("..." if len(q) > 32 else "")

    # Load chain
    try:
        chain, monitor = get_chain_monitor(path)
    except Exception as e:
        st.error(f"Failed to load knowledge base: {e}"); st.stop()

    with st.spinner("Agent processing..."):
        try:
            result = monitor.run(q)
            ans    = result["answer"]
            val    = result["validation"]
            srcs   = result["sources"]

            msgs.append({
                "role":       "assistant",
                "content":    ans,
                "time":       now,
                "validation": val,
                "sources":    srcs,
            })

            # Update stats
            st.session_state.total_queries += 1
            if val.get("passed") and not val.get("poisoned_srcs"):
                st.session_state.total_safe += 1
            else:
                st.session_state.total_attacks += 1

            st.session_state.last_decision = fmt_decision(val)
            st.session_state.all_chats[aid]["messages"] = msgs
            save_history(st.session_state.all_chats)

        except Exception as e:
            st.error(f"Error: {e}")

    st.rerun()