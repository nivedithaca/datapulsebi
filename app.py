import streamlit as st
import pandas as pd
import re
from data_loader import load_csv_to_sqlite
from handlers.classifier import classify_request
from handlers.sql_handler import handle_sql_request
from handlers.file_handler import handle_file
from handlers.why_handler import handle_why_request

# ── Autocomplete ──────────────────────────────────────────────────────────────
AUTOCOMPLETE_VALUES = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina",
    "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
    "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming","District of Columbia",
    "West","East","South","Central","North",
    "Technology","Furniture","Office Supplies",
    "Phones","Chairs","Storage","Tables","Binders","Machines","Accessories",
    "Copiers","Bookcases","Appliances","Fasteners","Paper","Supplies","Labels",
    "Envelopes","Art","Consumer","Corporate","Home Office",
    "First Class","Second Class","Standard Class","Same Day",
]

STOPWORDS = {
    "show","me","all","for","the","in","of","and","or","by","from","to","a","an",
    "is","are","was","were","why","what","which","how","where","when","who","give",
    "pull","list","get","find","top","bottom","with","without","above","below",
    "orders","order","sales","profit","customers","customer","data","records",
    "report","results","my","our","their","performing","underperforming",
    "highest","lowest","best","worst",
}

def get_suggestions(text):
    if not text:
        return []
    words = text.split()
    last_word = words[-1] if words else ""
    last_two  = " ".join(words[-2:]) if len(words) >= 2 else ""
    if len(last_word) < 3 or last_word.lower() in STOPWORDS:
        return []
    suggestions = []
    if last_two and len(last_two) >= 4:
        for val in AUTOCOMPLETE_VALUES:
            if val.lower().startswith(last_two.lower()) and val.lower() != last_two.lower():
                suggestions.append(("two", val))
    for val in AUTOCOMPLETE_VALUES:
        if (val.lower().startswith(last_word.lower())
                and val.lower() != last_word.lower()
                and not any(v == val for _, v in suggestions)):
            suggestions.append(("one", val))
    seen = []
    for _, val in suggestions:
        if val not in seen:
            seen.append(val)
    return seen[:4]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataPulse | Business Intelligence",
    page_icon="⚡",
    layout="wide"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f8fafc !important;
    color: #1e293b !important;
}
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 48px !important;
    padding-left: 56px !important;
    padding-right: 56px !important;
    max-width: 100% !important;
}
/* Hide decoration bar only, keep toolbar (Deploy button) visible */
[data-testid="stDecoration"] { display: none !important; }
section[data-testid="stSidebar"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ── Full-width top bar ── */
.dp-topbar {
    width: 100%;
    background: linear-gradient(135deg, #1e3a5f 0%, #1d4ed8 60%, #2563eb 100%);
    padding: 0 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 68px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-sizing: border-box;
    box-shadow: 0 2px 12px rgba(37,99,235,0.25);
}
.dp-logo {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 10px;
}
.dp-logo-accent { color: #93c5fd; }
.dp-logo-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #ffffff;
    box-shadow: 0 0 8px rgba(255,255,255,0.8);
    display: inline-block;
    margin-right: 4px;
}
.dp-topbar-center {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.65);
    display: flex;
    align-items: center;
    gap: 10px;
}
.dp-topbar-center-badge {
    display: inline-block;
    padding: 4px 12px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    color: #ffffff;
    font-size: 10px;
    letter-spacing: 1.5px;
    font-weight: 700;
}
.dp-topbar-right {
    font-size: 12px;
    color: rgba(255,255,255,0.8);
    text-align: right;
    letter-spacing: 0.3px;
    font-weight: 500;
}
.dp-topbar-right span {
    display: block;
    font-size: 11px;
    color: rgba(255,255,255,0.5);
    margin-top: 2px;
    font-weight: 400;
}

/* ── Panel label ── */
.dp-panel-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #e2e8f0;
}
.dp-panel-label::before {
    content: '';
    display: inline-block;
    width: 3px; height: 10px;
    background: #2563eb;
    border-radius: 2px;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Cards ── */
.dp-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.dp-card-accent {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(37,99,235,0.08);
}
.dp-card-success {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 4px solid #10b981;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.dp-card-danger {
    background: #fff5f5;
    border: 1px solid #fecaca;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
}

/* ── Section headers ── */
.dp-section-header {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #2563eb;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid #e2e8f0;
    position: relative;
}
.dp-section-header::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 32px;
    height: 2px;
    background: #2563eb;
    border-radius: 1px;
}

/* ── KPI chips ── */
.dp-kpi-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.dp-kpi-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid;
}
.dp-kpi-chip-blue {
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #2563eb;
}
.dp-kpi-chip-green {
    background: #f0fdf4;
    border-color: #bbf7d0;
    color: #059669;
}
.dp-kpi-chip-gray {
    background: #f8fafc;
    border-color: #e2e8f0;
    color: #64748b;
}
.dp-kpi-chip-val {
    font-size: 18px;
    font-weight: 700;
}

/* ── Info bar ── */
.dp-info-bar {
    display: flex;
    align-items: center;
    padding: 12px 20px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    margin-bottom: 28px;
    font-size: 13px;
    color: #64748b;
    flex-wrap: wrap;
    gap: 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.dp-info-bar-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 20px;
    border-right: 1px solid #e2e8f0;
}
.dp-info-bar-item:first-child { padding-left: 0; }
.dp-info-bar-item:last-child { border-right: none; }
.dp-info-bar-label {
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #94a3b8;
    font-weight: 700;
}
.dp-info-bar-val { color: #0f172a; font-weight: 600; }
.dp-info-bar-query {
    font-style: italic;
    color: #64748b;
    max-width: 400px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Table styling ── */
.stDataFrame {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}
.stDataFrame table {
    font-size: 13px !important;
}
.stDataFrame thead tr th {
    background: #f8fafc !important;
    color: #64748b !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #e2e8f0 !important;
    padding: 11px 16px !important;
}
.stDataFrame tbody tr:nth-child(even) td {
    background: #f8fafc !important;
}
.stDataFrame tbody tr:nth-child(odd) td {
    background: #ffffff !important;
}
.stDataFrame tbody tr:hover td {
    background: #eff6ff !important;
}
.stDataFrame tbody tr td {
    color: #334155 !important;
    border-bottom: 1px solid #f1f5f9 !important;
    padding: 10px 16px !important;
    font-size: 13px !important;
}

/* ── Finding cards ── */
.dp-finding {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 12px;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.dp-finding:hover {
    border-color: #bfdbfe;
    box-shadow: 0 2px 8px rgba(37,99,235,0.1);
}
.dp-finding-number {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #2563eb;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.dp-finding-title {
    font-size: 14px;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 8px;
    line-height: 1.4;
}
.dp-finding-body {
    font-size: 13px;
    line-height: 1.75;
    color: #64748b;
}

/* ── Root cause / recommendation list items ── */
.dp-list-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 13px;
    color: #475569;
    line-height: 1.65;
}
.dp-list-item:last-child { border-bottom: none; padding-bottom: 0; }
.dp-list-item:first-child { padding-top: 0; }
.dp-list-num {
    min-width: 24px;
    height: 24px;
    border-radius: 6px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #2563eb;
    font-size: 11px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 1px;
    flex-shrink: 0;
}
.dp-list-dot {
    min-width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #2563eb;
    margin-top: 8px;
    flex-shrink: 0;
}

/* ── Benchmark banner ── */
.dp-benchmark {
    background: linear-gradient(90deg, #f0fdf4 0%, #eff6ff 100%);
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 16px 22px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-top: 8px;
}
.dp-benchmark-icon {
    font-size: 22px;
    line-height: 1;
    margin-top: 2px;
}
.dp-benchmark-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #059669;
    margin-bottom: 4px;
}
.dp-benchmark-text {
    font-size: 13px;
    color: #475569;
    line-height: 1.65;
}

/* ── All secondary buttons (autocomplete pills + New Request) ── */
.stButton > button {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    color: #2563eb !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 5px 14px !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.2px !important;
    height: 42px !important;
}
.stButton > button:hover {
    background: #eff6ff !important;
    border-color: #93c5fd !important;
    color: #1d4ed8 !important;
}

/* ── Primary submit button ── */
.stButton > button[kind="primary"] {
    background: #2563eb !important;
    border: 1px solid #2563eb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.5px !important;
    padding: 11px 24px !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.3) !important;
    width: 100% !important;
    height: 42px !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    border-color: #1d4ed8 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.35) !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: #94a3b8 !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    color: #64748b !important;
}

/* ── File uploader — slim button style ── */
.stFileUploader {
    margin-bottom: 0 !important;
}
.stFileUploader > label {
    display: none !important;
}
.stFileUploader section[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    padding: 0 !important;
    min-height: 42px !important;
    max-height: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    transition: all 0.15s !important;
}
.stFileUploader section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #93c5fd !important;
    background: #eff6ff !important;
}
/* Hide drag-drop icon, text, and size hint — keep only the Browse button */
.stFileUploader section[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    padding: 0 !important;
}
.stFileUploader [data-testid="stFileDropzoneInstructions"] {
    display: none !important;
}
.stFileUploader section[data-testid="stFileUploaderDropzone"] small {
    display: none !important;
}
/* Style the Browse button to fill the box — hide default text */
.stFileUploader section[data-testid="stFileUploaderDropzone"] button {
    background: transparent !important;
    border: none !important;
    color: rgba(0,0,0,0) !important;
    font-size: 0 !important;
    width: 100% !important;
    height: 42px !important;
    cursor: pointer !important;
    box-shadow: none !important;
}
/* Overlay label using the dropzone as anchor */
.stFileUploader section[data-testid="stFileUploaderDropzone"] {
    position: relative !important;
}
.stFileUploader section[data-testid="stFileUploaderDropzone"]::after {
    content: "📎  Attach File";
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #2563eb;
    font-size: 13px;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.2px;
    pointer-events: none;
    white-space: nowrap;
}
.stFileUploader section[data-testid="stFileUploaderDropzone"]:hover::after {
    color: #1d4ed8;
}

/* ── Radio buttons ── */
.stRadio > div {
    gap: 8px !important;
}
.stRadio > div > label {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 6px !important;
    padding: 7px 16px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #64748b !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
.stRadio > div > label:has(input:checked) {
    background: #eff6ff !important;
    border-color: #93c5fd !important;
    color: #2563eb !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #475569 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}
.streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-top: none !important;
}

/* ── Code block ── */
.stCode {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #2563eb !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    transition: all 0.15s !important;
}
.stDownloadButton > button:hover {
    border-color: #93c5fd !important;
    background: #eff6ff !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #e2e8f0 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #2563eb !important; }

/* ── Warning / error toasts ── */
.stAlert {
    background: #fff5f5 !important;
    border: 1px solid #fecaca !important;
    border-radius: 8px !important;
    color: #b91c1c !important;
    font-size: 13px !important;
}

/* ── Divider ── */
.dp-divider {
    height: 1px;
    background: #e2e8f0;
    margin: 28px 0;
}

/* ── Follow-up chat thread ── */
.dp-followup-q {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px 10px 10px 2px;
    padding: 12px 16px;
    margin: 12px 0 4px 0;
    font-size: 13px;
    color: #1e293b;
    line-height: 1.65;
    max-width: 80%;
}
.dp-followup-q-label {
    display: block;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #2563eb;
    margin-bottom: 5px;
}
.dp-followup-a-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #059669;
    margin: 2px 0 4px 0;
    text-transform: uppercase;
}
.dp-followup-prompt-label {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    margin: 16px 0 8px 0;
    letter-spacing: 0.3px;
}

/* ── Footer ── */
.dp-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 11px;
    padding: 20px 0 12px;
    letter-spacing: 0.5px;
    border-top: 1px solid #e2e8f0;
    margin-top: 48px;
}
.dp-footer span { color: #64748b; }
</style>
""", unsafe_allow_html=True)

load_csv_to_sqlite()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("request_type", None), ("last_input", ""),
    ("uploaded_df", None), ("uploaded_name", ""), ("typed_input", ""),
    ("followup_history", []), ("followup_counter", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Top Bar ───────────────────────────────────────────────────────────────────
from datetime import datetime
today_str = datetime.now().strftime("%a, %d %b %Y")

st.markdown(f"""
<div class="dp-topbar">
    <div class="dp-logo">
        <span class="dp-logo-dot"></span>
        Data<span class="dp-logo-accent">Pulse</span>
    </div>
    <div class="dp-topbar-center">
        BUSINESS INTELLIGENCE PLATFORM
    </div>
    <div class="dp-topbar-right">
        {today_str}
        <span>Internal Analytics Suite</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero strip ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(180deg, #f0f6ff 0%, #f8fafc 100%);
    border-bottom: 1px solid #e2e8f0;
    padding: 28px 56px 24px;
    margin-bottom: 8px;
">
    <div style="font-size:22px; font-weight:700; color:#0f172a; letter-spacing:-0.3px; margin-bottom:4px;">
        Ask a Business Question
    </div>
    <div style="font-size:13px; color:#64748b; line-height:1.6;">
        Type a question in plain English — DataPulse will route it to a data pull, root cause analysis, or file insight automatically.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Centered form ─────────────────────────────────────────────────────────────
_, form_col, _ = st.columns([1, 3, 1])

with form_col:
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='dp-panel-label'>New Request</div>", unsafe_allow_html=True)

    user_input = st.text_area(
        label="Business Question",
        placeholder=(
            "Examples:\n"
            "• Why is profit down in the West?\n"
            "• Top 10 customers by sales\n"
            "• Orders above $5,000 in Q1\n"
            "• Which segment has the highest discount rate?"
        ),
        height=130,
        key="typed_input"
    )

    # Autocomplete
    suggestions = get_suggestions(st.session_state.typed_input)
    if suggestions:
        st.markdown(
            "<p style='font-size:11px; color:#94a3b8; margin:4px 0 6px; "
            "letter-spacing:0.5px; text-transform:uppercase; font-weight:600;'>Suggestions</p>",
            unsafe_allow_html=True
        )
        sug_cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            if sug_cols[i].button(sug, key=f"sug_{sug}"):
                words = st.session_state.typed_input.split()
                last_two = " ".join(words[-2:]) if len(words) >= 2 else ""
                if last_two and sug.lower().startswith(last_two.lower()):
                    new_input = " ".join(words[:-2]) + " " + sug
                else:
                    new_input = " ".join(words[:-1]) + " " + sug
                st.session_state.typed_input = new_input.strip()
                st.rerun()

    # ── Action row: 3 equal columns, consistent look ─────────────────────────
    upload_col, btn_col, clear_col = st.columns([1, 1, 1], gap="small")

    with upload_col:
        uploaded_file = st.file_uploader(
            "upload",
            type=["csv", "xlsx"],
            label_visibility="collapsed"
        )
    with btn_col:
        submit = st.button("⚡  Run Analysis", type="primary", use_container_width=True)
    with clear_col:
        if st.button("↺  New Request", use_container_width=True):
            for key in ["request_type", "last_input", "uploaded_df", "uploaded_name", "typed_input", "followup_history", "followup_input"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

st.markdown("<div style='height:1px; background:#e2e8f0; margin:24px 0'></div>", unsafe_allow_html=True)

# ── Results (centered) ─────────────────────────────────────────────────────────
_, results_col, _ = st.columns([0.3, 5, 0.3])

with results_col:

    # ── Submit logic
    if submit:
        if not user_input and not uploaded_file:
            st.warning("Please type a business question or attach a file before running.")
        else:
            has_file = uploaded_file is not None
            with st.spinner("Routing and processing your request…"):
                st.session_state.request_type  = classify_request(user_input or "analyze this file", has_file)
                st.session_state.last_input    = user_input
                st.session_state.requestor     = ""
                if has_file:
                    st.session_state.uploaded_df   = pd.read_csv(uploaded_file, encoding="latin1") if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                    st.session_state.uploaded_name = uploaded_file.name

    # ── No result yet ─────────────────────────────────────────────────────────
    if not st.session_state.request_type:
        st.markdown("""
            <div style='
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 55vh;
                text-align: center;
            '>
                <div style='font-size:52px; margin-bottom:18px; opacity:0.18'>⚡</div>
                <div style='font-size:15px; font-weight:700; color:#94a3b8; margin-bottom:10px; letter-spacing:2px; text-transform:uppercase;'>
                    Ready for Analysis
                </div>
                <div style='font-size:13px; color:#b0bec5; max-width:380px; line-height:1.9;'>
                    Enter a business question above and click
                    <strong style='color:#2563eb'>Run Analysis</strong> to generate
                    an instant, AI-powered report.
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.request_type:
        request_type = st.session_state.request_type
        saved_input  = st.session_state.last_input

        # ── Request info bar
        type_meta = {
            "SQL_PULL":      ("DATA PULL",          "dp-kpi-chip-blue"),
            "WHY_QUESTION":  ("ROOT CAUSE ANALYSIS", "dp-kpi-chip-blue"),
            "FILE_ANALYSIS": ("FILE ANALYSIS",       "dp-kpi-chip-blue"),
        }
        label, chip_cls = type_meta.get(request_type, ("UNKNOWN", "dp-kpi-chip-gray"))
        st.markdown(f"""
            <div class="dp-info-bar">
                <div class="dp-info-bar-item">
                    <span class="dp-info-bar-label">Type</span>
                    <span class="dp-info-bar-val">{label}</span>
                </div>
                <div class="dp-info-bar-item">
                    <span class="dp-info-bar-label">Query</span>
                    <span class="dp-info-bar-query">{saved_input}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── FILE ANALYSIS ──────────────────────────────────────────────────────
        if request_type == "FILE_ANALYSIS":
            df   = st.session_state.uploaded_df
            name = st.session_state.uploaded_name

            st.markdown("<div class='dp-section-header'>FILE SUMMARY</div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="dp-kpi-row">
                    <div class="dp-kpi-chip dp-kpi-chip-gray">
                        <span>{name}</span>
                    </div>
                    <div class="dp-kpi-chip dp-kpi-chip-blue">
                        <span class="dp-kpi-chip-val">{df.shape[0]:,}</span>
                        <span style="font-size:11px; font-weight:500; opacity:0.8">ROWS</span>
                    </div>
                    <div class="dp-kpi-chip dp-kpi-chip-blue">
                        <span class="dp-kpi-chip-val">{df.shape[1]}</span>
                        <span style="font-size:11px; font-weight:500; opacity:0.8">COLUMNS</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("Preview Data — First 10 Rows"):
                st.dataframe(df.head(10), use_container_width=True, hide_index=True)

            with st.spinner("Running AI analysis on uploaded file…"):
                insight, chart, specific_df, specific_chart, answered, query_meta = handle_file(df, name, saved_input)

            # ── Specific question answer (from file data directly) ─────────────
            if answered and specific_df is not None:
                st.markdown("<div class='dp-section-header'>QUERY RESULTS FROM FILE</div>", unsafe_allow_html=True)

                requested  = query_meta.get("requested")
                available  = query_meta.get("available", len(specific_df))
                returned   = len(specific_df)

                note = ""
                if requested and available < requested:
                    note = f"<span style='font-size:11px;color:#f59e0b;font-weight:600;margin-left:12px;'>⚠ Only {available} unique groups in data (top {requested} requested)</span>"

                st.markdown(f"""
                    <div class="dp-kpi-row">
                        <div class="dp-kpi-chip dp-kpi-chip-green">
                            <span class="dp-kpi-chip-val">{returned:,}</span>
                            <span style="font-size:11px; font-weight:500; opacity:0.8">RECORDS RETURNED</span>
                        </div>
                        <div class="dp-kpi-chip dp-kpi-chip-gray">
                            <span class="dp-kpi-chip-val">{available:,}</span>
                            <span style="font-size:11px; font-weight:500; opacity:0.8">TOTAL GROUPS</span>
                        </div>
                        {note}
                    </div>
                """, unsafe_allow_html=True)

                st.dataframe(specific_df, use_container_width=True, hide_index=True)

                # Show the operation used (equivalent to SQL)
                if len(specific_df.columns) >= 2:
                    group_c  = specific_df.columns[0]
                    metric_c = specific_df.columns[1]
                    op = "AVG" if "avg" in metric_c.lower() else ("COUNT" if "count" in metric_c.lower() else "SUM")
                    limit_clause = f"TOP {requested}" if requested else "ALL"
                    pseudo_query = f"SELECT {group_c}, {op}(...) AS [{metric_c}]\nFROM uploaded_file\nGROUP BY {group_c}\nORDER BY [{metric_c}] DESC\n— {limit_clause} —"
                    with st.expander("🔍  View Operation Used"):
                        st.code(pseudo_query, language="sql")

                if specific_chart:
                    st.plotly_chart(specific_chart, use_container_width=True)
                st.markdown("<div class='dp-divider'></div>", unsafe_allow_html=True)

            # ── AI file insight ────────────────────────────────────────────────
            st.markdown("<div class='dp-section-header'>FILE INSIGHTS</div>", unsafe_allow_html=True)
            st.markdown(insight)

            if chart and not specific_chart:
                st.markdown("<div class='dp-divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='dp-section-header'>AUTO-GENERATED CHART</div>", unsafe_allow_html=True)
                st.plotly_chart(chart, use_container_width=True)

        # ── SQL PULL ───────────────────────────────────────────────────────────
        elif request_type == "SQL_PULL":
            with st.spinner("Generating SQL and pulling data…"):
                result_df, sql_used, error, chart = handle_sql_request(saved_input)

            if error:
                st.markdown(f"""
                    <div class="dp-card-danger">
                        <div style="font-size:10px; font-weight:700; letter-spacing:1.5px;
                                    text-transform:uppercase; color:#dc2626; margin-bottom:8px;">
                            Query Error
                        </div>
                        <div style="font-size:13px; color:#64748b; line-height:1.65;">{error}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='dp-section-header'>QUERY RESULTS</div>", unsafe_allow_html=True)

                # KPI chips
                st.markdown(f"""
                    <div class="dp-kpi-row">
                        <div class="dp-kpi-chip dp-kpi-chip-green">
                            <span class="dp-kpi-chip-val">{len(result_df):,}</span>
                            <span style="font-size:11px; font-weight:500; opacity:0.8">RECORDS FOUND</span>
                        </div>
                        <div class="dp-kpi-chip dp-kpi-chip-gray">
                            <span class="dp-kpi-chip-val">{len(result_df.columns)}</span>
                            <span style="font-size:11px; font-weight:500; opacity:0.8">COLUMNS</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Format columns
                display_df = result_df.copy()
                for col in display_df.columns:
                    col_lower = col.lower()
                    try:
                        if any(k in col_lower for k in ["sales", "profit", "revenue", "amount", "($)"]):
                            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "-")
                        elif any(k in col_lower for k in ["discount", "margin", "(%)"]):
                            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "-")
                        elif any(k in col_lower for k in ["units", "quantity", "count", "orders", "customers"]):
                            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "-")
                    except Exception:
                        pass

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                if chart:
                    st.markdown("<div class='dp-divider'></div>", unsafe_allow_html=True)
                    st.markdown("<div class='dp-section-header'>VISUAL SUMMARY</div>", unsafe_allow_html=True)
                    st.plotly_chart(chart, use_container_width=True)

                st.markdown("<div class='dp-divider'></div>", unsafe_allow_html=True)
                col_exp, col_dl = st.columns([1, 1])
                with col_exp:
                    with st.expander("View Generated SQL"):
                        st.code(sql_used, language="sql")
                with col_dl:
                    st.download_button(
                        label="Download Results as CSV",
                        data=result_df.to_csv(index=False),
                        file_name="results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        # ── WHY QUESTION ───────────────────────────────────────────────────────
        elif request_type == "WHY_QUESTION":

            # KPI selector with labels
            kpi_col, _ = st.columns([2, 3])
            with kpi_col:
                st.markdown("<div class='dp-section-header'>KPI FOCUS</div>", unsafe_allow_html=True)
                selected_kpi = st.radio(
                    label="KPI",
                    options=["Sales", "Profit", "Quantity", "Discount"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="kpi_selector"
                )
            st.markdown("<div class='dp-divider'></div>", unsafe_allow_html=True)

            with st.spinner(f"Analysing {selected_kpi} performance…"):
                try:
                    structured, comparison_df, drill_df, chart, sql_queries, raw_df = handle_why_request(saved_input, selected_kpi.lower())

                    intent      = structured.get("_intent", "worst")
                    focus_group = structured.get("worst_performer", "")
                    is_best     = intent == "best"

                    # Dynamic label strings
                    perf_label     = "TOP PERFORMER" if is_best else "UNDERPERFORMER"
                    perf_color     = "#059669" if is_best else "#dc2626"
                    perf_subtext   = "ranked as the top performer" if is_best else "flagged as underperformer"
                    drivers_header = "KEY DRIVERS" if is_best else "ROOT CAUSES"
                    rec_header     = "HOW OTHERS CAN REPLICATE" if is_best else "RECOMMENDED ACTIONS"
                    bench_label    = "Competitive Gap" if is_best else "Best Performer Benchmark"
                    bench_icon     = "📈" if is_best else "🏆"

                    # ── Executive Summary
                    if structured.get("summary"):
                        st.markdown("<div class='dp-section-header'>EXECUTIVE SUMMARY</div>", unsafe_allow_html=True)

                        # Focus group KPI badge
                        badge_color = "#f0fdf4" if is_best else "#fff5f5"
                        badge_border = "#bbf7d0" if is_best else "#fecaca"
                        st.markdown(f"""
                            <div style="background:{badge_color}; border:1px solid {badge_border};
                                        border-left:4px solid {perf_color}; border-radius:10px;
                                        padding:18px 22px; margin-bottom:16px;">
                                <div style="font-size:11px; font-weight:700; letter-spacing:1.5px;
                                            text-transform:uppercase; color:{perf_color}; margin-bottom:6px;">
                                    {perf_label} &nbsp;·&nbsp; {focus_group}
                                </div>
                                <div style="font-size:15px; color:#0f172a; line-height:1.8; font-weight:400;">
                                    {structured["summary"]}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<div class='dp-divider'></div>", unsafe_allow_html=True)

                    # ── Performance Table + Chart side by side
                    if comparison_df is not None:
                        st.markdown("<div class='dp-section-header'>PERFORMANCE COMPARISON</div>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div style="font-size:13px; color:#64748b; margin-bottom:14px; line-height:1.7;">
                                Full comparison across all groups —
                                <strong style="color:{perf_color};">{focus_group}</strong>
                                {perf_subtext}.
                            </div>
                        """, unsafe_allow_html=True)

                        tbl_col, chart_col = st.columns([1, 1], gap="large")
                        with tbl_col:
                            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        with chart_col:
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)

                    st.markdown("<div class='dp-divider'></div>", unsafe_allow_html=True)

                    # ── Key Findings in 2-col grid
                    if structured.get("findings"):
                        st.markdown("<div class='dp-section-header'>KEY FINDINGS</div>", unsafe_allow_html=True)
                        findings = structured["findings"]
                        for row_start in range(0, len(findings), 2):
                            cols = st.columns(2, gap="medium")
                            for col_idx, finding in enumerate(findings[row_start:row_start + 2]):
                                abs_idx = row_start + col_idx + 1
                                with cols[col_idx]:
                                    st.markdown(f"""
                                        <div class="dp-finding">
                                            <div class="dp-finding-number">Finding {abs_idx:02d}</div>
                                            <div class="dp-finding-title">{finding.get("title", "")}</div>
                                            <div class="dp-finding-body">{finding.get("explanation", "")}</div>
                                        </div>
                                    """, unsafe_allow_html=True)

                        # Category drill-down directly below findings
                        if drill_df is not None:
                            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
                            st.markdown(f"""
                                <div style="font-size:11px; font-weight:700; letter-spacing:1.5px;
                                            text-transform:uppercase; color:#94a3b8; margin-bottom:8px;">
                                    Category Breakdown — {focus_group}
                                </div>
                            """, unsafe_allow_html=True)
                            st.dataframe(drill_df, use_container_width=True, hide_index=True)

                    st.markdown("<div class='dp-divider'></div>", unsafe_allow_html=True)

                    # ── Drivers / Root Causes + Recommendations side by side
                    rc_col, rec_col = st.columns(2, gap="large")

                    with rc_col:
                        if structured.get("root_causes"):
                            st.markdown(f"<div class='dp-section-header'>{drivers_header}</div>", unsafe_allow_html=True)
                            items_html = "".join([
                                f"<div class='dp-list-item'><div class='dp-list-dot'></div><div>{c}</div></div>"
                                for c in structured["root_causes"]
                            ])
                            st.markdown(f"<div class='dp-card'>{items_html}</div>", unsafe_allow_html=True)

                    with rec_col:
                        if structured.get("recommendations"):
                            st.markdown(f"<div class='dp-section-header'>{rec_header}</div>", unsafe_allow_html=True)
                            items_html = "".join([
                                f"<div class='dp-list-item'><div class='dp-list-num'>{i:02d}</div><div>{r}</div></div>"
                                for i, r in enumerate(structured["recommendations"], 1)
                            ])
                            st.markdown(f"<div class='dp-card'>{items_html}</div>", unsafe_allow_html=True)

                    # ── Benchmark / Gap banner
                    if structured.get("benchmark"):
                        st.markdown(f"""
                            <div class="dp-benchmark">
                                <div class="dp-benchmark-icon">{bench_icon}</div>
                                <div>
                                    <div class="dp-benchmark-label">{bench_label}</div>
                                    <div class="dp-benchmark-text">{structured["benchmark"]}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<div class='dp-divider'></div>", unsafe_allow_html=True)

                    # ── SQL + Downloads side by side
                    exp_col1, exp_col2 = st.columns(2, gap="medium")
                    with exp_col1:
                        with st.expander("🗄️  SQL Query Used"):
                            if sql_queries:
                                st.code(sql_queries[0], language="sql")
                    with exp_col2:
                        with st.expander("📥  Download Supporting Data"):
                            focus_dl = focus_group
                            tab1, tab2, tab3 = st.tabs(["Comparison Table", "Category Breakdown", "Raw Data"])
                            with tab1:
                                if comparison_df is not None:
                                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                                    st.download_button("⬇ Download CSV", comparison_df.to_csv(index=False), "comparison_table.csv", "text/csv", use_container_width=True)
                            with tab2:
                                if drill_df is not None:
                                    st.dataframe(drill_df, use_container_width=True, hide_index=True)
                                    st.download_button("⬇ Download CSV", drill_df.to_csv(index=False), "category_breakdown.csv", "text/csv", use_container_width=True)
                            with tab3:
                                if raw_df is not None:
                                    st.caption(f"{len(raw_df):,} total rows — showing first 10")
                                    st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)
                                    st.download_button(
                                        f"⬇ Download All {len(raw_df):,} Rows",
                                        raw_df.to_csv(index=False),
                                        f"raw_{focus_dl.lower().replace(' ', '_')}.csv",
                                        "text/csv",
                                        use_container_width=True
                                    )

                except Exception as e:
                    st.markdown(f"""
                        <div class="dp-card-danger">
                            <div style="font-size:10px; font-weight:700; letter-spacing:1.5px;
                                        text-transform:uppercase; color:#dc2626; margin-bottom:8px;">
                                Analysis Error
                            </div>
                            <div style="font-size:13px; color:#64748b; line-height:1.65;">{str(e)}</div>
                        </div>
                    """, unsafe_allow_html=True)

        # ── UNKNOWN ────────────────────────────────────────────────────────────
        else:
            st.markdown("""
                <div class="dp-card">
                    <div style="font-size:10px; font-weight:700; letter-spacing:2px;
                                text-transform:uppercase; color:#dc2626; margin-bottom:12px;">
                        Unrecognized Request Format
                    </div>
                    <div style="font-size:13px; color:#64748b; line-height:2.1;">
                        Please rephrase your question using one of these patterns:<br>
                        <span style="color:#0f172a; font-weight:600;">DATA PULL</span> &nbsp;—&nbsp;
                        <span style="color:#475569;">"Show me sales by region"</span><br>
                        <span style="color:#0f172a; font-weight:600;">ROOT CAUSE</span> &nbsp;—&nbsp;
                        <span style="color:#475569;">"Why is profit down this quarter?"</span><br>
                        <span style="color:#0f172a; font-weight:600;">FILE ANALYSIS</span> &nbsp;—&nbsp;
                        <span style="color:#475569;">Upload a CSV or Excel file via the panel</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # ── Follow-up Section ─────────────────────────────────────────────────
        st.markdown("<div class='dp-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='dp-section-header'>FOLLOW-UP & CONTINUE ANALYSIS</div>", unsafe_allow_html=True)

        # Render all previous exchanges as a live chat thread
        if st.session_state.followup_history:
            for exchange in st.session_state.followup_history:
                # Question bubble
                st.markdown(f"""
                    <div class="dp-followup-q">
                        <span class="dp-followup-q-label">YOU ASKED</span>
                        {exchange["question"]}
                    </div>
                """, unsafe_allow_html=True)
                # Answer label
                st.markdown('<div class="dp-followup-a-label">DATAPULSE</div>', unsafe_allow_html=True)
                # Render: file result table OR plain markdown answer
                with st.container():
                    if exchange.get("answer") == "_FILE_RESULT_":
                        res_df    = exchange.get("result_df")
                        res_chart = exchange.get("result_chart")
                        if res_df is not None:
                            st.markdown(f"<span style='font-size:12px;color:#64748b;'>{len(res_df):,} records from uploaded file</span>", unsafe_allow_html=True)
                            st.dataframe(res_df, use_container_width=True, hide_index=True)
                        if res_chart:
                            st.plotly_chart(res_chart, use_container_width=True)
                    else:
                        st.markdown(exchange["answer"])
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Fresh input always at bottom — counter key forces widget reset after each ask
        st.markdown("""
            <div class="dp-followup-prompt-label">💬 &nbsp; Continue the conversation</div>
        """, unsafe_allow_html=True)

        fu_input_col, fu_btn_col = st.columns([5, 1], gap="small")
        with fu_input_col:
            followup_q = st.text_area(
                label="followup",
                placeholder="e.g. 'Which sub-category is hurting profit most?' or 'Compare this to last year'\n\nType your follow-up question here — you can write multiple lines.",
                label_visibility="collapsed",
                height=90,
                key=f"fu_widget_{st.session_state.followup_counter}"
            )
        with fu_btn_col:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            ask_followup = st.button("➤  Ask", type="primary", use_container_width=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("＋  New", use_container_width=True):
                for key in ["request_type", "last_input", "uploaded_df", "uploaded_name",
                            "typed_input", "followup_history", "followup_counter"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Handle follow-up submission
        if ask_followup and followup_q.strip():
            with st.spinner("Analysing…"):
                try:
                    has_uploaded_file = st.session_state.get("uploaded_df") is not None

                    # ── FILE MODE: answer directly from uploaded dataframe ─────
                    if has_uploaded_file:
                        from handlers.file_handler import _answer_from_file
                        uploaded_df   = st.session_state.uploaded_df
                        uploaded_name = st.session_state.get("uploaded_name", "uploaded file")

                        ans = _answer_from_file(uploaded_df, followup_q)
                        spec_df, spec_chart, answered = ans[0], ans[1], ans[2]

                        if answered and spec_df is not None:
                            # Store result df/chart in exchange for rendering
                            st.session_state.followup_history.append({
                                "question": followup_q,
                                "answer":   f"_FILE_RESULT_",   # sentinel
                                "result_df": spec_df,
                                "result_chart": spec_chart,
                            })
                            st.session_state.followup_counter += 1
                            st.rerun()
                        else:
                            # File loaded but question not answerable via pandas —
                            # build compact stats context and ask LLM
                            numeric_cols = uploaded_df.select_dtypes(include="number").columns.tolist()
                            col_ranges = []
                            for c in numeric_cols[:6]:
                                try:
                                    col_ranges.append(
                                        f"{c}: min={uploaded_df[c].min():.2f}, max={uploaded_df[c].max():.2f}, mean={uploaded_df[c].mean():.2f}"
                                    )
                                except Exception:
                                    pass
                            file_context = (
                                f"Uploaded file: {uploaded_name} ({uploaded_df.shape[0]:,} rows, {uploaded_df.shape[1]} cols). "
                                f"Columns: {list(uploaded_df.columns)}. "
                                f"Stats: {'; '.join(col_ranges)}"
                            )
                            from groq import Groq as _Groq
                            from utils.config import get_groq_api_key as _get_key
                            _client = _Groq(api_key=_get_key())
                            messages = [{"role": "system", "content": (
                                "You are a senior data analyst. Answer only from the uploaded file context provided. "
                                "Do NOT reference Superstore or any other dataset. "
                                "Use bullet points and **bold** key numbers. Max 200 words."
                            )}]
                            for ex in st.session_state.followup_history[-4:]:
                                messages.append({"role": "user",      "content": ex["question"]})
                                messages.append({"role": "assistant", "content": ex.get("answer", "")})
                            messages.append({"role": "user", "content":
                                f"File context: {file_context}\n\nQuestion: {followup_q}"
                            })
                            resp = _client.chat.completions.create(
                                model="llama-3.1-8b-instant", messages=messages, temperature=0.3, max_tokens=400
                            )
                            answer = resp.choices[0].message.content.strip()
                            st.session_state.followup_history.append({"question": followup_q, "answer": answer})
                            st.session_state.followup_counter += 1
                            st.rerun()

                    # ── SUPERSTORE MODE: use LLM with superstore context ───────
                    else:
                        from groq import Groq as _Groq
                        from utils.config import get_groq_api_key as _get_key
                        _client = _Groq(api_key=_get_key())

                        context_parts = [f"Original question: {st.session_state.last_input}"]
                        if st.session_state.request_type == "WHY_QUESTION":
                            context_parts.append("Root cause / performance analysis on the Superstore dataset.")
                        elif st.session_state.request_type == "SQL_PULL":
                            context_parts.append("Data pull from the Superstore dataset (sales, profit, orders).")

                        messages = [{"role": "system", "content": (
                            "You are a senior data analyst. Answer concisely and analytically. "
                            "Use bullet points and **bold** key numbers. Max 200 words. "
                            "Only cite real numbers from the Superstore dataset context provided."
                        )}]
                        for ex in st.session_state.followup_history[-4:]:
                            messages.append({"role": "user",      "content": ex["question"]})
                            messages.append({"role": "assistant", "content": ex.get("answer", "")})
                        messages.append({"role": "user", "content":
                            f"Context: {' | '.join(context_parts)}\n\nQuestion: {followup_q}"
                        })
                        resp = _client.chat.completions.create(
                            model="llama-3.1-8b-instant", messages=messages, temperature=0.3, max_tokens=400
                        )
                        answer = resp.choices[0].message.content.strip()
                        st.session_state.followup_history.append({"question": followup_q, "answer": answer})
                        st.session_state.followup_counter += 1
                        st.rerun()

                except Exception as e:
                    st.error(f"Follow-up failed: {str(e)}")

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown("""
            <div class="dp-footer">
                DataPulse &nbsp;&middot;&nbsp;
                <span>Powered by Groq + LLaMA 3.1</span>
                &nbsp;&middot;&nbsp;
                <span>Built with Streamlit</span>
                &nbsp;&middot;&nbsp;
                Internal Use Only
            </div>
        """, unsafe_allow_html=True)
