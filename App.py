import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
import base64
from datetime import datetime
import os

# -------------------------
# Configuration / constants
# -------------------------
PAGE_TITLE = "Police Stop Outcome Predictor"
PAGE_ICON = "🚨"
MODEL_PATH = "police_arrest_model.pkl"

CREDENTIALS = {
    "admin": "password123",
    "user": "userpass"
}

# -------------------------
# Global CSS — Navy Blue Premium Theme
# -------------------------
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --navy-950: #050d1a;
    --navy-900: #0a1628;
    --navy-800: #0f2044;
    --navy-700: #163060;
    --navy-600: #1e4080;
    --navy-500: #2651a0;
    --navy-400: #3b6bbf;
    --accent:   #4f8ef7;
    --accent-glow: #4f8ef740;
    --gold:     #f0b429;
    --gold-soft: #f0b42920;
    --success:  #10b981;
    --danger:   #ef4444;
    --text-primary:   #ffffff;
    --text-secondary: #cccccc;
    --text-muted:     #888888;
    --glass: rgba(14, 30, 60, 0.7);
    --glass-border: rgba(79, 142, 247, 0.15);
    --card-shadow: 0 8px 32px rgba(5, 13, 26, 0.6), 0 0 0 1px var(--glass-border);
}

/* ── Base Reset ── */
* { box-sizing: border-box; }

html, body, .stApp {
    background: var(--navy-950) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Animated starfield background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(30, 64, 128, 0.35) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(79, 142, 247, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse 40% 60% at 50% 50%, rgba(10, 22, 40, 0.8) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ── Typography ── */
h1, h2, h3, h4, h5 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy-900) !important;
    border-right: 1px solid var(--glass-border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Main Content Area ── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1100px !important;
}

/* ── Cards & Containers ── */
[data-testid="stForm"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 1.8rem !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: var(--card-shadow) !important;
}

/* ── Input Fields ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(10, 22, 40, 0.8) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(10, 22, 40, 0.8) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    border: 2px solid var(--navy-950) !important;
    box-shadow: 0 0 12px var(--accent-glow) !important;
}

[data-testid="stSlider"] > div > div > div {
    background: var(--navy-700) !important;
}

[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--navy-400)) !important;
}

/* ── Radio Buttons ── */
[data-testid="stRadio"] label {
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-secondary) !important;
}

/* ── Buttons ── */
[data-testid="stFormSubmitButton"] button,
.stButton > button {
    background: linear-gradient(135deg, var(--navy-600), var(--accent)) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.8rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 18px rgba(79, 142, 247, 0.35) !important;
    cursor: pointer !important;
}

[data-testid="stFormSubmitButton"] button:hover,
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(79, 142, 247, 0.55) !important;
    background: linear-gradient(135deg, var(--accent), var(--navy-400)) !important;
}

/* Primary button */
[data-testid="stButton"][data-baseweb="button"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--navy-500), var(--accent)) !important;
}

/* ── Alerts / Info / Success / Error ── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid !important;
    font-family: 'DM Sans', sans-serif !important;
}

div[data-baseweb="notification"][kind="positive"],
.stSuccess {
    background: rgba(16, 185, 129, 0.12) !important;
    border-color: rgba(16, 185, 129, 0.35) !important;
    color: #6ee7b7 !important;
}

div[data-baseweb="notification"][kind="negative"],
.stError {
    background: rgba(239, 68, 68, 0.12) !important;
    border-color: rgba(239, 68, 68, 0.35) !important;
    color: #fca5a5 !important;
}

div[data-baseweb="notification"][kind="info"],
.stInfo {
    background: rgba(79, 142, 247, 0.1) !important;
    border-color: rgba(79, 142, 247, 0.3) !important;
    color: var(--text-secondary) !important;
}

/* ── Progress Bar ── */
[data-testid="stProgress"] > div > div {
    background: var(--navy-800) !important;
    border-radius: 99px !important;
    overflow: hidden !important;
}

[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--gold)) !important;
    border-radius: 99px !important;
    transition: width 0.6s ease !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(10, 22, 40, 0.6) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Markdown / Text ── */
p, li, span, label {
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stMarkdown strong {
    color: var(--text-primary) !important;
}

/* ── Header ── */
.stApp header {
    background: transparent !important;
}

/* ── Divider ── */
hr {
    border-color: var(--glass-border) !important;
    margin: 1.2rem 0 !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
}

/* ── Download Link ── */
a {
    color: var(--accent) !important;
    text-decoration: none !important;
    font-weight: 500 !important;
    transition: color 0.2s !important;
}

a:hover {
    color: var(--gold) !important;
    text-decoration: underline !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy-900); }
::-webkit-scrollbar-thumb {
    background: var(--navy-600);
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Login Page Specific ── */
.login-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 80vh;
}

.login-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 2.8rem 2.4rem;
    width: 100%;
    max-width: 420px;
    box-shadow: var(--card-shadow), 0 0 60px rgba(79, 142, 247, 0.08);
    backdrop-filter: blur(16px);
    animation: fadeUp 0.5s ease forwards;
}

.login-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--gold-soft);
    border: 1px solid rgba(240, 180, 41, 0.3);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-family: 'DM Sans', sans-serif;
    color: var(--gold) !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.login-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    line-height: 1.2 !important;
    margin-bottom: 0.4rem !important;
}

.login-subtitle {
    color: var(--text-muted) !important;
    font-size: 0.88rem !important;
    margin-bottom: 1.8rem !important;
}

.result-card {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.6rem;
    backdrop-filter: blur(12px);
    box-shadow: var(--card-shadow);
    animation: fadeIn 0.4s ease;
}

.stat-chip {
    display: inline-block;
    background: rgba(79, 142, 247, 0.12);
    border: 1px solid rgba(79, 142, 247, 0.25);
    border-radius: 99px;
    padding: 3px 12px;
    font-size: 0.8rem;
    color: var(--accent) !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    margin: 2px;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* Column labels */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin-bottom: 0.6rem;
}
</style>
"""

# -------------------------
# Utility functions
# -------------------------
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model file not found: '{path}'. Please ensure the model file is in the correct directory.")
        st.stop()
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

def make_feature_dict(age, gender, race, search, drugs):
    return {
        "driver_age": age,
        "search_conducted": 1 if search == "Yes" else 0,
        "drugs_related_stop": 1 if drugs == "Yes" else 0,
        "driver_gender_M": 1 if gender == "Male" else 0,
        "driver_race_Black": 1 if race == "Black" else 0,
        "driver_race_Hispanic": 1 if race == "Hispanic" else 0,
        "driver_race_Other": 1 if race == "Other" else 0,
        "driver_race_White": 1 if race == "White" else 0,
    }

def get_download_link(df: pd.DataFrame, filename="result.csv"):
    towrite = BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f"data:text/csv;base64,{b64}"
    return href

def format_percent(p):
    return f"{p * 100:.2f}%"

# -------------------------
# App setup
# -------------------------
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
model = load_model(MODEL_PATH)

# -------------------------
# Session state init
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# -------------------------
# LOGIN PAGE
# -------------------------
if not st.session_state.authenticated:

    st.markdown("""
    <div style="text-align:center; padding-top: 3rem;">
        <div class="login-badge">🔒 Secure Access Portal</div>
        <div class="login-title">Police Stop<br>Outcome Predictor</div>
        <div class="login-subtitle">Enter your credentials to access the prediction dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        username = st.text_input("👤  Username", placeholder="Enter username")
        password = st.text_input("🔑  Password", type="password", placeholder="Enter password")
        login_pressed = st.button("Sign In  →", use_container_width=True, type="primary")

        st.markdown("""
        <div style="text-align:center; margin-top:1rem;">
            <span style="color: #4d6a8a; font-size:0.8rem;">
                Demo — use <code style="color:#4f8ef7">admin</code> / <code style="color:#4f8ef7">password123</code>
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if login_pressed:
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

# -------------------------
# MAIN APP
# -------------------------
else:

    # ── Top Header Bar ──
    st.markdown(f"""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        background: rgba(14,30,60,0.7);
        border: 1px solid rgba(79,142,247,0.15);
        border-radius: 14px;
        padding: 1rem 1.6rem;
        margin-bottom: 1.6rem;
        backdrop-filter: blur(12px);
    ">
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:1.6rem;">🚨</span>
            <div>
                <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.15rem; color:#e8edf5;">
                    {PAGE_TITLE}
                </div>
                <div style="font-size:0.75rem; color:#4d6a8a; font-family:'DM Sans',sans-serif;">
                    ML-Powered Decision Support Tool
                </div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="
                background: rgba(16,185,129,0.12);
                border: 1px solid rgba(16,185,129,0.3);
                border-radius:99px; padding:4px 12px;
                font-size:0.75rem; color:#6ee7b7;
                font-family:'DM Sans',sans-serif;
            ">● Live</div>
            <div style="
                background: rgba(79,142,247,0.12);
                border: 1px solid rgba(79,142,247,0.25);
                border-radius:99px; padding:4px 12px;
                font-size:0.75rem; color:#4f8ef7;
                font-family:'DM Sans',sans-serif;
            ">👤 {st.session_state.get('username','')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("⚠️ Predictions are probabilistic. This tool is for demonstration and analysis purposes only.")
    st.markdown("---")

    # ── Layout ──
    left, right = st.columns([1, 1])
    submitted = False

    with left:
        st.markdown('<div class="section-label">📋 Stop Details Input</div>', unsafe_allow_html=True)

        with st.form("input_form", clear_on_submit=False):
            age = st.slider("Driver Age", min_value=16, max_value=90, value=28)
            gender = st.radio("Gender", ("Male", "Female"), horizontal=True)
            race = st.selectbox("Race / Ethnicity", ["White", "Black", "Hispanic", "Asian", "Other"])
            search = st.selectbox("Search Conducted?", ["No", "Yes"])
            drugs = st.selectbox("Drugs Related Stop?", ["No", "Yes"])
            notes = st.text_area("Notes (optional)", placeholder="Add any relevant notes about the stop...", help="Optional field")
            submitted = st.form_submit_button("🔍  Run Prediction Analysis", use_container_width=True)

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # ── Prediction Logic ──
    if submitted:
        feat = make_feature_dict(age=age, gender=gender, race=race, search=search, drugs=drugs)
        input_df = pd.DataFrame([feat])

        try:
            probs = model.predict_proba(input_df)[0]
            prob_no_arrest, prob_arrest = probs[0], probs[1]
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "username": st.session_state.get("username", ""),
            "age": age,
            "gender": gender,
            "race": race,
            "search_conducted": search,
            "drugs_related_stop": drugs,
            "prob_arrest": float(prob_arrest),
            "prob_no_arrest": float(prob_no_arrest),
            "notes": notes,
        }
        st.session_state.last_result = result

        with right:
            st.markdown('<div class="section-label">📊 Prediction Results</div>', unsafe_allow_html=True)

            if prob_arrest >= 0.5:
                st.markdown(f"""
                <div style="
                    background: rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.35);
                    border-radius:12px; padding:1rem 1.2rem; margin-bottom:1rem;
                    display:flex; align-items:center; gap:10px;
                ">
                    <span style="font-size:1.4rem;">🔴</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif; font-weight:700; color:#fca5a5; font-size:1.05rem;">Likely Arrest</div>
                        <div style="color:#f87171; font-size:0.82rem;">High risk indicator detected</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="
                    background: rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.35);
                    border-radius:12px; padding:1rem 1.2rem; margin-bottom:1rem;
                    display:flex; align-items:center; gap:10px;
                ">
                    <span style="font-size:1.4rem;">🟢</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif; font-weight:700; color:#6ee7b7; font-size:1.05rem;">Likely No Arrest</div>
                        <div style="color:#34d399; font-size:0.82rem;">Low risk indicator detected</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("""
            <div style="font-family:'DM Sans',sans-serif; font-size:0.82rem; color:#8ba3c4;
                        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">
                Confidence Scores
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(prob_arrest), text=f"🔴 Arrest Probability: {format_percent(prob_arrest)}")
            st.progress(float(prob_no_arrest), text=f"🟢 No-Arrest Probability: {format_percent(prob_no_arrest)}")

            # Stats chips
            st.markdown(f"""
            <div style="margin: 1rem 0 0.6rem;">
                <span class="stat-chip">Age: {age}</span>
                <span class="stat-chip">Gender: {gender}</span>
                <span class="stat-chip">Race: {race}</span>
                <span class="stat-chip">Search: {search}</span>
                <span class="stat-chip">Drugs: {drugs}</span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔢 Numeric Breakdown"):
                st.write(f"**Arrest probability:** `{prob_arrest:.6f}`")
                st.write(f"**No-Arrest probability:** `{prob_no_arrest:.6f}`")
                st.write(f"**Timestamp (UTC):** `{result['timestamp']}`")

            if notes:
                st.markdown(f"""
                <div style="background:rgba(79,142,247,0.07); border:1px solid rgba(79,142,247,0.2);
                    border-radius:10px; padding:0.8rem 1rem; margin-top:0.8rem;
                    font-size:0.88rem; color:#8ba3c4; font-family:'DM Sans',sans-serif;">
                    📝 <strong style="color:#e8edf5;">Notes:</strong> {notes}
                </div>
                """, unsafe_allow_html=True)

            download_df = pd.DataFrame([result])
            csv_href = get_download_link(download_df, filename="police_stop_result.csv")
            st.markdown(f"""
            <div style="margin-top:1rem;">
                <a href="{csv_href}" download="police_stop_result.csv"
                   style="display:inline-flex; align-items:center; gap:6px;
                          background:rgba(79,142,247,0.12); border:1px solid rgba(79,142,247,0.3);
                          border-radius:8px; padding:8px 16px; color:#4f8ef7 !important;
                          font-family:'DM Sans',sans-serif; font-size:0.88rem; font-weight:500;
                          text-decoration:none; transition:all 0.2s;">
                    ⬇️ &nbsp;Download Result as CSV
                </a>
            </div>
            """, unsafe_allow_html=True)

            st.info("This prediction is produced by a trained ML model. Use responsibly.")

    else:
        with right:
            st.markdown('<div class="section-label">⏳ Awaiting Input</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="
                background: rgba(14,30,60,0.5); border:1px dashed rgba(79,142,247,0.2);
                border-radius:16px; padding:2.5rem 1.5rem; text-align:center; margin-top:0.5rem;
            ">
                <div style="font-size:2.5rem; margin-bottom:0.8rem;">🔍</div>
                <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700; color:#e8edf5;">
                    Ready to Analyze
                </div>
                <div style="font-size:0.85rem; color:#4d6a8a; margin-top:0.4rem; font-family:'DM Sans',sans-serif;">
                    Fill in the stop details on the left and click<br><strong style="color:#4f8ef7;">Run Prediction Analysis</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.last_result:
                lr = st.session_state.last_result
                st.markdown("""
                <div style="margin-top:1.2rem; font-family:'DM Sans',sans-serif;">
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
                                color:#4f8ef7; font-weight:700; margin-bottom:0.6rem;">Last Analysis</div>
                """, unsafe_allow_html=True)
                st.write(f"🕐 `{lr['timestamp']}`")
                st.write(f"👤 User: **{lr['username']}** | Age: **{lr['age']}** | {lr['gender']}")
                st.write(f"📊 Arrest probability: **{format_percent(lr['prob_arrest'])}**")
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Sidebar ──
    st.sidebar.markdown("""
    <div style="
        font-family:'Syne',sans-serif; font-weight:800; font-size:1rem;
        color:#e8edf5; margin-bottom:0.2rem;
    ">🚨 Session Controls</div>
    <div style="font-size:0.75rem; color:#4d6a8a; font-family:'DM Sans',sans-serif;
                margin-bottom:1.2rem;">Manage your active session</div>
    """, unsafe_allow_html=True)

    if st.sidebar.button("🚪  Log Out", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    <div style="font-family:'DM Sans',sans-serif;">
        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
                    color:#4f8ef7; font-weight:700; margin-bottom:0.8rem;">System Info</div>
        <div style="font-size:0.82rem; color:#4d6a8a; line-height:1.8;">
            Model Version: <span style="color:#8ba3c4;">1.0</span><br>
            Status: <span style="color:#6ee7b7;">● Online</span><br>
            Mode: <span style="color:#8ba3c4;">Demo Only</span>
        </div>
    </div>
    """, unsafe_allow_html=True)