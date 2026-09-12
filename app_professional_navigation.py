import streamlit as st
import tempfile
from pathlib import Path
import pandas as pd
import re
import hashlib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import extract_resume_text
from nlp_preprocessor import clean_text
from skill_matcher import analyze_skills


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TalentScreen AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# TALENTSCREEN AI — PROFESSIONAL UI THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #101a43;
    --text: #18234d;
    --muted: #64748b;
    --blue: #2f80ed;
    --purple: #5b4bdb;
    --green: #18a66a;
    --border: #dbe4f2;
    --surface: #ffffff;
    --page: #f5f8fd;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 75% 0%, rgba(91,75,219,.08), transparent 25%),
        radial-gradient(circle at 35% 20%, rgba(47,128,237,.06), transparent 28%),
        var(--page);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(255,255,255,.88);
    border-bottom: 1px solid #e7edf6;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1738 0%, #111d48 100%);
    border-right: 0;
}

[data-testid="stSidebar"] * {
    color: #e7edff;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #e7edff;
}

.sidebar-brand {
    padding: 8px 4px 22px;
}

.sidebar-brand .logo {
    font-size: 31px;
    margin-bottom: 6px;
}

.sidebar-brand .brand {
    font-size: 21px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -.03em;
}

.sidebar-brand .tagline {
    color: #aebce2;
    font-size: 11px;
    margin-top: 4px;
}

.sidebar-card {
    margin-top: 24px;
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,.10);
    background: rgba(255,255,255,.045);
}

.sidebar-card-title {
    color: #ffffff;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
}

.sidebar-check {
    color: #b9c8eb;
    font-size: 11px;
    line-height: 2;
}

.sidebar-footer {
    margin-top: 28px;
    color: #8494bd;
    font-size: 10px;
    line-height: 1.7;
}

/* Top bar */
.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
    color: #66738f;
    font-size: 12px;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 12px;
    border-radius: 999px;
    background: #e8f8f0;
    color: #087a4b;
    border: 1px solid #c7eedc;
    font-weight: 700;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #18a66a;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    padding: 30px 34px;
    border: 1px solid #dbe5f3;
    border-radius: 22px;
    background:
        radial-gradient(circle at 88% 15%, rgba(91,75,219,.15), transparent 28%),
        linear-gradient(135deg, #ffffff 0%, #f4f7ff 58%, #eef5ff 100%);
    box-shadow: 0 15px 40px rgba(28,50,91,.08);
    margin-bottom: 26px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -90px;
    top: -90px;
    border-radius: 50%;
    background: rgba(91,75,219,.08);
}

.hero-kicker {
    color: #5b4bdb;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .14em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.hero-title {
    color: #101a43;
    font-size: clamp(30px, 4vw, 48px);
    line-height: 1.06;
    font-weight: 800;
    letter-spacing: -.045em;
    margin: 0;
}

.hero-subtitle {
    color: #52617d;
    font-size: 14px;
    line-height: 1.7;
    margin-top: 11px;
    max-width: 850px;
}

/* Sections */
.section-heading {
    color: #142044;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -.025em;
    margin-top: 20px;
    margin-bottom: 4px;
}

.section-note {
    color: #71809a;
    font-size: 12px;
    margin-bottom: 12px;
}

/* Cards */
.panel {
    background: rgba(255,255,255,.92);
    border: 1px solid #dce5f2;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(31,55,91,.055);
    margin-bottom: 15px;
}

.panel-title {
    color: #17234c;
    font-size: 16px;
    font-weight: 800;
}

.panel-note {
    color: #71809a;
    font-size: 11px;
    line-height: 1.6;
    margin-top: 4px;
}

/* Metrics */
.metric-card {
    background: #ffffff;
    border: 1px solid #dce5f2;
    border-radius: 16px;
    padding: 16px;
    min-height: 104px;
    box-shadow: 0 8px 24px rgba(31,55,91,.05);
}

.metric-label {
    color: #71809a;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .08em;
}

.metric-value {
    color: #17234c;
    font-size: 23px;
    font-weight: 800;
    margin-top: 8px;
    letter-spacing: -.03em;
}

.metric-sub {
    color: #64748b;
    font-size: 10px;
    margin-top: 4px;
}

/* Top candidate */
.top-candidate {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    padding: 18px 22px;
    border-radius: 18px;
    border: 1px solid #cfe9dd;
    background: linear-gradient(135deg, #f5fffa, #ffffff);
    box-shadow: 0 8px 24px rgba(24,166,106,.07);
    margin: 18px 0 20px;
}

.top-eyebrow {
    color: #138a5a;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .10em;
    text-transform: uppercase;
}

.top-name {
    color: #17234c;
    font-size: 23px;
    font-weight: 800;
    margin-top: 3px;
}

.top-file {
    color: #71809a;
    font-size: 10px;
    margin-top: 3px;
}

.top-score {
    color: #12935e;
    font-size: 32px;
    font-weight: 800;
    white-space: nowrap;
}

/* Skill pills */
.skill-pill, .match-pill, .missing-pill {
    display: inline-block;
    padding: 6px 10px;
    margin: 3px 4px 3px 0;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
}

.skill-pill {
    background: #eef4ff;
    color: #315db5;
    border: 1px solid #d7e4ff;
}

.match-pill {
    background: #e9f8f0;
    color: #087a4b;
    border: 1px solid #c8eddc;
}

.missing-pill {
    background: #fff0f2;
    color: #bd3d54;
    border: 1px solid #ffd5dc;
}

/* Buttons */
.stButton > button {
    border-radius: 11px;
    min-height: 44px;
    font-weight: 800;
    border: 1px solid #d4def0;
    box-shadow: 0 5px 15px rgba(40,65,105,.05);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 9px 22px rgba(47,128,237,.14);
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #fbfdff;
    border: 1.5px dashed #b9cdf0;
    border-radius: 15px;
    padding: 7px;
}

[data-testid="stFileUploaderDropzone"] {
    background: #f5f9ff;
    border-radius: 10px;
}

.stTextArea textarea {
    background: #ffffff !important;
    color: #17234c !important;
    border: 1px solid #d5dfed !important;
    border-radius: 11px !important;
}

/* Radio controls */
[data-testid="stRadio"] label {
    color: #27365e !important;
    font-weight: 600 !important;
}

/* Tables and expanders */
[data-testid="stDataFrame"] {
    border: 1px solid #dce5f2;
    border-radius: 14px;
    overflow: hidden;
}

.stExpander {
    background: #ffffff;
    border: 1px solid #dce5f2;
    border-radius: 14px;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dce5f2;
    border-radius: 12px;
}

/* Footer */
.footer {
    text-align: center;
    color: #8290a8;
    font-size: 10px;
    padding: 25px 0 5px;
}

[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    color: #e7edff;
    border: 1px solid transparent;
    box-shadow: none;
    text-align: left;
    justify-content: flex-start;
    min-height: 38px;
    margin: 2px 0;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(91,113,220,.22);
    border-color: rgba(255,255,255,.10);
    transform: none;
    box-shadow: none;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# APP NAVIGATION
# ============================================================

if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"

def navigate_to(page):
    st.session_state.active_page = page

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo">🧠</div>
        <div class="brand">TalentScreen AI</div>
        <div class="tagline">Smarter Screening. Better Hiring.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")

    if st.button("◉  Dashboard", key="sidebar_dashboard", use_container_width=True):
        navigate_to("Dashboard")
        st.rerun()

    if st.button("📄  Screen Resumes", key="sidebar_screen", use_container_width=True):
        navigate_to("Screen Resumes")
        st.rerun()

    if st.button("📊  Results", key="sidebar_results", use_container_width=True):
        navigate_to("Results")
        st.rerun()

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">AI-Powered Hiring Intelligence</div>
        <div class="sidebar-check">● Semantic AI Matching</div>
        <div class="sidebar-check">● Skill Analysis</div>
        <div class="sidebar-check">● Experience Evaluation</div>
        <div class="sidebar-check">● Education Relevance</div>
        <div class="sidebar-check">● Explainable Results</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-footer">
        Model: all-MiniLM-L6-v2<br>
        Local processing • No paid API<br>
        Version 1.0 • Academic Prototype
    </div>
    """, unsafe_allow_html=True)





# ============================================================
# CANDIDATE NAME EXTRACTION
# ============================================================

def extract_candidate_name(resume_text, filename):
    """Extract candidate name from the resume."""

    lines = [
        line.strip()
        for line in resume_text.splitlines()
        if line.strip()
    ]

    # Check for explicit Name/Candidate labels
    for line in lines[:10]:

        match = re.search(
            r"^(?:name|candidate)\s*[:\-]\s*(.+)$",
            line,
            re.IGNORECASE
        )

        if match:
            return match.group(1).strip()

    # Check first few lines for a likely person's name
    for line in lines[:5]:

        cleaned = re.sub(
            r"[^A-Za-z .'-]",
            "",
            line
        ).strip()

        words = cleaned.split()

        if 2 <= len(words) <= 5:

            if all(
                word.replace("-", "").replace("'", "").isalpha()
                for word in words
            ):
                return cleaned.title()

    # Filename fallback
    return (
        Path(filename)
        .stem
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


# ============================================================
# EXPERIENCE SCORING
# ============================================================

def calculate_experience_score(resume_text):
    """
    Estimate experience relevance from resume text.
    """

    text = clean_text(resume_text)

    strong_experience_keywords = [
        "machine learning engineer",
        "machine learning developer",
        "machine learning intern",
        "machine learning project",
        "ml engineer",
        "ml developer",
        "ml intern",
        "data scientist",
        "data science intern",
        "data analyst",
        "data analysis"
    ]

    for keyword in strong_experience_keywords:

        if keyword in text:
            return 100.0

    technical_experience_keywords = [
        "software engineer",
        "software developer",
        "developer",
        "engineer",
        "internship",
        "intern",
        "work experience",
        "professional experience"
    ]

    for keyword in technical_experience_keywords:

        if keyword in text:
            return 70.0

    return 0.0


# ============================================================
# EDUCATION SCORING
# ============================================================

def calculate_education_score(resume_text):
    """
    Estimate education relevance based on degree and field.
    """

    text = clean_text(resume_text)

    highly_relevant = [
        "computer science",
        "computer applications",
        "data science",
        "artificial intelligence",
        "artificial intelligence machine learning",
        "ai ml",
        "machine learning"
    ]

    for keyword in highly_relevant:

        if keyword in text:
            return 100.0

    relevant_degrees = [
        "information technology",
        "information science",
        "software engineering",
        "computer engineering",
        "electronics and computer",
        "mathematics",
        "statistics"
    ]

    for keyword in relevant_degrees:

        if keyword in text:
            return 85.0

    general_degrees = [
        "bachelor",
        "bachelors",
        "bca",
        "bsc",
        "b tech",
        "btech",
        "master",
        "masters",
        "mca",
        "msc",
        "m tech",
        "mtech"
    ]

    for keyword in general_degrees:

        if keyword in text:
            return 70.0

    return 0.0


# ============================================================
# LOAD TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ============================================================
# APPLICATION TITLE
# ============================================================

st.markdown("""
<div class="topbar">
    <span>✦ Find the right talent with the power of AI</span>
    <span class="status"><span class="status-dot"></span> AI Engine Active</span>
</div>

<div class="hero">
    <div class="hero-kicker">Welcome to TalentScreen AI</div>
    <div class="hero-title">AI Resume Screening System</div>
    <div class="hero-subtitle">
        Turn resumes into real opportunities. Upload a job description and candidate
        resumes to get AI-powered insights, match scores, and candidate rankings.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# JOB DESCRIPTION
# ============================================================


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.active_page == "Dashboard":
    st.markdown('<div class="section-heading">Welcome to your screening workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Start a new screening or review your latest candidate analysis.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">📄 Screen Resumes</div>
            <div class="panel-note">
                Upload a job description and multiple resumes to run the AI screening pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Screening  →", key="dash_screen", use_container_width=True):
            navigate_to("Screen Resumes")
            st.rerun()

    with c2:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">📊 Results</div>
            <div class="panel-note">
                Review candidate rankings, scores, skill matches and screening insights.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Results  →", key="dash_results", use_container_width=True):
            navigate_to("Results")
            st.rerun()

    with c3:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">🧠 AI Screening Engine</div>
            <div class="panel-note">
                Transformer embeddings, semantic similarity, skill matching,
                experience and education analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-title">How TalentScreen AI Works</div>
        <div class="panel-note" style="font-size:12px;line-height:1.9;">
            <b>01</b> Job requirements → <b>02</b> Resume upload →
            <b>03</b> Transformer embeddings → <b>04</b> Similarity scoring →
            <b>05</b> Explainable candidate ranking
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# RESULTS PAGE
# ============================================================

if st.session_state.active_page == "Results":
    saved = st.session_state.get("screening_results")

    st.markdown('<div class="section-heading">03 · Results Overview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Your latest AI-powered candidate screening results.</div>',
        unsafe_allow_html=True
    )

    if not saved:
        st.info("No screening results are available yet. Run a screening first.")
        if st.button("Start Screening  →", key="results_start", type="primary", use_container_width=True):
            navigate_to("Screen Resumes")
            st.rerun()
        st.stop()

    top = saved[0]
    avg = sum(float(r["final"]) for r in saved) / len(saved)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Top Candidate", top["name"])
    with k2:
        st.metric("Top Score", f'{float(top["final"]):.2f}%')
    with k3:
        st.metric("Average Score", f"{avg:.2f}%")

    rows = []
    for rank, result in enumerate(saved, 1):
        rows.append({
            "Rank": rank,
            "Candidate": result["name"],
            "Semantic Match": f'{float(result["semantic"]):.2f}%',
            "Skills": f'{float(result["skills"]):.2f}%',
            "Experience": f'{float(result["experience"]):.2f}%',
            "Education": f'{float(result["education"]):.2f}%',
            "Final Score": f'{float(result["final"]):.2f}%'
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    result_chart = pd.DataFrame([
        {"Candidate": r["name"], "Final Score": float(r["final"])}
        for r in saved
    ])
    st.bar_chart(
        result_chart,
        x="Candidate",
        y="Final Score",
        y_label="Final Score (%)",
        height=360
    )

    if st.button("← Back to Screening", key="results_back", use_container_width=True):
        navigate_to("Screen Resumes")
        st.rerun()

    st.stop()


if st.session_state.active_page == "Screen Resumes":
    st.markdown('<div class="section-heading">01 · Job Description</div>', unsafe_allow_html=True)

    st.write(
        "Provide the job description either by uploading a document "
        "or by entering it manually."
    )

    job_input_method = st.radio(
        "Choose Job Description Input Method",
        [
            "📁 Upload Job Description",
            "✍️ Enter Job Description Manually"
        ],
        horizontal=True
    )

    job_description = ""


    # ============================================================
    # JOB DESCRIPTION FILE UPLOAD
    # ============================================================

    if job_input_method == "📁 Upload Job Description":

        uploaded_job_description = st.file_uploader(
            "Upload Job Description",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=False
        )

        if uploaded_job_description:

            temp_job_path = None

            try:

                suffix = Path(
                    uploaded_job_description.name
                ).suffix.lower()

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:

                    temp_file.write(
                        uploaded_job_description.getbuffer()
                    )

                    temp_job_path = temp_file.name

                job_description = extract_resume_text(
                    temp_job_path
                )

                if job_description.strip():

                    st.success(
                        f"Job description loaded successfully: "
                        f"{uploaded_job_description.name}"
                    )

                    with st.expander(
                        "👁️ Preview Job Description"
                    ):

                        st.write(
                            job_description
                        )

                else:

                    st.warning(
                        "The uploaded job description appears to be empty."
                    )

            except Exception as error:

                st.error(
                    f"Could not read the job description: {error}"
                )

            finally:

                if temp_job_path:

                    Path(
                        temp_job_path
                    ).unlink(
                        missing_ok=True
                    )


    # ============================================================
    # MANUAL JOB DESCRIPTION
    # ============================================================

    else:

        job_description = st.text_area(
            "Enter the Job Description",
            height=220,
            placeholder=(
                "Example:\n\n"
                "We are looking for a Machine Learning Engineer.\n"
                "Required skills: Python, Machine Learning, Data Analysis, "
                "Pandas, NumPy, Scikit-learn and SQL.\n"
                "Preferred skills: Deep Learning, TensorFlow and Data Visualization."
            )
        )


    # ============================================================
    # RESUME UPLOAD
    # ============================================================

    st.markdown('<div class="section-heading">02 · Candidate Resumes</div>', unsafe_allow_html=True)

    uploaded_resumes = st.file_uploader(
        "Upload candidate resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )


    # ============================================================
    # PREVIEW CANDIDATE RESUMES
    # ============================================================

    if uploaded_resumes:

        st.markdown("### 👁️ Preview Uploaded Resumes")
        st.markdown(
            '<div class="section-note">Review the extracted text from each resume before running the AI screening.</div>',
            unsafe_allow_html=True
        )

        preview_hashes = set()

        for resume_index, uploaded_file in enumerate(uploaded_resumes, start=1):

            preview_temp_path = None

            try:

                preview_suffix = Path(uploaded_file.name).suffix.lower()

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=preview_suffix
                ) as preview_temp_file:

                    preview_temp_file.write(uploaded_file.getbuffer())
                    preview_temp_path = preview_temp_file.name

                preview_text = extract_resume_text(preview_temp_path)

                if not preview_text or not preview_text.strip():
                    with st.expander(
                        f"#{resume_index} — {uploaded_file.name} · ⚠️ Empty / No readable text"
                    ):
                        st.warning(
                            "No readable text was found in this file. "
                            "It will be skipped during AI screening."
                        )
                    continue

                normalized_preview = re.sub(
                    r"\s+",
                    " ",
                    preview_text.lower()
                ).strip()
                preview_hash = hashlib.sha256(
                    normalized_preview.encode("utf-8")
                ).hexdigest()

                duplicate_preview = preview_hash in preview_hashes
                preview_hashes.add(preview_hash)

                candidate_preview_name = extract_candidate_name(
                    preview_text,
                    uploaded_file.name
                )

                status_text = "⚠️ Duplicate content" if duplicate_preview else "✅ Ready for screening"

                with st.expander(
                    f"#{resume_index} — {candidate_preview_name} · {status_text}"
                ):

                    st.caption(
                        f"File: {uploaded_file.name} · Size: {uploaded_file.size / 1024:.1f} KB"
                    )

                    if duplicate_preview:
                        st.warning(
                            "This resume has the same extracted content as an earlier uploaded resume. "
                            "It will be skipped as a duplicate during screening."
                        )

                    st.text_area(
                        "Extracted Resume Text",
                        value=preview_text,
                        height=260,
                        disabled=True,
                        key=f"resume_preview_{resume_index}"
                    )

            except Exception as preview_error:

                with st.expander(
                    f"#{resume_index} — {uploaded_file.name} · ❌ Cannot preview"
                ):
                    st.error(
                        f"This file could not be read or previewed: {preview_error}"
                    )
                    st.caption(
                        "The file will also be skipped during AI screening."
                    )

            finally:

                if preview_temp_path:
                    Path(preview_temp_path).unlink(missing_ok=True)


    # ============================================================
    # SCREEN RESUMES
    # ============================================================

    button_left, button_center, button_right = st.columns([1, 1.5, 1])
    with button_center:
        run_screening = st.button(
            "✨  ANALYZE CANDIDATES   →",
            type="primary",
            use_container_width=True
        )

    if run_screening:

        # --------------------------------------------------------
        # Validate Job Description
        # --------------------------------------------------------

        if not job_description.strip():

            st.warning(
                "Please provide a Job Description before screening resumes."
            )

            st.stop()


        # --------------------------------------------------------
        # Validate Resumes
        # --------------------------------------------------------

        if not uploaded_resumes:

            st.warning(
                "Please upload at least one resume."
            )

            st.stop()


        # --------------------------------------------------------
        # Prepare Job Description
        # --------------------------------------------------------

        cleaned_job = clean_text(
            job_description
        )

        job_embedding = model.encode(
            [cleaned_job]
        )

        results = []
        processed_resume_hashes = set()
        duplicate_count = 0
        empty_count = 0
        invalid_count = 0


        # ========================================================
        # PROCESS EACH RESUME
        # ========================================================

        with st.spinner(
            "Analyzing resumes using AI..."
        ):

            for uploaded_file in uploaded_resumes:

                temp_path = None

                try:

                    # ------------------------------------------------
                    # Create Temporary Resume File
                    # ------------------------------------------------

                    suffix = Path(
                        uploaded_file.name
                    ).suffix.lower()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = temp_file.name


                    # ------------------------------------------------
                    # Extract Resume Text
                    # ------------------------------------------------

                    resume_text = extract_resume_text(
                        temp_path
                    )

                    # ------------------------------------------------
                    # Validate Extracted Resume Text
                    # ------------------------------------------------
                    # A file can be technically valid but still contain
                    # no readable text (for example, an empty TXT file or
                    # an image-only/scanned PDF). Do not send such files
                    # to the AI model.
                    if not resume_text or not resume_text.strip():
                        empty_count += 1
                        st.warning(
                            f"Empty resume skipped: {uploaded_file.name}. "
                            "No readable text was found in this file."
                        )
                        continue


                    # ------------------------------------------------
                    # Detect Duplicate Resume
                    # ------------------------------------------------
                    # Normalize the extracted text and hash it. This
                    # catches the same resume even if uploaded again
                    # with a different filename.
                    normalized_resume = re.sub(
                        r"\s+",
                        " ",
                        resume_text.lower()
                    ).strip()
                    resume_hash = hashlib.sha256(
                        normalized_resume.encode("utf-8")
                    ).hexdigest()

                    if resume_hash in processed_resume_hashes:
                        duplicate_count += 1
                        st.warning(
                            f"Duplicate resume skipped: {uploaded_file.name}. "
                            "This resume has already been included in this screening."
                        )
                        continue

                    processed_resume_hashes.add(resume_hash)


                    # ------------------------------------------------
                    # Extract Candidate Name
                    # ------------------------------------------------

                    candidate_name = extract_candidate_name(
                        resume_text,
                        uploaded_file.name
                    )


                    # ------------------------------------------------
                    # Clean Resume
                    # ------------------------------------------------

                    cleaned_resume = clean_text(
                        resume_text
                    )


                    # ------------------------------------------------
                    # Generate Resume Embedding
                    # ------------------------------------------------

                    resume_embedding = model.encode(
                        [cleaned_resume]
                    )


                    # ------------------------------------------------
                    # Calculate Semantic Similarity
                    # ------------------------------------------------

                    similarity = cosine_similarity(
                        resume_embedding,
                        job_embedding
                    )[0][0]

                    similarity = float(
                        similarity
                    )

                    semantic_score = float(
                        similarity * 100
                    )


                    # ------------------------------------------------
                    # Skill Analysis
                    # ------------------------------------------------

                    skill_results = analyze_skills(
                        resume_text,
                        job_description
                    )

                    job_skills = skill_results[
                        "job_skills"
                    ]

                    matched_skills = skill_results[
                        "matched_skills"
                    ]

                    missing_skills = skill_results[
                        "missing_skills"
                    ]


                    # ------------------------------------------------
                    # Skill Score
                    # ------------------------------------------------

                    if len(job_skills) > 0:

                        skill_score = float(
                            (
                                len(matched_skills)
                                /
                                len(job_skills)
                            )
                            * 100
                        )

                    else:

                        skill_score = 0.0


                    # ------------------------------------------------
                    # Experience Score
                    # ------------------------------------------------

                    experience_score = (
                        calculate_experience_score(
                            resume_text
                        )
                    )


                    # ------------------------------------------------
                    # Education Score
                    # ------------------------------------------------

                    education_score = (
                        calculate_education_score(
                            resume_text
                        )
                    )


                    # ------------------------------------------------
                    # Final Weighted Score
                    # ------------------------------------------------
                    #
                    # Semantic Similarity = 60%
                    # Skills             = 25%
                    # Experience         = 10%
                    # Education          = 5%
                    #

                    final_score = float(

                        (semantic_score * 0.60)

                        +

                        (skill_score * 0.25)

                        +

                        (experience_score * 0.10)

                        +

                        (education_score * 0.05)

                    )


                    # ------------------------------------------------
                    # Store Result
                    # ------------------------------------------------

                    results.append({

                        "name": candidate_name,

                        "filename": uploaded_file.name,

                        "semantic": float(
                            semantic_score
                        ),

                        "skills": float(
                            skill_score
                        ),

                        "experience": float(
                            experience_score
                        ),

                        "education": float(
                            education_score
                        ),

                        "final": float(
                            final_score
                        ),

                        "job_skills": job_skills,

                        "matched_skills": matched_skills,

                        "missing_skills": missing_skills

                    })


                # ----------------------------------------------------
                # Error Handling
                # ----------------------------------------------------

                except Exception as error:

                    invalid_count += 1
                    st.error(
                        f"Invalid or corrupted resume skipped: "
                        f"{uploaded_file.name}. The file could not be read or processed."
                    )


                # ----------------------------------------------------
                # Delete Temporary File
                # ----------------------------------------------------

                finally:

                    if temp_path:

                        Path(
                            temp_path
                        ).unlink(
                            missing_ok=True
                        )


        # ========================================================
        # DUPLICATE SUMMARY
        # ========================================================

        # ========================================================
        # FILE PROCESSING SUMMARY
        # ========================================================

        skipped_count = duplicate_count + empty_count + invalid_count

        if skipped_count > 0:
            summary_parts = [
                f"{len(results)} valid",
            ]

            if duplicate_count > 0:
                summary_parts.append(f"{duplicate_count} duplicate skipped")

            if empty_count > 0:
                summary_parts.append(f"{empty_count} empty/unreadable skipped")

            if invalid_count > 0:
                summary_parts.append(f"{invalid_count} corrupted/invalid skipped")

            st.info(
                "File processing summary: " + " • ".join(summary_parts) + ". "
                "Only valid, unique resumes are included in the ranking."
            )


        # ========================================================
        # SORT RESULTS
        # ========================================================

        results.sort(
            key=lambda x: float(
                x["final"]
            ),
            reverse=True
        )

        st.session_state.screening_results = results


        # ========================================================
        # DISPLAY RESULTS
        # ========================================================

        if results:

            st.divider()

            st.markdown('<div class="section-heading">03 · Results Overview</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-note">AI-powered analysis and candidate ranking</div>',
                unsafe_allow_html=True
            )

            top_candidate = results[0]
            average_score = sum(float(r["final"]) for r in results) / len(results)
            average_skill_score = sum(float(r["skills"]) for r in results) / len(results)

            k1, k2, k3, k4 = st.columns(4)

            with k1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">TOTAL CANDIDATES</div>'
                    f'<div class="metric-value">{len(results)}</div>'
                    f'<div class="metric-sub">Resumes analyzed</div></div>',
                    unsafe_allow_html=True
                )

            with k2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">TOP CANDIDATE</div>'
                    f'<div class="metric-value">{top_candidate["name"]}</div>'
                    f'<div class="metric-sub">Final score: {float(top_candidate["final"]):.1f}%</div></div>',
                    unsafe_allow_html=True
                )

            with k3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">AVERAGE SCORE</div>'
                    f'<div class="metric-value">{average_score:.1f}%</div>'
                    f'<div class="metric-sub">Across all candidates</div></div>',
                    unsafe_allow_html=True
                )

            with k4:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">AVG. SKILL MATCH</div>'
                    f'<div class="metric-value">{average_skill_score:.1f}%</div>'
                    f'<div class="metric-sub">Detected job skills</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown(f"""
            <div class="top-candidate">
                <div>
                    <div class="top-eyebrow">★ Recommended Candidate</div>
                    <div class="top-name">{top_candidate["name"]}</div>
                    <div class="top-file">{top_candidate["filename"]}</div>
                </div>
                <div style="text-align:right;">
                    <div class="top-score">{float(top_candidate["final"]):.2f}%</div>
                    <div style="color:#71809a;font-size:10px;">Overall compatibility</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


            # ====================================================
            # DETECTED JOB SKILLS
            # ====================================================

            st.subheader(
                "🎯 Skills Detected From Job Description"
            )

            detected_skills = results[0][
                "job_skills"
            ]

            if detected_skills:
                pills = "".join(
                    f'<span class="skill-pill">{skill.title()}</span>'
                    for skill in detected_skills
                )
                st.markdown(pills, unsafe_allow_html=True)

            else:

                st.warning(
                    "No skills were detected from the job description."
                )


            # ====================================================
            # CANDIDATE RANKING
            # ====================================================

            st.markdown("### Candidate Ranking")
            st.markdown(
                '<div class="section-note">Weighted by semantic match 60% · skills 25% · experience 10% · education 5%</div>',
                unsafe_allow_html=True
            )

            ranking_data = []


            for rank, result in enumerate(
                results,
                start=1
            ):

                ranking_data.append({

                    "Rank": int(rank),

                    "Candidate": result[
                        "name"
                    ],

                    "Semantic Match": (
                        f"{float(result['semantic']):.2f}%"
                    ),

                    "Skills": (
                        f"{float(result['skills']):.2f}%"
                    ),

                    "Experience": (
                        f"{float(result['experience']):.2f}%"
                    ),

                    "Education": (
                        f"{float(result['education']):.2f}%"
                    ),

                    "Final Score": (
                        f"{float(result['final']):.2f}%"
                    )

                })


            st.dataframe(
                ranking_data,
                use_container_width=True,
                hide_index=True
            )


            # ====================================================
            # DOWNLOAD RESULTS
            # ====================================================

            st.markdown("### Export Screening Results")
            st.markdown(
                '<div class="section-note">Download the complete candidate screening report.</div>',
                unsafe_allow_html=True
            )

            download_data = []


            for rank, result in enumerate(
                results,
                start=1
            ):

                download_data.append({

                    "Rank": int(rank),

                    "Candidate": result[
                        "name"
                    ],

                    "Filename": result[
                        "filename"
                    ],

                    "Semantic Match (%)": round(
                        float(
                            result["semantic"]
                        ),
                        2
                    ),

                    "Skill Match (%)": round(
                        float(
                            result["skills"]
                        ),
                        2
                    ),

                    "Experience (%)": round(
                        float(
                            result["experience"]
                        ),
                        2
                    ),

                    "Education (%)": round(
                        float(
                            result["education"]
                        ),
                        2
                    ),

                    "Final Score (%)": round(
                        float(
                            result["final"]
                        ),
                        2
                    ),

                    "Matched Skills": ", ".join(
                        result[
                            "matched_skills"
                        ]
                    ),

                    "Missing Skills": ", ".join(
                        result[
                            "missing_skills"
                        ]
                    )

                })


            download_df = pd.DataFrame(
                download_data
            )


            csv_data = download_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            )


            st.download_button(

                label="📥 Download Results as CSV",

                data=csv_data,

                file_name=(
                    "AI_Resume_Screening_Results.csv"
                ),

                mime="text/csv",

                type="primary"

            )


            # ====================================================
            # SCORE COMPARISON CHART
            # ====================================================

            st.markdown("### Candidate Score Comparison")
            st.markdown(
                '<div class="section-note">Final match score comparison across analyzed candidates.</div>',
                unsafe_allow_html=True
            )

            chart_data = []


            for result in results:

                chart_data.append({

                    "Candidate": result[
                        "name"
                    ],

                    "Final Score": float(
                        result["final"]
                    )

                })


            chart_df = pd.DataFrame(
                chart_data
            )


            st.bar_chart(

                chart_df,

                x="Candidate",

                y="Final Score",

                y_label="Final Score (%)",

                height=400

            )


            # ====================================================
            # CANDIDATE DETAILS
            # ====================================================

            st.markdown("### Candidate Intelligence")
            st.markdown(
                '<div class="section-note">Inspect each candidate’s score components, strengths, gaps, and explainable insight.</div>',
                unsafe_allow_html=True
            )


            for rank, result in enumerate(
                results,
                start=1
            ):

                with st.expander(
                    f"#{rank} — {result['name']}"
                ):

                    # --------------------------------------------
                    # Candidate Name
                    # --------------------------------------------

                    st.markdown(
                        f"### 👤 {result['name']}"
                    )

                    st.caption(
                        f"Resume file: "
                        f"{result['filename']}"
                    )


                    # --------------------------------------------
                    # Final Score
                    # --------------------------------------------

                    st.metric(

                        "Final Match Score",

                        f"{float(result['final']):.2f}%"

                    )


                    # --------------------------------------------
                    # Individual Scores
                    # --------------------------------------------

                    col1, col2, col3, col4 = st.columns(
                        4
                    )


                    col1.metric(
                        "Semantic Match",
                        f"{float(result['semantic']):.2f}%"
                    )


                    col2.metric(
                        "Skills",
                        f"{float(result['skills']):.2f}%"
                    )


                    col3.metric(
                        "Experience",
                        f"{float(result['experience']):.2f}%"
                    )


                    col4.metric(
                        "Education",
                        f"{float(result['education']):.2f}%"
                    )


                    st.divider()


                    # --------------------------------------------
                    # Job Skills
                    # --------------------------------------------

                    st.markdown(
                        "### 🎯 Job Skills"
                    )


                    if result["job_skills"]:
                        pills = "".join(
                            f'<span class="skill-pill">{skill.title()}</span>'
                            for skill in result["job_skills"]
                        )
                        st.markdown(pills, unsafe_allow_html=True)

                    else:

                        st.info(
                            "No skills detected."
                        )


                    # --------------------------------------------
                    # Matched Skills
                    # --------------------------------------------

                    st.markdown(
                        "### ✅ Matched Skills"
                    )


                    if result["matched_skills"]:
                        pills = "".join(
                            f'<span class="match-pill">✓ {skill.title()}</span>'
                            for skill in result["matched_skills"]
                        )
                        st.markdown(pills, unsafe_allow_html=True)

                    else:

                        st.info(
                            "No matching skills found."
                        )


                    # --------------------------------------------
                    # Missing Skills
                    # --------------------------------------------

                    st.markdown(
                        "### ❌ Missing Skills"
                    )


                    if result["missing_skills"]:
                        pills = "".join(
                            f'<span class="missing-pill">✕ {skill.title()}</span>'
                            for skill in result["missing_skills"]
                        )
                        st.markdown(pills, unsafe_allow_html=True)

                    else:

                        st.success(
                            "No detected skills are missing."
                        )


                    # --------------------------------------------
                    # Explainable Insight
                    # --------------------------------------------

                    st.markdown(
                        "### 💡 Explainable Insight"
                    )


                    final_score = float(
                        result["final"]
                    )


                    if final_score >= 75:

                        st.success(

                            "Strong candidate match. "
                            "The resume has a high overall "
                            "compatibility with the job requirements."

                        )

                    elif final_score >= 50:

                        st.warning(

                            "Moderate candidate match. "
                            "The candidate matches several "
                            "job requirements but may have "
                            "some skill gaps."

                        )

                    else:

                        st.error(

                            "Low candidate match. "
                            "The resume has significant differences "
                            "from the job requirements."

                        )


            # ====================================================
            # FOOTER
            # ====================================================

            st.divider()

            st.markdown("""
    <div class="footer">
        TalentScreen AI · Transformer Embeddings · Cosine Similarity ·
        Dynamic Skill Matching · Experience & Education Analysis ·
        Local Processing
    </div>
    """, unsafe_allow_html=True)