import streamlit as st
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #18181b;
    --ink-soft: #52525b;
    --muted: #71717a;
    --border: #e4e4e7;
    --accent: #7c3aed;
    --accent-soft: #f5f3ff;
    --surface: #ffffff;
    --bg: #fafafa;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

html, body {
    background: var(--bg);
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.stApp {
    background: var(--bg) !important;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 620px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* Hero header */
.hero {
    text-align: center;
    padding-bottom: 26px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 30px;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--accent);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: var(--ink);
    margin: 0;
    letter-spacing: -0.6px;
}
.hero-subtitle {
    color: var(--muted);
    font-size: 14.5px;
    margin-top: 8px;
    line-height: 1.5;
    max-width: 440px;
    margin-left: auto;
    margin-right: auto;
}

/* Card container */
.card {
    background-color: var(--surface);
    padding: 26px 28px;
    border-radius: 14px;
    border: 1px solid var(--border);
    box-shadow: 0 1px 2px rgba(24, 24, 27, 0.03);
    margin-bottom: 24px;
}

/* Text area label */
div[data-testid="stTextArea"] label {
    font-weight: 600 !important;
    color: var(--ink-soft) !important;
    font-size: 13.5px !important;
}

div[data-testid="stTextArea"] textarea {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* Predict button */
div.stButton > button {
    background: var(--accent);
    color: white;
    font-weight: 600;
    font-size: 15px;
    padding: 12px 0;
    border-radius: 10px;
    border: none;
    letter-spacing: 0.2px;
    box-shadow: none;
    transition: background 0.15s ease;
}
div.stButton > button:hover {
    background: #6d28d9;
    color: white;
}
div.stButton > button:active {
    background: #5b21b6;
    color: white;
}

/* Result card */
.result-card {
    background: var(--accent-soft);
    border: 1px solid #ddd6fe;
    padding: 26px;
    border-radius: 14px;
    text-align: center;
    margin-top: 4px;
}
.result-label {
    font-size: 11.5px;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}
.result-genre {
    font-size: 26px;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.3px;
}

/* Warning tweak */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
}

/* Section header */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin: 0 0 14px 2px;
}
.section-header h3 {
    font-size: 15.5px;
    font-weight: 700;
    color: var(--ink);
    margin: 0;
}

/* About row */
.about-row {
    display: flex;
    gap: 28px;
    margin-top: 14px;
    flex-wrap: wrap;
}
.about-item .value {
    color: var(--ink);
    font-size: 16px;
    font-weight: 700;
}
.about-item .label {
    color: var(--muted);
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
    margin-top: 1px;
}

hr {
    border-color: var(--border) !important;
    margin: 2rem 0 !important;
}

.footer {
    text-align: center;
    color: var(--muted);
    margin-top: 40px;
    font-size: 12.5px;
    padding-top: 18px;
    border-top: 1px solid var(--border);
}
.footer b {
    color: var(--ink-soft);
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("movie_genre_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">🎬 AI-powered classification</div>
        <div class="hero-title">Movie Genre Classifier</div>
        <div class="hero-subtitle">
            Enter a movie plot summary and a trained machine learning model
            will predict its genre.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# USER INPUT
# ============================================================

st.markdown(
    '<div class="section-header"><h3>Plot summary</h3></div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

plot = st.text_area(
    "Enter movie plot summary",
    placeholder=(
        "Example: A detective investigates a mysterious murder "
        "and discovers several hidden clues..."
    ),
    height=180,
    label_visibility="collapsed"
)

predict_clicked = st.button("Predict genre", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    if plot.strip() == "":
        st.warning("Please enter a movie plot summary.")

    else:

        # Convert text into TF-IDF
        plot_tfidf = vectorizer.transform([plot])

        # Predict genre
        prediction = model.predict(plot_tfidf)

        predicted_genre = prediction[0]

        # Display result
        st.markdown(
            '<div class="section-header"><h3>Result</h3></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted genre</div>
                <div class="result-genre">{predicted_genre.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# INFORMATION
# ============================================================

st.markdown(
    '<div class="section-header" style="margin-top: 36px;"><h3>About the model</h3></div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

st.write(
    "This application uses **TF-IDF** for text feature extraction and "
    "**Logistic Regression** for movie genre classification."
)

st.markdown(
    """
    <div class="about-row">
        <div class="about-item">
            <div class="value">27</div>
            <div class="label">Genres</div>
        </div>
        <div class="about-item">
            <div class="value">58.05%</div>
            <div class="label">Test accuracy</div>
        </div>
        <div class="about-item">
            <div class="value">TF-IDF + LogReg</div>
            <div class="label">Pipeline</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with <b>Python</b> • <b>Scikit-learn</b> • <b>TF-IDF</b> • <b>Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True
)