import streamlit as st
import joblib
from preprocess import preprocess

st.set_page_config(page_title="AG News Classifier", page_icon="📰", layout="centered")

# ---------- Custom styling ----------
st.markdown("""
<style>
.main { background-color: #f7f9fc; }
h1 { color: #1a3c6e; font-weight: 800; }
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    height: 2.8em;
}
div[data-testid="stTextArea"] textarea {
    border-radius: 10px;
    border: 1px solid #c8d4e3;
}
.result-card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1.2em 1.5em;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-top: 1em;
}
.caption-box {
    background-color: #eef3fb;
    border-radius: 8px;
    padding: 0.6em 1em;
    font-size: 0.9em;
    color: #3a4a63;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    label_names = joblib.load("label_names.joblib")
    metrics = joblib.load("test_metrics.joblib")
    return model, label_names, metrics

model, label_names, metrics = load_artifacts()

# ---------- Header ----------
st.title("📰 AG News Topic Classifier")
st.write("Enter a news headline or short article, and the model will predict its topic category.")
st.markdown(
    f"<div class='caption-box'>🔧 Model: Linear SVM + TF-IDF (calibrated) &nbsp;|&nbsp; "
    f"🎯 Test accuracy: {metrics['accuracy']:.1%} &nbsp;|&nbsp; Macro-F1: {metrics['macro_f1']:.1%}</div>",
    unsafe_allow_html=True
)
st.write("")

# ---------- Input ----------
if "news_text" not in st.session_state:
    st.session_state.news_text = ""

def clear_text():
    st.session_state.news_text = ""

user_input = st.text_area(
    "News text:",
    height=150,
    key="news_text",
    placeholder="e.g. Apple unveils new chip for its laptops..."
)

col1, col2 = st.columns([1, 1])
with col1:
    classify_clicked = st.button("🔍 Classify", use_container_width=True)
with col2:
    st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)

# ---------- Prediction ----------
if classify_clicked:
    text = st.session_state.news_text
    if not text or not text.strip():
        st.warning("Please enter some text before classifying.")
    elif len(text.strip().split()) < 3:
        st.warning("Please enter a longer piece of text (at least a few words) for a reliable prediction.")
    else:
        cleaned = preprocess(text, remove_stop=True, stem=False, remove_sources=False)
        pred_label = model.predict([cleaned])[0]
        pred_name = label_names[pred_label]
        probs = model.predict_proba([cleaned])[0]

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.success(f"Predicted category: **{pred_name}**")

        st.write("**Confidence by category:**")
        for label_id, name in label_names.items():
            st.progress(float(probs[label_id]), text=f"{name}: {probs[label_id]:.1%}")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("See cleaned text used by the model"):
            st.code(cleaned)