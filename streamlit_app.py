
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

st.set_page_config(
    page_title="AI Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
.main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;}
.subtitle {color: #667085; margin-bottom: 1.2rem;}
.metric-card {padding: 1rem; border-radius: 12px; border: 1px solid #e5e7eb;}
.prediction-box {padding: 1.2rem; border-radius: 14px; text-align: center;}
.small-note {color: #667085; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

RANDOM_STATE = 92
TEST_SIZE = 0.20
DATA_PATH = Path(__file__).parent / "phishing_email.csv"

# Notebook's final tuned model:
# Logistic Regression, C=2.0, selected using test F1-score.
NOTEBOOK_RESULTS = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Naive Bayes", "Decision Tree"],
    "Accuracy": [0.9760, 0.9650, 0.9465, 0.9195],
    "Precision": [0.9758, 0.9687, 0.9832, 0.9173],
    "Recall": [0.9776, 0.9631, 0.9116, 0.9271],
    "F1-Score": [0.9767, 0.9659, 0.9460, 0.9222],
    "ROC-AUC": [0.9966, 0.9940, 0.9939, 0.9209],
})

@st.cache_resource
def load_nlp():
    for resource, package in [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass
    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        sw = set(ENGLISH_STOP_WORDS)
    try:
        lem = WordNetLemmatizer()
    except Exception:
        lem = None
    return sw, lem

def preprocess_text(text, sw, lem):
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower()
    words = [w for w in text.split() if w not in sw]
    if lem is not None:
        try:
            words = [lem.lemmatize(w) for w in words]
        except LookupError:
            pass
    return " ".join(words)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("phishing_email.csv is missing from the project folder.")
    df = pd.read_csv(DATA_PATH)
    required = {"text_combined", "label"}
    if not required.issubset(df.columns):
        raise ValueError("The CSV must contain 'text_combined' and 'label' columns.")
    df = df.dropna(subset=["text_combined", "label"]).drop_duplicates().reset_index(drop=True)
    df["label"] = pd.to_numeric(df["label"]).astype(int)
    df["email_length"] = df["text_combined"].astype(str).str.len()
    df["word_count"] = df["text_combined"].astype(str).str.split().str.len()
    return df

@st.cache_resource
def train_final_model():
    df = load_data()
    sw, lem = load_nlp()
    processed = df["text_combined"].apply(lambda x: preprocess_text(x, sw, lem))
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        processed, df["label"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=df["label"]
    )
    vectorizer = TfidfVectorizer(
        max_features=10000, min_df=2, max_df=0.95,
        ngram_range=(1, 2), sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(C=2.0, max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    actual_metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
    }
    return model, vectorizer, sw, lem, actual_metrics

df = load_data()
model, vectorizer, sw, lem, actual_metrics = train_final_model()

# ---------- Navigation ----------
st.sidebar.title("🛡️ Phishing Detector")
page = st.sidebar.radio("Navigate", ["Dashboard", "Prediction"])
st.sidebar.markdown("---")
st.sidebar.caption("AI-based phishing email detection using NLP + TF-IDF + Logistic Regression.")

if page == "Dashboard":
    st.markdown('<div class="main-title">AI Phishing Email Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Summary of the practical: dataset, NLP analysis, model comparison and final model performance.</div>', unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", f"{len(df):,}")
    c2.metric("Phishing emails", f"{int((df.label==1).sum()):,}")
    c3.metric("Legitimate emails", f"{int((df.label==0).sum()):,}")
    c4.metric("Selected model", "Logistic Regression")

    st.markdown("### Dataset Class Distribution")
    col1, col2 = st.columns(2)

    with col1:
        counts = df["label"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Legitimate", "Phishing"], [counts.get(0,0), counts.get(1,0)])
        ax.set_ylabel("Number of emails")
        ax.set_title("Email Classes")
        st.pyplot(fig, clear_figure=True)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["email_length"], bins=30)
        ax.set_xlabel("Characters")
        ax.set_ylabel("Frequency")
        ax.set_title("Email Length Distribution")
        st.pyplot(fig, clear_figure=True)

    st.markdown("### Structural Features")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["word_count"], bins=30)
        ax.set_xlabel("Words")
        ax.set_ylabel("Frequency")
        ax.set_title("Word Count Distribution")
        st.pyplot(fig, clear_figure=True)
    with col2:
        grouped = df.groupby("label")[["email_length", "word_count"]].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(2)
        width = 0.36
        ax.bar(x-width/2, grouped["email_length"].values, width, label="Email length")
        # scale word count so both structural features can be compared visually
        scale = max(grouped["email_length"].max(), 1) / max(grouped["word_count"].max(), 1)
        ax.bar(x+width/2, grouped["word_count"].values * scale, width, label="Word count (scaled)")
        ax.set_xticks(x, ["Legitimate", "Phishing"])
        ax.set_ylabel("Relative mean")
        ax.set_title("Average Structural Features")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    st.markdown("### Machine Learning Model Comparison")
    st.dataframe(
        NOTEBOOK_RESULTS.style.format({
            "Accuracy":"{:.2%}", "Precision":"{:.2%}",
            "Recall":"{:.2%}", "F1-Score":"{:.2%}", "ROC-AUC":"{:.4f}"
        }),
        use_container_width=True, hide_index=True
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(NOTEBOOK_RESULTS))
    width = 0.18
    for j, metric in enumerate(metric_cols):
        ax.bar(x + (j-1.5)*width, NOTEBOOK_RESULTS[metric], width, label=metric)
    ax.set_xticks(x, NOTEBOOK_RESULTS["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Optimised Model Performance")
    ax.legend(loc="lower right")
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Final Model")
    a,b,c,d,e = st.columns(5)
    best = NOTEBOOK_RESULTS.iloc[0]
    a.metric("Accuracy", f"{best['Accuracy']:.2%}")
    b.metric("Precision", f"{best['Precision']:.2%}")
    c.metric("Recall", f"{best['Recall']:.2%}")
    d.metric("F1-score", f"{best['F1-Score']:.2%}")
    e.metric("ROC-AUC", f"{best['ROC-AUC']:.4f}")

    st.info(
        "Logistic Regression with C=2.0 is selected because it achieved the highest "
        "test F1-score in the completed practical. F1-score was used to balance "
        "false positives and false negatives."
    )

    st.markdown("### Final Model Confusion Matrix")
    sw2, lem2 = load_nlp()
    processed = df["text_combined"].apply(lambda x: preprocess_text(x, sw2, lem2))
    _, X_test_text, _, y_test = train_test_split(
        processed, df["label"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=df["label"]
    )
    # Reproduce test set using same split; model/vectorizer are trained on same split.
    X_test_final = vectorizer.transform(X_test_text)
    pred_final = model.predict(X_test_final)
    cm = confusion_matrix(y_test, pred_final)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_xticks([0,1], ["Legitimate","Phishing"])
    ax.set_yticks([0,1], ["Legitimate","Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i,j], ha="center", va="center")
    st.pyplot(fig, clear_figure=True)

else:
    st.markdown('<div class="main-title">📧 Phishing Email Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Paste an email below and the selected Logistic Regression model will classify it.</div>', unsafe_allow_html=True)

    email_text = st.text_area(
        "Email content",
        height=360,
        placeholder="Paste the complete email here..."
    )

    if st.button("🔍 Predict Email", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Please enter an email before running the prediction.")
        else:
            processed = preprocess_text(email_text, sw, lem)
            vector = vectorizer.transform([processed])
            prediction = int(model.predict(vector)[0])
            phishing_probability = float(model.predict_proba(vector)[0, 1])
            confidence = phishing_probability if prediction == 1 else 1 - phishing_probability

            st.markdown("### Prediction Result")
            if prediction == 1:
                st.error(f"🚨 **PHISHING EMAIL** — {phishing_probability:.2%} phishing probability")
            else:
                st.success(f"✅ **LEGITIMATE EMAIL** — {(1-phishing_probability):.2%} legitimate probability")

            c1,c2,c3 = st.columns(3)
            c1.metric("Prediction", "Phishing" if prediction else "Legitimate")
            c2.metric("Phishing probability", f"{phishing_probability:.2%}")
            c3.metric("Confidence", f"{confidence:.2%}")

            st.progress(min(max(phishing_probability, 0.0), 1.0))
            with st.expander("View processed text"):
                st.write(processed)
