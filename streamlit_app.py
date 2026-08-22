import json, re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Phishing Email Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "logistic_regression_phishing.joblib")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    config = joblib.load(MODEL_DIR / "preprocessing_config.joblib")
    return model, vectorizer, config

@st.cache_data
def load_dashboard():
    with open(MODEL_DIR / "dashboard_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

model, vectorizer, config = load_artifacts()
dashboard = load_dashboard()

# Same text-cleaning logic used in the python analysis
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    lemmatizer = WordNetLemmatizer()
except Exception:
    # Saved stop words keep the app functional if NLTK data is unavailable.
    stop_words = set(config["stop_words"])
    lemmatizer = None

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower()
    words = [word for word in text.split() if word not in stop_words]
    if lemmatizer is not None:
        try:
            words = [lemmatizer.lemmatize(word) for word in words]
        except LookupError:
            pass
    return " ".join(words)

def predict_email(text):
    processed = preprocess_text(text)
    vector = vectorizer.transform([processed])
    prediction = int(model.predict(vector)[0])
    probability = float(model.predict_proba(vector)[0, 1])
    label = "Phishing" if prediction == 1 else "Legitimate"
    confidence = probability if prediction == 1 else 1 - probability
    return label, probability, confidence, processed

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.metric-card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: 18px;
    background: rgba(128,128,128,.05);
}
.small-note {color: #6b7280; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.title("🛡️ Phishing Detector")
st.sidebar.caption("AI-based phishing email detection")
page = st.sidebar.radio("Navigation", ["Dashboard", "Prediction"])
st.sidebar.divider()
st.sidebar.info(
    "Model: Logistic Regression\n\n"
    "TF-IDF features: 10,000\n\n"
    "Optimised C: 2.0"
)

# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------
if page == "Dashboard":
    st.title("📊 Phishing Detection Dashboard")
    st.caption("Summary of the completed practical experiment and final deployed model.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleaned Emails", f"{dashboard['dataset_records']:,}")
    c2.metric("Phishing Emails", f"{dashboard['class_counts']['Phishing']:,}")
    c3.metric("Legitimate Emails", f"{dashboard['class_counts']['Legitimate']:,}")
    c4.metric("TF-IDF Features", f"{dashboard['tfidf_features']:,}")

    st.subheader("Dataset Overview")
    left, right = st.columns(2)

    with left:
        class_df = pd.DataFrame({
            "Class": list(dashboard["class_counts"].keys()),
            "Count": list(dashboard["class_counts"].values())
        })
        fig = px.pie(
            class_df, names="Class", values="Count", hole=.55,
            title="Class Distribution"
        )
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        overview = pd.DataFrame({
            "Measure": ["Rows before cleaning", "Duplicates removed", "Rows after cleaning"],
            "Value": [
                dashboard["rows_before_cleaning"],
                dashboard["duplicates_removed"],
                dashboard["dataset_records"]
            ]
        })
        fig = px.bar(
            overview, x="Measure", y="Value",
            title="Data Cleaning Summary", text="Value"
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Email Structure")
    feature_df = pd.DataFrame([
        {"Class":"Legitimate","Feature":"Average email length","Value":dashboard["mean_features"]["Legitimate"]["email_length"]},
        {"Class":"Phishing","Feature":"Average email length","Value":dashboard["mean_features"]["Phishing"]["email_length"]},
        {"Class":"Legitimate","Feature":"Average word count","Value":dashboard["mean_features"]["Legitimate"]["word_count"]},
        {"Class":"Phishing","Feature":"Average word count","Value":dashboard["mean_features"]["Phishing"]["word_count"]},
    ])
    fig = px.bar(
        feature_df, x="Feature", y="Value", color="Class",
        barmode="group", title="Average Structural Features"
    )
    fig.update_layout(height=390, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Comparison")
    results = pd.DataFrame(dashboard["metrics"])
    metric_choice = st.selectbox(
        "Select metric",
        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        index=3
    )
    fig = px.bar(
        results.sort_values(metric_choice, ascending=True),
        x=metric_choice, y="Model", orientation="h",
        text=metric_choice, title=f"Model Comparison — {metric_choice}"
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_xaxes(range=[0.88, 1.01])
    fig.update_layout(height=400, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Final Model Performance")
    best = results.iloc[0]
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Accuracy", f"{best['Accuracy']:.2%}")
    k2.metric("Precision", f"{best['Precision']:.2%}")
    k3.metric("Recall", f"{best['Recall']:.2%}")
    k4.metric("F1-score", f"{best['F1-Score']:.2%}")
    k5.metric("ROC-AUC", f"{best['ROC-AUC']:.4f}")

    cm = dashboard["confusion_matrix"]
    cm_df = pd.DataFrame(
        [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
        index=["Actual Legitimate", "Actual Phishing"],
        columns=["Predicted Legitimate", "Predicted Phishing"]
    )
    fig = px.imshow(
        cm_df, text_auto=True, aspect="auto",
        title="Final Model Confusion Matrix"
    )
    fig.update_layout(height=400, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "Final deployed model: Logistic Regression (C = 2.0). "
        "It was selected in the practical using the highest test F1-score."
    )
    st.caption(
        "Research Dashboard"
    )

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
else:
    st.title("🔍 Phishing Email Prediction")
    st.caption("Paste an email below and classify it using the final Logistic Regression model.")

    example = st.selectbox(
        "Optional example",
        ["Choose an example", "Long phishing email", "Long legitimate email"]
    )

    phishing_example = """Subject: URGENT: Your Account Has Been Temporarily Suspended

Dear Valued Customer,

We are writing to inform you that we have detected unusual activity associated with your account. For your protection, our security system has placed a temporary restriction on your account until your identity can be verified.

Our records indicate that a recent login attempt was made from an unrecognised device and location. If this activity was not authorised by you, you must verify your account immediately to prevent permanent suspension and possible loss of access.

Please complete the security verification process within the next 24 hours by clicking the secure verification link below:

https://secure-account-verification.example.com/verify

During the verification process, you may be asked to confirm your full name, account number, email address, password and other security information. This information is required to confirm that you are the legitimate account holder.

Failure to complete the verification within the specified time may result in permanent suspension of your account and loss of access to important account information.

Thank you for your immediate cooperation.

Security and Account Protection Department"""

    legitimate_example = """Subject: Meeting Agenda and Schedule for Next Week

Dear Team,

I hope everyone is doing well.

I am writing to confirm the arrangements for our project meeting next week. The meeting has been scheduled for Tuesday at 3:00 PM and will take place in Meeting Room B.

The purpose of the meeting is to review the progress made during the current development phase and discuss the activities that need to be completed before the next project milestone.

The following items will be discussed during the meeting:

1. Review of completed project activities
2. Current development progress
3. Discussion of outstanding tasks
4. Review of testing results
5. Identification of technical issues
6. Allocation of tasks for the coming week
7. Confirmation of the next project milestone

Please review the latest project documentation before attending the meeting and bring a short update on your current work.

If you are unable to attend, please let me know in advance so that alternative arrangements can be made.

Thank you for your continued work on the project.

Kind regards,
Project Coordinator
Research and Development Team"""

    default_text = ""
    if example == "Long phishing email":
        default_text = phishing_example
    elif example == "Long legitimate email":
        default_text = legitimate_example

    email = st.text_area(
        "Email content",
        value=default_text,
        height=360,
        placeholder="Paste the complete email here..."
    )

    if st.button("🔍 Predict Email", type="primary", use_container_width=True):
        if not email.strip():
            st.warning("Please paste an email before running the prediction.")
        else:
            label, probability, confidence, processed = predict_email(email)

            st.divider()
            if label == "Phishing":
                st.error("🚨 PHISHING EMAIL")
            else:
                st.success("✅ LEGITIMATE EMAIL")

            a,b,c = st.columns(3)
            a.metric("Prediction", label)
            b.metric("Phishing Probability", f"{probability:.2%}")
            c.metric("Confidence", f"{confidence:.2%}")

            prob_df = pd.DataFrame({
                "Class": ["Legitimate", "Phishing"],
                "Probability": [1-probability, probability]
            })
            fig = px.bar(
                prob_df, x="Class", y="Probability",
                text="Probability", title="Prediction Probability"
            )
            fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            fig.update_yaxes(range=[0,1])
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View NLP processed text"):
                st.write(processed)

            st.caption("Deployed model: Logistic Regression (C = 2.0) with the practical's TF-IDF representation.")
