# AI-Based Phishing Email Detector

Two-page Streamlit application based on the completed practical.

## Pages
1. Dashboard - practical results, dataset summary, model comparison, final metrics and confusion matrix.
2. Prediction - classifies a pasted email using the final Logistic Regression model.

## Final model
Logistic Regression with C=2.0, selected in the practical using the highest test F1-score.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Streamlit Community Cloud
Push the project to GitHub and create an app using `streamlit_app.py` as the main file.

