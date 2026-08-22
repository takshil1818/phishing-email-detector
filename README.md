# AI-Based Phishing Email Detection - Streamlit

Two-page Streamlit application 

## Pages
1. Dashboard: dataset overview, class distribution, email length and word-count visualisations, model comparison, final metrics and confusion matrix.
2. Prediction: paste an email and classify it with the final Logistic Regression model.

## Final model
The application selected **Logistic Regression (C=2.0)** using the highest test F1-score:
1. Accuracy: 97.60%
2. Precision: 97.58%
3. Recall: 97.76%
4. F1-score: 97.67%
5. ROC-AUC: 0.9966

The app trains this same final model configuration from `phishing_email.csv` when it starts.

## Run locally
```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `streamlit_app.py`, `requirements.txt`, and `phishing_email.csv`.
3. On Streamlit Community Cloud, choose **Create app**.
4. Select your repository, branch (`main`), and file (`streamlit_app.py`).
5. Deploy.


