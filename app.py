import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef, confusion_matrix)

# UI Configuration
st.set_page_config(page_title="Breast Cancer Diagnostics Predictor", layout="wide")
st.title("Diagnostic ML Classification Evaluator")
st.write("Upload the unseen test dataset to evaluate model performance without retraining.")

# Sidebar Configuration
st.sidebar.header("Model Parameters")
test_file = st.sidebar.file_uploader("Upload CSV Test File", type=["csv"])

model_dict = {
    'Logistic Regression': 'model/logistic_regression_model.pkl',
    'Decision Tree': 'model/decision_tree_model.pkl',
    'K-Nearest Neighbors': 'model/knn_model.pkl',
    'Naive Bayes': 'model/naive_bayes_model.pkl',
    'Support Vector Machine': 'model/svm_model.pkl'
}

chosen_model = st.sidebar.selectbox("Choose a Classifier", list(model_dict.keys()))

if test_file is not None:
    df_test = pd.read_csv(test_file)
    st.subheader("Data Preview")
    st.dataframe(df_test.head())
    
    if 'target_label' not in df_test.columns:
        st.error("Missing 'target_label' column in the uploaded dataset.")
    else:
        # Split features and target
        X_test = df_test.drop(columns=['target_label'])
        y_true = df_test['target_label']
        
        # Load Model Pipeline
        try:
            pipeline = joblib.load(model_dict[chosen_model])
            
            # Generate Predictions
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1] 
            
            # Compute Metrics
            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "AUC": roc_auc_score(y_true, y_prob),
                "Precision": precision_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred),
                "F1 Score": f1_score(y_true, y_pred),
                "MCC": matthews_corrcoef(y_true, y_pred)
            }
            
            # Render Metrics
            st.subheader(f"Evaluation Results: {chosen_model}")
            cols = st.columns(6)
            for col, (metric_name, value) in zip(cols, metrics.items()):
                col.metric(metric_name, f"{value:.4f}")
            
            # Render Confusion Matrix
            st.subheader("Classification Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('Actual Class')
            st.pyplot(fig)
            
        except FileNotFoundError:
            st.error("Model artifact not found. Check the model_artifacts directory.")
else:
    st.info("Awaiting test data upload via the sidebar.")