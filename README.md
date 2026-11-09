# End-to-End Classification Deployment: Breast Cancer Diagnostics

Divyapratap Singh Chauhan
2026AK05625 

**a. Problem Statement**
The goal of this assignment is to design, evaluate, and deploy an end-to-end machine learning pipeline. The resulting interactive web application allows users to upload unseen test data and evaluate five different classification algorithms dynamically without model retraining.

**b. Dataset Description**
* **Name:** Breast Cancer Wisconsin (Diagnostic)
* **Source:** Scikit-learn (originally UCI Machine Learning Repository)
* **Features:** 30 continuous numerical features computed from digitized images of fine needle aspirates (FNA) of breast masses.
* **Instances:** 569 
* **Target:** Binary classification (Malignant vs. Benign).

**c. Github Repository Link**

https://github.com/Divyapratap-Singh-Chauhan/ML-Assignment-2

**d. Models Used & Comparison Table**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9443 |
| **Decision Tree** | 0.9474 | 0.9405 | 0.9577 | 0.9577 | 0.9577 | 0.8872 |
| **KNN** | 0.9474 | 0.9856 | 0.9577 | 0.9577 | 0.9577 | 0.8872 |
| **Naive Bayes** | 0.9649 | 0.9984 | 0.9589 | 0.9859 | 0.9722 | 0.9248 |
| **Support Vector Machine** | 0.9825 | 0.9974 | 0.9726 | 1.0000 | 0.9861 | 0.9619 |

**Observations on Model Performance**

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Showed high stability and excellent precision, proving that a linear decision boundary is highly effective for this dataset after scaling. |
| **Decision Tree** | Exhibited the lowest AUC, reflecting a slight tendency to overfit the training data due to the lack of pruning or ensemble methods. |
| **KNN** | Performed strongly, indicating that tumors of the same diagnostic class cluster closely together in the 30-dimensional feature space. |
| **Naive Bayes** | Achieved a near-perfect AUC score, demonstrating that the assumption of feature independence holds up well practically for this medical data. |
| **Support Vector Machine** | **Overall, Winner for your dataset? Yes.** It yielded the highest Accuracy (0.98) and a perfect Recall (1.00) for the benign class. Its ability to handle non-linear boundaries via the RBF kernel made it the most robust classifier. |
