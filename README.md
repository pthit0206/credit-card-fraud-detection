## 📌 Project Overview
This project detects fraudulent credit card transactions using machine learning on anonymized European cardholder data (Sept 2013).  
The dataset is highly imbalanced: 492 frauds out of 284,807 transactions (0.172%).

## 📂 Dataset
- Source: Kaggle – Credit Card Fraud Detection
- Features: V1–V28 are PCA-transformed components.
- Non-PCA features: `Time`, `Amount`
- Target: `Class` (1 = Fraud, 0 = Genuine)

## 🎯 Problem Statement
Build a model to identify fraud transactions reliably despite extreme class imbalance.

## 🛠️ Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- ML models: (example: Logistic Regression, Random Forest, Isolation Forest, XGBoost, etc.)

## ✅ Steps Performed
1. Data loading & inspection
2. Exploratory Data Analysis (EDA)
3. Handling imbalance (undersampling/oversampling/SMOTE/etc.)
4. Feature scaling
5. Model training
6. Model evaluation using:
   - Precision-Recall AUC (AUPRC)
   - Confusion Matrix
   - Precision, Recall, F1 score
7. Final model selection
