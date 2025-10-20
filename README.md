# 💳 Fraud Detection in Credit Card Transactions(CCFD)

## 📌 Description
This project aims to **detect fraudulent credit card transactions** using a combination of **unsupervised anomaly detection** and **supervised machine learning**.  
The workflow integrates models like **Isolation Forest** and **Local Outlier Factor (LOF)** for anomaly detection, followed by an **XGBoost classifier** for accurate fraud classification.  

A **Streamlit web app** enables real-time transaction input and prediction, allowing users to test transactions interactively and view model confidence visually.

---

## 🎯 Objectives
- Identify fraudulent transactions with high precision and minimal false positives.  
- Utilize both anomaly detection and supervised classification for robust predictions.  
- Address data imbalance through appropriate resampling and scaling.  
- Develop an **interactive web interface** for transaction testing and live prediction.  
- Evaluate model performance using **confusion matrices**, **ROC curves**, and **F1-scores**.  

---

## 🛠 Tools and Libraries
**Language:** Python  

**Frameworks & Libraries:**
- Scikit-learn  
- XGBoost  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Joblib  
- Streamlit (for deployment)  
- Jupyter Notebook (for analysis and model training)

---

## 📂 Dataset
**Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

**Details:**
- **Total transactions:** 284,807  
- **Fraud cases:** 492 (highly imbalanced)  
- **Features:** 30 total (V1–V28, `Time`, `Amount`, and `Class`)  

**Preprocessing Steps:**
- Managed imbalance using under/oversampling.  
- Scaled numerical features.  
- Engineered new features:
  - `balanceDiffOrig = oldbalanceOrg - newbalanceOrig`
  - `balanceDiffDest = newbalanceDest - oldbalanceDest`  
- Split dataset into training, validation, and test sets.

---

## 🏃 How to Run

### 1️⃣ Clone Repository
```bash
git clone git@github.com:Elevate-Labs-Intern/AIML--Fraud-Detection-in-Credit-Card-Transactions.git
cd AIML--Fraud-Detection-in-Credit-Card-Transactions
```
### 2️⃣ Select Kernel for the Jupyter notebook([Click here for guide](https://www.youtube.com/watch?v=-j6y-5t37os))
### 3️⃣ Add the "AIML Dataset.csv" downloaded from the earlier mentioned link to the project's root.
### 4️⃣ One by one, run the Jupyter notebook cells.
### 5️⃣ Run the Python file as: 
```bash
streamlit run fraud_detection.py
```
Then open the web-app at `localhost:8501`
