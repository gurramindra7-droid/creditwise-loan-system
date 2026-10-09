<div align="center">

# 💳 CreditWise — Loan Approval System

### ⚡ Predict Smarter. Analyze Risk. Make Data-Driven Decisions.

A Machine Learning-powered web application that predicts loan approval eligibility using applicant financial and demographic information.

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_LIVE_DEMO-CreditWise-00C853?style=for-the-badge)](https://creditwise-loan-system-skoc7zpht22cdcdjhsteol.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

<br/>

**🔴 Live Application:**  
### 🌐 [Open CreditWise](https://creditwise-loan-system-skoc7zpht22cdcdjhsteol.streamlit.app/)

</div>

---

## 🧠 About the Project

**CreditWise** is a machine learning-based loan approval prediction system designed to analyze applicant information and estimate whether a loan application is likely to be approved.

The application processes financial, demographic, employment, and loan-related attributes through a trained **Logistic Regression** model to generate predictions through an interactive web interface.

Built with Python and Streamlit, CreditWise demonstrates how machine learning can be integrated into a user-friendly application to support data-driven lending assessments.

> ⚠️ **Disclaimer:** CreditWise is an educational machine learning project. Its predictions are not financial advice, actual lending decisions, or a substitute for a lender's formal underwriting process.

---

## ✨ Key Features

| 🚀 Feature | ⚙️ Description |
|---|---|
| 🤖 ML-Powered Prediction | Uses Logistic Regression to estimate loan approval eligibility. |
| 📊 Financial Data Analysis | Processes applicant income, credit score, savings, debt-to-income ratio, and other attributes. |
| 🧹 Automated Preprocessing | Handles missing values and prepares numerical and categorical features. |
| 📐 Feature Scaling | Applies StandardScaler to numerical features. |
| 🔠 Categorical Encoding | Uses OneHotEncoder to transform categorical attributes into machine-readable features. |
| 🖥️ Interactive Web Interface | Provides an interactive Streamlit application for entering applicant information. |
| ⚡ Real-Time Inference | Generates a prediction when the user submits the application form. |
| ☁️ Cloud Deployment | Deployed using Streamlit Community Cloud. |

---

## 🛠️ Tech Stack

### 🐍 Programming Language
- Python

### 🤖 Machine Learning & Data Processing
- Scikit-learn
- Pandas
- NumPy
- Logistic Regression
- StandardScaler
- OneHotEncoder
- SimpleImputer

### 🌐 Frontend & Application Framework
- Streamlit

### 📦 Model Persistence
- Joblib
- Pickle-compatible model serialization through the saved model artifact

### ☁️ Deployment & Version Control
- Streamlit Community Cloud
- Git
- GitHub

---

## 🏗️ System Architecture

```text
             👤 USER
                |
                ▼
      ┌─────────────────────┐
      │  🖥️ Streamlit UI    │
      │ Applicant Input Form│
      └──────────┬──────────┘
                 |
                 ▼
      ┌─────────────────────┐
      │ 🧾 Input Validation  │
      │ & Data Preparation  │
      └──────────┬──────────┘
                 |
                 ▼
      ┌─────────────────────┐
      │ ⚙️ Preprocessing    │
      │                     │
      │ • Missing Values    │
      │ • Feature Scaling   │
      │ • One-Hot Encoding  │
      └──────────┬──────────┘
                 |
                 ▼
      ┌─────────────────────┐
      │ 🤖 Logistic         │
      │    Regression Model │
      └──────────┬──────────┘
                 |
                 ▼
      ┌─────────────────────┐
      │ 📊 Prediction       │
      │                     │
      │ Loan Eligibility    │
      └─────────────────────┘
