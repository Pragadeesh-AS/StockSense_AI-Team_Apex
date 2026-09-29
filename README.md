# 📦 NovaMart STOCKSENSE AI
**Enterprise Inventory Intelligence & Predictive Replenishment**

**Team:** Team Apex  
**Event:** IntelliData 2026 Data Science Hackathon  

---

## 🎯 Project Overview
NovaMart STOCKSENSE AI is an end-to-end data science and machine learning solution built to solve retail stock-outs and optimize inventory management. Instead of reactive tracking, STOCKSENSE uses predictive AI to forecast exact customer demand and flag high-risk stock-out scenarios *before* they happen.

## 🚀 Key Features & Hackathon Deliverables

**1. Round 1 (EDA & Data Quality)**
* Extracted and cleaned a massive multi-year retail dataset.
* Handled injected data traps (e.g., mean imputation for temperature, fixing negative inventory arithmetic).
* Generated automated, management-ready `.docx` data quality reports.

**2. Round 2 (Machine Learning)**
* **Demand Forecasting:** Trained an **XGBoost Regressor** to accurately predict a rolling 7-day future demand.
* **Stock-out Risk:** Trained a **Random Forest Classifier** to identify critical inventory shortages and calculate risk probabilities.
* Implemented time-aware splitting to prevent data leakage and extracted robust feature importances.

**3. Round 3 (Prototype Dashboard)**
* Developed a premium, dark-themed **Streamlit SaaS Dashboard** featuring:
  * 📊 **Executive Overview:** High-level KPIs, Inventory Value, and Portfolio Risk.
  * ⚡ **Action Center:** Automated AI replenishment recommendations with simulated approval workflows.
  * 📈 **Demand Intelligence:** Visual tracking of historical demand vs. XGBoost forecasts.
  * 🧠 **AI Control Room (XAI):** Feature importance visualizations proving the model isn't a "black box" and live model performance metrics.

---

## 📁 Repository Structure
```text
STOCKSENSE_Team Apex/
├── data/
│   ├── raw/                       # Original Kaggle datasets
│   └── processed/                 # model_predictions.csv and master datasets
├── notebooks/
│   ├── Round1_EDA.ipynb           # Exploratory Data Analysis & Cleaning
│   └── Round2_FeatureEng_ML.ipynb # Feature Engineering & Model Training
├── reports/
│   ├── Data_Quality_Report.docx   # Round 1 Submission
│   └── Round2_ML_Report.docx      # Round 2 Submission
├── dashboard/
│   └── app.py                     # Final Streamlit Application
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 🛠️ Installation & Usage

**1. Navigate to the project directory**
```bash
cd "STOCKSENSE_Team Apex"
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the AI Dashboard**
```bash
streamlit run dashboard/app.py
```

---
*Built with ❤️ by Team Apex for IntelliData 2026.*
