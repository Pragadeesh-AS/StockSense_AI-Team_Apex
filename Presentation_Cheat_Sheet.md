# 🏆 NovaMart STOCKSENSE AI: Master Defense Guide

This is your complete cheat sheet for defending your project to the hackathon judges. It covers the entire workflow, how the code works, and the critical "Why" behind every model choice.

---

## 1. THE COMPLETE WORKFLOW
1. **Data Ingestion:** We started with a massive 5GB real-world retail dataset (Corporacion Favorita). Because standard laptops cannot process 5GB in memory, we filtered the data down to 2 specific stores to ensure the code ran smoothly without crashing (OOM - Out of Memory errors).
2. **Data Engineering (Round 1):** We mapped the raw data into the NovaMart schema (`inventory.csv`, `transactions.csv`, etc.), fixed injected data traps (like missing temperatures), and merged them into one `master_dataset.csv`.
3. **Feature Engineering (Round 2):** We created "Time-Series Features" (like trailing 7-day averages and previous day sales) to give the AI context about the past.
4. **Machine Learning:** We split the data by *Time*, trained two separate AI models, and exported their predictions to a CSV file.
5. **Dashboard Deployment (Round 3):** We built a Streamlit SaaS application that reads the AI's predictions and applies actual retail business rules to generate Manager Alerts.

---

## 2. HOW THE JUPYTER NOTEBOOK CODE WORKS
When the judges look at your Round 2 notebook, they will test your understanding of these specific code concepts:

* **Data Leakage Prevention:** 
  * *What it is:* If an AI knows tomorrow's sales while predicting tomorrow's sales, it is "cheating" (Data Leakage). 
  * *How we fixed it:* We used the Pandas `.shift(-7)` function. We forced the target variable (`next_7_day_demand`) to shift backwards, ensuring the model only used *past* data to predict the *future*.
* **Time-Aware Train/Test Split:**
  * *What it is:* In normal ML, you randomly scramble data (80% train, 20% test). 
  * *Why we DID NOT do that:* If you randomly scramble retail data, you are training the AI on December's data to predict January's data. That makes no sense. We split the data chronologically (first 80% of the timeline for training, last 20% for testing).
* **Target Variable Creation:**
  * We created a classification target (`is_stockout_risk`). Because inventory rarely hit exactly 0 mathematically, we defined a stock-out risk as: *If the future demand leaves us with less than 25 units of buffer stock, flag it as a 1 (Risk), otherwise 0 (Safe).*

---

## 3. MODEL SELECTION (THE "WHY")
*Remember: The hackathon banned Deep Learning (Neural Networks, LSTMs). We had to use Classical Machine Learning.*

### Model 1: Demand Forecasting
* **What we used:** `XGBoost Regressor`
* **Why we used it:** XGBoost builds sequential decision trees where each tree corrects the errors of the previous one. Retail sales are highly non-linear (sales suddenly spike because of a promotion or a weekend). XGBoost is the absolute best classical algorithm for handling sudden spikes, non-linear patterns, and missing values.
* **Alternative 1 (Linear Regression):** *Rejected.* Linear regression assumes a straight-line relationship. Retail sales do not go in a straight line; they bounce up and down based on holidays and weekends. Linear Regression would have failed miserably.
* **Alternative 2 (ARIMA/Prophet):** *Rejected.* Traditional time-series models like ARIMA are great for single lines of data, but they struggle to ingest "External Features" (like `promotion_flag`, `temperature`, `category`). XGBoost can ingest thousands of diverse features easily.

### Model 2: Stock-out Risk Classification
* **What we used:** `Random Forest Classifier`
* **Why we used it:** Random Forest builds hundreds of independent decision trees and makes them "vote" on the outcome. It is incredibly robust against "overfitting" (memorizing the data). Most importantly, Random Forest inherently provides **Feature Importance**, which allowed us to build the "Explainable AI (XAI)" tab in the dashboard so managers know *why* a risk was flagged.
* **Alternative 1 (Logistic Regression):** *Rejected.* It is too simple and cannot capture the complex, overlapping interactions between inventory levels and sudden promotion spikes.
* **Alternative 2 (Support Vector Machines - SVM):** *Rejected.* SVMs are mathematically beautiful but computationally highly expensive. Training an SVM on millions of retail rows would have taken hours, if not days, on a standard laptop. Random Forest is much faster.

---

## 4. THE DASHBOARD BUSINESS LOGIC
The judges want to know that you understand Business, not just Coding. 

Our AI models do **NOT** output the exact number of units to reorder. The AI strictly predicts the future demand and the risk probability. 
We wrote custom business logic in `app.py` to bridge the gap between AI and Management:

1. **Risk Tagging:** We take the `stockout_prob` from the Random Forest. If it's > 70%, we tag it `HIGH`. If > 40%, we tag it `MEDIUM`.
2. **Safety Stock:** We calculate safety stock as `rolling_7_day_average * 2`. A supermarket always needs a buffer.
3. **Reorder Quantity Formula:** 
   `Reorder = (AI Forecasted Demand + Safety Stock) - (Current Stock + Incoming Stock)`
   *If the result is negative, we change it to 0 (no order needed).*
4. **Explainability Logic:** We wrote a python function that looks at the features. If `promotion_flag == 1`, the dashboard tells the manager: *"Flagged due to active promotion."*

By splitting the AI Prediction from the Business Rules, we built a true, enterprise-grade decision support system.
