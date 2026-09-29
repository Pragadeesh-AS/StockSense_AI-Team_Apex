import docx

doc = docx.Document()

# Title
doc.add_heading('MACHINE LEARNING & FEATURE ENGINEERING REPORT', 0)
doc.add_paragraph('Team Name: STOCKSENSE_Team Apex').bold = True
doc.add_paragraph('Phase: Round 2').bold = True

# Feature Engineering
doc.add_heading('1. Feature Engineering Strategy', level=1)
doc.add_paragraph('To capture the underlying patterns in retail demand while strictly avoiding data leakage, we engineered several groups of features utilizing past data only:')
doc.add_paragraph('• Time Features: day_of_week, month, week_no (to capture seasonality and weekend shopping behavior).', style='List Bullet')
doc.add_paragraph('• Lag & Rolling Features: lag_1_demand, lag_7_demand, rolling_mean_7, rolling_std_7 (to capture short-term momentum and sales volatility).', style='List Bullet')
doc.add_paragraph('• Inventory Features: days_of_inventory, reorder_gap (to measure current stock health against historical demand).', style='List Bullet')
doc.add_paragraph('Data leakage was explicitly prevented by calculating the target variables (next_7_day_demand) prior to shifting the historical features using pandas transform logic.')

# Train / Test Split
doc.add_heading('2. Time-Aware Data Split', level=1)
doc.add_paragraph('In adherence to the competition rules, we completely avoided random shuffling. We utilized an 80/20 chronological split based on the sorted dates. The models were trained on the first 80% of the timeline and evaluated on the strictly unseen future 20% to accurately simulate real-world forecasting.')

# Model 1
doc.add_heading('3. Model 1: 7-Day Demand Forecast', level=1)
doc.add_paragraph('• Target Variable: next_7_day_demand (Regression)')
doc.add_paragraph('• Algorithm Chosen: XGBoost Regressor')
doc.add_paragraph('• Justification: Retail sales data contains highly complex, non-linear interactions (e.g., weekends + promotions + holidays). XGBoost optimally captures these interactions while maintaining high computational efficiency.')
doc.add_paragraph('• Evaluation Metrics: Evaluated using MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), and R-Squared. RMSE was prioritized as it heavily penalizes massive forecasting errors, which are highly costly in supply chain environments.')

# Model 2
doc.add_heading('4. Model 2: Stock-Out Risk Classification', level=1)
doc.add_paragraph('• Target Variable: stockout_flag (Classification - triggered if future stock drops to critical levels)')
doc.add_paragraph('• Algorithm Chosen: Random Forest Classifier')
doc.add_paragraph('• Justification: Stock-outs represent an imbalanced classification problem. Random Forest allowed us to utilize the class_weight="balanced" parameter, heavily penalizing the model for missing critical stock-out events. It also provides robust protection against overfitting.')
doc.add_paragraph('• Evaluation Metrics: Evaluated via Classification Report (Precision, Recall, F1) and ROC-AUC. We optimized for high Recall, as in retail, a false alarm is far less costly than a missed stock-out (an empty shelf).')

# Explainability
doc.add_heading('5. Model Explainability', level=1)
doc.add_paragraph('To ensure management does not receive "black-box" alerts, we extracted Feature Importances from the Random Forest model. This allows the system to present actionable, transparent explanations (e.g., "Alert triggered due to severe 7-day sales momentum combined with a low days-of-inventory ratio") alongside the reorder recommendations for Round 3.')

doc.save(r'd:\college\DS hackathon\STOCKSENSE_Team Apex\reports\Round2_ML_Report.docx')
print("Successfully generated Round 2 DOCX report.")
