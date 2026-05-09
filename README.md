# 4thYearProject

🔵 Smart Warehouse Demand Prediction — LPG Cylinders
Mitra Bharatgas Agency · Murshidabad, West Bengal
Setup & Run

Install dependencies:
pip install -r requirements.txt
Place your Excel file in the same folder:
Warehouse_Demand_5000_1.xlsx
Run the app:
streamlit run app.py

App Pages
PageWhat it does📊 Overview & DataDataset preview, KPIs, download🤖 Train ModelsTrains 4 regression + 4 classification models📈 Visualizations8 charts — model performance, trends, zones🔮 Predict DemandSingle-scenario gas consumption + stockout risk📋 Batch ReportAll zones × months → heatmaps + Excel export
ML Targets

Regression: Gas Consumption (kg) per household — R²=0.97
Classification: Zone Stockout Risk (0/1) — F1=1.00

Best Models (on 80/20 split)

Regression: Random Forest
Classification: Random Forest
