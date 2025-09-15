# 🛒 Walmart Retail Sales & Demand Forecasting

[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-brightgreen?logo=streamlit)](https://share.streamlit.io)  
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)  
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)](https://scikit-learn.org/stable/)  
[![RandomForest](https://img.shields.io/badge/Model-Random%20Forest-yellowgreen)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

> **Accurately forecast Walmart’s weekly sales across stores & departments using feature-engineered machine learning models.**  
> This project is built as an **end-to-end ML pipeline + Streamlit app** to demonstrate practical demand forecasting.

---

## 📌 Project Overview

Walmart operates thousands of stores across the US. Accurate demand forecasting is key to:

- 📦 Optimize **inventory** (avoid stockouts & overstocking)  
- 👩‍💼 Manage **staffing & operations** efficiently  
- 💰 Improve **financial planning**  
- 📊 Tailor **marketing & promotions** around holidays  

This project predicts **weekly sales** using **historical data** enriched with feature engineering (lags, rolling averages, holiday indicators).  
The best-performing model (**Random Forest Regressor**) achieved **R² ~ 0.99** with very low forecasting error.

---

## 🚀 Demo (Streamlit App)

👉 [Live Demo on Streamlit Cloud](https://share.streamlit.io) *(link to your deployment once you push)*

### App Features
✅ Enter **store, department, date, holiday flag, markdowns, CPI, unemployment** → get predicted sales  
✅ Visualize **historical sales trends vs forecast**  
✅ Provides **uncertainty interval** for predictions  
✅ Works with or without historical dataset uploaded  

---

## 📊 Key Results

| Model                       | MAE       | MSE          | R²     |  
|-----------------------------|-----------|--------------|--------|  
| 🌟 Random Forest Regressor  | **835**  | 2.84M        | **0.99** |  
| XGBoost Regressor           | 841       | 2.76M        | 0.99   |  
| LightGBM Regressor          | 848       | 2.80M        | 0.99   |  
| Linear Regression           | 3605      | 29.4M        | 0.87   |  
| Prophet                     | 670k      | 9.64e11      | 0.52   |  
| SARIMA                      | 1.02M     | 2.04e12      | -0.02  |  
| ARIMA                       | 1.09M     | 2.42e12      | -0.20  |  

✅ **Random Forest Regressor was selected as the final model.**

---

## 📂 Repository Structure

walmart-forecasting/

├── app.py # Streamlit app

├── requirements.txt # dependencies

├── README.md # project documentation

├── .gitignore

├── data/

│ └── walmart_sales.csv # dataset 

├── rf_pipeline.pkl

├── feature_stats.json

└── meta.json

---

## ⚙️ Tech Stack

- **Python 3.9+**  
- **Streamlit** → interactive web app  
- **Scikit-learn** → Random Forest model & pipeline  
- **Pandas / NumPy** → data wrangling & feature engineering  
- **Matplotlib / Seaborn** → EDA & visualizations  
- **Joblib** → save & load trained models  

---

## 🛠️ Setup & Usage

🔹 1. Clone repo

git clone https://github.com/<your-username>/walmart-forecasting.git
cd walmart-forecasting

🔹 2. Add dataset

Place your cleaned dataset at:
data/walmart_sales.csv
Must include: store, date, temperature, fuel_price, markdown1..markdown5, cpi, unemployment, dept, weekly_sales, IsHoliday, type, size

🔹 3. Install dependencies

pip install -r requirements.txt

🔹 4. Train & save pipeline

python src/train_save_model.py

🔹 5. Run Streamlit app

streamlit run app.py

App will open in browser → http://localhost:8501

🌐 Deployment (Streamlit Cloud)

Push repo to GitHub

Go to Streamlit Cloud

Click New App → Select repo & app.py → Deploy 🚀

🔧 Future Enhancements

* Hyperparameter tuning (Optuna/RandomSearch)

* Time-series cross-validation

* Deep learning models (LSTM, GRU, TFT)

* Cloud deployment with CI/CD & monitoring


👩‍💻 Author

Your Name — Shyamily Haridas

📌 Streamlit - 

💻 GitHub - 
