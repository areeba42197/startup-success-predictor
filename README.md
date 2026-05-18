# SmartStartup AI

**Explainable Startup Success Prediction and Business Analysis System**

---

## 📌 Overview

SmartStartup AI is a full-stack AI-powered web application that predicts whether a startup will succeed or fail — and explains *why* in plain business language. It combines:

| Component | Technology |
|-----------|-----------|
| ML Prediction | Logistic Regression, Random Forest, XGBoost |
| Explainable AI | SHAP (SHapley Additive Explanations) |
| Generative Analysis | Groq Chat Completions API |
| Simulation Mode | Interactive startup budget simulation |
| Currency Switcher | USD, PKR, EUR, GBP, AED, SAR, CAD |
| Pakistan Data | 21 real Pakistan startups added to dataset |

---

## 🛠️ Setup & Installation

```bash
# 1. Clone or download this project
cd smartstartup_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📁 File Structure

```
smartstartup_ai/
├── app.py              # Main Streamlit application (entry point)
├── preprocessing.py    # Data cleaning, feature engineering
├── model.py            # ML model training and prediction
├── xai.py              # SHAP explainability layer
├── generative.py       # Groq AI integration (LLM insights)
├── gamification.py     # (integrated in app.py)
├── currency.py         # Currency conversion (USD ↔ PKR etc.)
├── pakistan_data.py    # 21 real Pakistan startup records
└── requirements.txt    # Python dependencies
```

---

## Pakistan Data

21 real Pakistani startups have been added to the Crunchbase dataset, including:
- Careem, Daraz, Bykea, Airlift, Safepay
- Jugnu, Bazaar Technologies, Zameen.com, Rozee.pk
- Oraan, MedznMore, Zindigi, and more

---

## 💱 Currency Support

Switch currency in the sidebar:
- **PKR** (Pakistani Rupee) — ₨
- **USD** (US Dollar) — $
- **EUR** (Euro) — €
- **GBP** (British Pound) — £
- **AED** (UAE Dirham) — د.إ
- **SAR** (Saudi Riyal) — ﷼

All financial figures update automatically. Results are specific to your chosen currency context.

---

## 🗃️ Dataset

Based on the [Crunchbase Startup Dataset](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase) (66,369 records), augmented with 21 Pakistan startup records per course requirement.

---

## 📋 System Architecture

```
User Input / Gamified Scenario
        ↓
Data Preprocessing (preprocessing.py)
        ↓
Machine Learning Model (model.py)
        ↓
Explainable AI Layer — SHAP (xai.py)
        ↓
Generative AI Layer — Groq LLM (generative.py)
        ↓
Simulation Decision Engine (app.py)
        ↓
Streamlit Web Interface (app.py)
```

---

## 👥 Team

SmartStartup AI — AI Project, Semester Project
