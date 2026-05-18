"""
preprocessing.py — Data cleaning, feature engineering, and preparation.
Handles both the Crunchbase dataset and Pakistan startup augmentation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


CATEGORY_MAP = {
    "Transportation": "Transportation", "Ride-Hailing": "Transportation",
    "Logistics": "Logistics", "Last-Mile": "Logistics",
    "E-Commerce": "E-Commerce", "Marketplace": "E-Commerce",
    "FinTech": "FinTech", "Payments": "FinTech", "Digital Banking": "FinTech",
    "Digital Wallet": "FinTech", "Microfinance": "FinTech", "Lending": "FinTech",
    "Savings": "FinTech", "Women": "FinTech",
    "Real Estate": "Real Estate", "PropTech": "Real Estate",
    "HR": "HR & Recruitment", "Recruitment": "HR & Recruitment", "Jobs": "HR & Recruitment",
    "Music": "Media & Entertainment", "Entertainment": "Media & Entertainment",
    "B2B": "B2B Commerce", "Commerce": "B2B Commerce", "FMCG": "B2B Commerce",
    "Quick Commerce": "E-Commerce", "Grocery": "E-Commerce",
    "HealthTech": "HealthTech", "Pharmacy": "HealthTech",
    "EdTech": "EdTech", "Freelancing": "EdTech",
    "Food Delivery": "Food & Beverage",
    "Price Comparison": "E-Commerce",
    "Software": "Software", "web": "Software",
    "mobile": "Mobile", "games": "Gaming",
    "biotech": "Biotech", "cleantech": "CleanTech",
    "hardware": "Hardware", "enterprise": "Enterprise Software",
    "advertising": "Advertising", "analytics": "Analytics & Data",
    "social": "Social Media", "messaging": "Social Media",
    "health": "HealthTech", "medical": "HealthTech",
    "education": "EdTech", "travel": "Travel",
    "finance": "FinTech", "fintech": "FinTech",
    "ecommerce": "E-Commerce", "marketplace": "E-Commerce",
    "security": "Cybersecurity", "network": "Networking",
    "semiconductor": "Hardware", "cloud": "Cloud Computing",
    "saas": "Software", "paas": "Cloud Computing",
    "other": "Other",
}


def map_category(cat_str):
    if pd.isna(cat_str) or cat_str == "":
        return "Other"
    parts = str(cat_str).replace("|", ",").split(",")
    for part in parts:
        part = part.strip().lower()
        for key, val in CATEGORY_MAP.items():
            if key.lower() in part:
                return val
    return "Other"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Derive target
    if "status" in df.columns:
        df["target"] = df["status"].apply(
            lambda s: 1 if str(s).lower() in ["acquired", "ipo", "operating"] else 0
        )

    # Funding. The Crunchbase file stores unknown funding as "-", so use the
    # observed median instead of treating unknown funding as zero.
    df["funding_total_usd"] = pd.to_numeric(df.get("funding_total_usd", 0), errors="coerce")
    funding_median = df["funding_total_usd"].dropna().median()
    if pd.isna(funding_median):
        funding_median = 0
    df["funding_total_usd"] = df["funding_total_usd"].fillna(funding_median).clip(lower=0)
    df["funding_rounds"] = pd.to_numeric(df.get("funding_rounds", 1), errors="coerce").fillna(1).clip(lower=1)

    # Dates → startup age
    for col in ["founded_at", "first_funding_at", "last_funding_at"]:
        df[col] = pd.to_datetime(df.get(col), errors="coerce")

    now = pd.Timestamp("2024-01-01")
    df["founding_year"] = df["founded_at"].dt.year
    founding_year_median = df["founding_year"].dropna().median()
    if pd.isna(founding_year_median):
        founding_year_median = 2014
    df["founding_year"] = df["founding_year"].fillna(founding_year_median).clip(lower=1970, upper=2024)

    df["startup_age_years"] = (now - df["founded_at"]).dt.days / 365.25
    df["startup_age_years"] = df["startup_age_years"].fillna(df["startup_age_years"].median()).clip(lower=0)

    df["days_first_to_last_funding"] = (df["last_funding_at"] - df["first_funding_at"]).dt.days
    df["days_first_to_last_funding"] = df["days_first_to_last_funding"].fillna(0).clip(lower=0)

    df["days_to_first_funding"] = (df["first_funding_at"] - df["founded_at"]).dt.days
    df["days_to_first_funding"] = df["days_to_first_funding"].fillna(365).clip(lower=0)

    # Category
    df["category_clean"] = df.get("category_list", pd.Series(["Other"] * len(df))).apply(map_category)

    # Country / region label encoding
    le_country = LabelEncoder()
    le_region = LabelEncoder()
    le_cat = LabelEncoder()

    df["country_enc"] = le_country.fit_transform(df.get("country_code", pd.Series(["USA"] * len(df))).fillna("USA"))
    df["region_enc"] = le_region.fit_transform(df.get("region", pd.Series(["Unknown"] * len(df))).fillna("Unknown"))
    df["category_enc"] = le_cat.fit_transform(df["category_clean"])

    # Log-transform skewed features
    df["log_funding"] = np.log1p(df["funding_total_usd"])
    df["log_age"] = np.log1p(df["startup_age_years"])
    df["log_days_funding"] = np.log1p(df["days_first_to_last_funding"])
    df["avg_funding_per_round"] = df["funding_total_usd"] / df["funding_rounds"].replace(0, np.nan)
    df["avg_funding_per_round"] = df["avg_funding_per_round"].fillna(funding_median).clip(lower=0)
    df["log_avg_funding_per_round"] = np.log1p(df["avg_funding_per_round"])

    # Is Pakistan
    df["is_pakistan"] = (df.get("country_code", pd.Series([""] * len(df))).fillna("").str.upper() == "PAK").astype(int)

    return df, le_country, le_region, le_cat


FEATURE_COLS = [
    "log_funding", "funding_rounds", "log_age",
    "log_days_funding", "days_to_first_funding",
    "founding_year", "log_avg_funding_per_round",
    "country_enc", "region_enc", "category_enc", "is_pakistan"
]

FEATURE_DISPLAY_NAMES = {
    "log_funding": "Total Funding (log)",
    "funding_rounds": "# Funding Rounds",
    "log_age": "Startup Age (log years)",
    "log_days_funding": "Funding Duration (log days)",
    "days_to_first_funding": "Days to First Funding",
    "founding_year": "Founded Year",
    "log_avg_funding_per_round": "Average Funding per Round",
    "country_enc": "Country",
    "region_enc": "Region",
    "category_enc": "Industry Category",
    "is_pakistan": "Pakistan-based",
}
