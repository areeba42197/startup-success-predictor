"""
startup_data.py - Crunchbase + Pakistan dataset builder.

Loads the uploaded Crunchbase zip, creates a Pakistan-focused augmentation
with the same schema, and returns a combined dataset for model training.
"""

from __future__ import annotations

import os
import zipfile
from datetime import timedelta

import numpy as np
import pandas as pd

from pakistan_data import get_pakistan_df
from failed_startups_data import get_failed_startups_df


DATA_COLUMNS = [
    "permalink", "name", "homepage_url", "category_list", "funding_total_usd",
    "status", "country_code", "state_code", "region", "city", "funding_rounds",
    "founded_at", "first_funding_at", "last_funding_at",
]

PAKISTAN_REGIONS = [
    ("Sindh", "Karachi", "SD"),
    ("Punjab", "Lahore", "PJ"),
    ("Punjab", "Islamabad", "PJ"),
    ("Punjab", "Faisalabad", "PJ"),
    ("KPK", "Peshawar", "KP"),
    ("Balochistan", "Quetta", "BA"),
]


def default_crunchbase_zip_path() -> str:
    project_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(project_dir, "..", "crunchbase dataset.zip"))


def load_crunchbase_dataset(zip_path: str | None = None) -> pd.DataFrame:
    """Load the Kaggle Crunchbase CSV from the uploaded zip file."""
    zip_path = zip_path or default_crunchbase_zip_path()
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Crunchbase dataset zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("No CSV file found inside Crunchbase zip.")
        with zf.open(csv_names[0]) as csv_file:
            df = pd.read_csv(csv_file)

    return normalize_startup_schema(df)


def normalize_startup_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the startup dataframe matches the app's expected 14-column schema."""
    df = df.copy()
    for col in DATA_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[DATA_COLUMNS]
    df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")
    df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce").fillna(1).clip(lower=1)
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    df["country_code"] = df["country_code"].fillna("Unknown").astype(str).str.upper()
    return df


def _pick_status(rng: np.random.Generator, funding: float, rounds: int, category: str) -> str:
    success_score = 0.88
    if funding >= 1_000_000:
        success_score += 0.02
    if funding >= 10_000_000:
        success_score += 0.02
    if rounds >= 2:
        success_score += 0.02
    if category in {"FinTech", "Software", "E-Commerce", "HealthTech", "B2B Commerce"}:
        success_score += 0.015
    success_score = min(success_score, 0.965)

    draw = rng.random()
    if draw < success_score * 0.06:
        return "ipo"
    if draw < success_score * 0.22:
        return "acquired"
    if draw < success_score:
        return "operating"
    return "closed"


def generate_pakistan_augmentation(target_count: int, seed: int = 42) -> pd.DataFrame:
    """Generate Pakistan startup rows from real Pakistan startup anchors."""
    rng = np.random.default_rng(seed)
    anchors = normalize_startup_schema(get_pakistan_df())
    categories = [
        "FinTech", "Software", "E-Commerce", "HealthTech", "EdTech", "Logistics",
        "B2B Commerce", "Transportation", "Real Estate", "Analytics & Data",
        "Cloud Computing", "Mobile", "Cybersecurity", "Food & Beverage",
    ]

    rows = []
    for idx in range(target_count):
        anchor = anchors.iloc[int(rng.integers(0, len(anchors)))]
        region, city, state = PAKISTAN_REGIONS[int(rng.integers(0, len(PAKISTAN_REGIONS)))]
        category = anchor["category_list"] if rng.random() < 0.38 else categories[int(rng.integers(0, len(categories)))]
        base_funding = max(float(anchor["funding_total_usd"]), 50_000)
        funding = float(np.clip(base_funding * rng.lognormal(mean=-0.15, sigma=1.05), 15_000, 950_000_000))
        rounds = int(np.clip(round(rng.normal(loc=max(float(anchor["funding_rounds"]), 1.0), scale=1.4)), 1, 12))
        founded_year = int(rng.integers(2006, 2023))
        founded = pd.Timestamp(f"{founded_year}-01-01") + timedelta(days=int(rng.integers(0, 320)))
        first_funding = founded + timedelta(days=int(rng.integers(45, 900)))
        last_funding = first_funding + timedelta(days=int(rng.integers(0, 1500)))
        status = _pick_status(rng, funding, rounds, str(category).split("|")[0])
        name = f"{anchor['name']} PK Variant {idx + 1:05d}"

        rows.append({
            "permalink": f"/organization/{name.lower().replace(' ', '-')}",
            "name": name,
            "homepage_url": f"https://{name.lower().replace(' ', '').replace('.', '')}.pk",
            "category_list": category,
            "funding_total_usd": round(funding, 2),
            "status": status,
            "country_code": "PAK",
            "state_code": state,
            "region": region,
            "city": city,
            "funding_rounds": rounds,
            "founded_at": founded.strftime("%Y-%m-%d"),
            "first_funding_at": first_funding.strftime("%Y-%m-%d"),
            "last_funding_at": last_funding.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(rows, columns=DATA_COLUMNS)


def generate_failure_augmentation(target_count: int, seed: int = 99) -> pd.DataFrame:
    """Generate closed startup rows from recent verified failure anchors."""
    rng = np.random.default_rng(seed)
    anchors = normalize_startup_schema(get_failed_startups_df())
    rows = []
    for idx in range(target_count):
        anchor = anchors.iloc[int(rng.integers(0, len(anchors)))]
        base_funding = max(float(anchor["funding_total_usd"]), 50_000)
        funding = float(np.clip(base_funding * rng.lognormal(mean=-0.25, sigma=0.95), 10_000, 2_000_000_000))
        rounds = int(np.clip(round(rng.normal(loc=max(float(anchor["funding_rounds"]), 1.0), scale=1.6)), 1, 14))
        founded = pd.to_datetime(anchor["founded_at"], errors="coerce")
        if pd.isna(founded):
            founded = pd.Timestamp(f"{int(rng.integers(2012, 2024))}-01-01")
        founded = founded + timedelta(days=int(rng.integers(-180, 365)))
        first_funding = founded + timedelta(days=int(rng.integers(30, 900)))
        last_funding = first_funding + timedelta(days=int(rng.integers(0, 1800)))
        name = f"{anchor['name']} Failure Analog {idx + 1:05d}"

        rows.append({
            "permalink": f"/organization/{name.lower().replace(' ', '-')}",
            "name": name,
            "homepage_url": "",
            "category_list": anchor["category_list"],
            "funding_total_usd": round(funding, 2),
            "status": "closed",
            "country_code": anchor["country_code"],
            "state_code": anchor["state_code"],
            "region": anchor["region"],
            "city": anchor["city"],
            "funding_rounds": rounds,
            "founded_at": founded.strftime("%Y-%m-%d"),
            "first_funding_at": first_funding.strftime("%Y-%m-%d"),
            "last_funding_at": last_funding.strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(rows, columns=DATA_COLUMNS)


def build_combined_startup_dataset(
    zip_path: str | None = None,
    pakistan_share: float = 0.25,
    failure_share: float = 0.25,
    save_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return combined full dataset and the supervised training subset."""
    crunchbase = load_crunchbase_dataset(zip_path)
    verified_pakistan = normalize_startup_schema(get_pakistan_df())
    verified_failures = normalize_startup_schema(get_failed_startups_df())
    existing_pakistan = (crunchbase["country_code"].str.upper() == "PAK").sum()
    generated_count = max(
        0,
        int(round(
            (pakistan_share * len(crunchbase) - existing_pakistan - (1 - pakistan_share) * len(verified_pakistan))
            / (1 - pakistan_share)
        )),
    )
    pakistan_aug = generate_pakistan_augmentation(generated_count)
    base_combined = pd.concat([crunchbase, verified_pakistan, pakistan_aug, verified_failures], ignore_index=True)
    base_combined = normalize_startup_schema(base_combined)
    closed_count = (base_combined["status"] == "closed").sum()
    failure_aug_count = max(
        0,
        int(round((failure_share * len(base_combined) - closed_count) / (1 - failure_share))),
    )
    failure_aug = generate_failure_augmentation(failure_aug_count)
    combined = pd.concat([base_combined, failure_aug], ignore_index=True)
    combined = normalize_startup_schema(combined)

    pakistan_count = (combined["country_code"].str.upper() == "PAK").sum()
    extra_pakistan_count = max(
        0,
        int(round((pakistan_share * len(combined) - pakistan_count) / (1 - pakistan_share))),
    )
    if extra_pakistan_count:
        combined = pd.concat([
            combined,
            generate_pakistan_augmentation(extra_pakistan_count, seed=142),
        ], ignore_index=True)
        combined = normalize_startup_schema(combined)

    closed_count = (combined["status"] == "closed").sum()
    extra_failure_count = max(
        0,
        int(round((failure_share * len(combined) - closed_count) / (1 - failure_share))),
    )
    if extra_failure_count:
        combined = pd.concat([
            combined,
            generate_failure_augmentation(extra_failure_count, seed=199),
        ], ignore_index=True)

    combined = normalize_startup_schema(combined)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        combined.to_csv(save_path, index=False)

    supervised = combined[combined["status"].isin(["operating", "acquired", "ipo", "closed"])].copy()
    return combined, supervised
