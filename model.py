"""
model.py — ML model training, evaluation, and prediction.
Trains Logistic Regression, Random Forest, and XGBoost classifiers.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, balanced_accuracy_score,
                             average_precision_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle, os, warnings
warnings.filterwarnings("ignore")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models.pkl")


def train_models(X, y):
    """Train all three models; return fitted models, scaler, and metrics."""
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    classes, counts = np.unique(y_train, return_counts=True)
    minority_rate = counts.min() / counts.sum() if len(classes) > 1 else 1.0
    if len(classes) > 1 and counts.min() > 1 and minority_rate < 0.08:
        smote = SMOTE(random_state=42, k_neighbors=min(5, counts.min() - 1))
        X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)
    else:
        X_train_fit, y_train_fit = X_train, y_train

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_fit)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    neg_count = max(1, int(np.sum(y_train == 0)))
    pos_count = max(1, int(np.sum(y_train == 1)))
    xgb_pos_weight = neg_count / pos_count

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=16, min_samples_leaf=2,
            random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, random_state=42, eval_metric="logloss",
            tree_method="hist", scale_pos_weight=xgb_pos_weight, verbosity=0
        ),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        Xtr = X_train_sc if name == "Logistic Regression" else X_train_fit
        Xv = X_val_sc if name == "Logistic Regression" else X_val
        Xte = X_test_sc if name == "Logistic Regression" else X_test
        Xtr_eval = scaler.transform(X_train) if name == "Logistic Regression" else X_train
        model.fit(Xtr, y_train_fit)

        val_prob = model.predict_proba(Xv)[:, 1]
        thresholds = np.linspace(0.05, 0.95, 181)
        val_scores = [
            balanced_accuracy_score(y_val, (val_prob >= threshold).astype(int))
            for threshold in thresholds
        ]
        best_threshold = float(thresholds[int(np.argmax(val_scores))])
        model.smart_threshold_ = best_threshold

        y_train_prob = model.predict_proba(Xtr_eval)[:, 1]
        y_prob = model.predict_proba(Xte)[:, 1]
        y_train_pred = (y_train_prob >= best_threshold).astype(int)
        y_pred = (y_prob >= best_threshold).astype(int)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        failure_precision = tn / (tn + fn) if (tn + fn) else 0.0
        failure_recall = specificity
        results[name] = {
            "train_accuracy": train_accuracy,
            "accuracy": test_accuracy,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "specificity": specificity,
            "failure_precision": failure_precision,
            "failure_recall": failure_recall,
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "threshold": best_threshold,
            "overfit_gap": train_accuracy - test_accuracy,
            "cm": cm,
        }
        trained[name] = model

    return trained, scaler, results, X_test, y_test


def predict_single(models_dict, scaler, X_input: np.ndarray, model_name: str = "XGBoost"):
    """Predict a single sample. Returns (label, probability)."""
    model = models_dict[model_name]
    if model_name == "Logistic Regression":
        X_sc = scaler.transform(X_input)
        prob = model.predict_proba(X_sc)[0][1]
    else:
        prob = model.predict_proba(X_input)[0][1]
    threshold = float(getattr(model, "smart_threshold_", 0.5))
    return int(prob >= threshold), float(prob)


def save_models(trained, scaler, results, le_country, le_region, le_cat):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "models": trained, "scaler": scaler,
            "results": results,
            "le_country": le_country, "le_region": le_region, "le_cat": le_cat
        }, f)


def load_models():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None
