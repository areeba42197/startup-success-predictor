"""
xai.py - Explainable AI helpers.
Uses SHAP/LIME when installed, with lightweight fallbacks for Streamlit Cloud.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from preprocessing import FEATURE_COLS, FEATURE_DISPLAY_NAMES


def _fallback_values(X_background, X_input):
    bg = np.asarray(X_background, dtype=float)
    x = np.asarray(X_input, dtype=float)[0]
    if bg.ndim != 2 or bg.size == 0:
        return np.zeros_like(x)
    med = np.nanmedian(bg, axis=0)
    spread = np.nanstd(bg, axis=0)
    spread = np.where(spread == 0, 1, spread)
    values = (x - med) / spread
    return np.clip(values, -1.5, 1.5) * 0.35


def get_shap_explainer(model, X_background, model_name="XGBoost"):
    """Create SHAP explainer appropriate for model type."""
    if shap is None:
        return {"fallback_background": np.asarray(X_background)}
    if model_name in ["XGBoost", "Random Forest"]:
        return shap.TreeExplainer(model)
    return shap.KernelExplainer(model.predict_proba, shap.sample(X_background, 50))


def get_shap_values_single(explainer, X_input, model_name="XGBoost"):
    """Get SHAP values for a single input."""
    if isinstance(explainer, dict) and "fallback_background" in explainer:
        return _fallback_values(explainer["fallback_background"], X_input)

    if model_name in ["XGBoost", "Random Forest"]:
        sv = explainer.shap_values(X_input)
        if isinstance(sv, list):
            return sv[1][0] if len(sv) > 1 else sv[0][0]
        if sv.ndim == 3:
            return sv[0, :, 1]
        return sv[0]

    sv = explainer.shap_values(X_input)
    if isinstance(sv, list):
        return sv[1][0]
    return sv[0]


def plot_shap_bar(shap_values, feature_names=None):
    """Return a matplotlib figure of a SHAP-style feature impact bar chart."""
    if plt is None:
        return None
    if feature_names is None:
        feature_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_COLS]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E63946" if v < 0 else "#2A9D8F" for v in shap_values]
    y_pos = np.arange(len(shap_values))
    ax.barh(y_pos, shap_values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=11)
    ax.set_xlabel("Feature impact on prediction", fontsize=11)
    ax.set_title("Feature Impact on This Prediction", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")
    plt.tight_layout()
    return fig


def get_lime_explanation(model, X_background, X_input, feature_names, model_name="XGBoost", scaler=None):
    """Return LIME local feature weights for one startup prediction."""
    if LimeTabularExplainer is None:
        values = _fallback_values(X_background, X_input)
        rows = []
        for feature, weight in sorted(zip(feature_names, values), key=lambda item: abs(item[1]), reverse=True)[:8]:
            rows.append({
                "Feature condition": feature,
                "Effect": "Supports success" if weight >= 0 else "Supports high risk",
                "Weight": round(float(weight), 4),
            })
        return rows

    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_background),
        feature_names=feature_names,
        class_names=["High Risk", "Success"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    def predict_fn(samples):
        if model_name == "Logistic Regression" and scaler is not None:
            return model.predict_proba(scaler.transform(samples))
        return model.predict_proba(samples)

    explanation = explainer.explain_instance(
        np.asarray(X_input)[0],
        predict_fn,
        num_features=min(8, len(feature_names)),
        labels=(1,),
    )
    rows = []
    for condition, weight in explanation.as_list(label=1):
        rows.append({
            "Feature condition": condition,
            "Effect": "Supports success" if weight >= 0 else "Supports high risk",
            "Weight": round(float(weight), 4),
        })
    return rows


def get_feature_importance_text(shap_values, feature_names=None):
    """Return top positive and negative feature influences as text."""
    if feature_names is None:
        feature_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_COLS]

    pairs = sorted(zip(feature_names, shap_values), key=lambda x: x[1])
    negative = [(n, v) for n, v in pairs if v < 0]
    positive = [(n, v) for n, v in pairs if v >= 0][::-1]

    pos_text = "\n".join([f"- **{n}** (+{v:.3f})" for n, v in positive[:3]])
    neg_text = "\n".join([f"- **{n}** ({v:.3f})" for n, v in negative[:3]])
    return pos_text, neg_text
