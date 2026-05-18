"""
xai.py — Explainable AI using SHAP.
Generates feature importance, local explanations, and plots.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, warnings
warnings.filterwarnings("ignore")

from preprocessing import FEATURE_COLS, FEATURE_DISPLAY_NAMES


def get_shap_explainer(model, X_background, model_name="XGBoost"):
    """Create SHAP explainer appropriate for model type."""
    if model_name == "XGBoost":
        return shap.TreeExplainer(model)
    elif model_name == "Random Forest":
        return shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_background, 50))
        return explainer


def get_shap_values_single(explainer, X_input, model_name="XGBoost"):
    """Get SHAP values for a single input."""
    if model_name in ["XGBoost", "Random Forest"]:
        sv = explainer.shap_values(X_input)
        # For tree models with binary classification, shap_values is a list or array
        if isinstance(sv, list):
            return sv[1][0] if len(sv) > 1 else sv[0][0]
        if sv.ndim == 3:
            return sv[0, :, 1]  # (samples, features, classes)[sample, :, positive_class]
        return sv[0]
    else:
        sv = explainer.shap_values(X_input)
        if isinstance(sv, list):
            return sv[1][0]
        return sv[0]


def plot_shap_bar(shap_values, feature_names=None):
    """Return a matplotlib figure of a SHAP waterfall bar chart."""
    if feature_names is None:
        feature_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_COLS]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#E63946" if v < 0 else "#2A9D8F" for v in shap_values]
    y_pos = np.arange(len(shap_values))
    ax.barh(y_pos, shap_values, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=11)
    ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=11)
    ax.set_title("Feature Impact on This Prediction", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#F8F9FA")
    plt.tight_layout()
    return fig


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
