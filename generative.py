"""
generative.py - Groq-powered business insight layer.
Converts ML/XAI outputs into concise startup analysis and strategy advice.
"""

from __future__ import annotations

import os
from typing import Any

import requests


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _get_api_key() -> str:
    """Read Groq API key from Streamlit secrets or environment variables."""
    try:
        import streamlit as st

        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "").strip()


def _fallback_text(kind: str, context: dict[str, Any]) -> str:
    """Return useful local analysis when the external Groq service is unavailable."""
    probability = float(context.get("probability", 0.0)) * 100
    outcome = "commercially promising" if int(context.get("prediction", 0)) == 1 else "higher risk"

    if kind == "business":
        info = context.get("startup_info", {})
        positives = context.get("top_positive") or "The profile has some supportive commercial signals."
        negatives = context.get("top_negative") or "The model did not identify a single dominant risk factor."
        return (
            f"{info.get('name', 'This startup')} is assessed as {outcome}, with an estimated "
            f"success probability of {probability:.1f}%. The strongest supportive signals are: "
            f"{positives}. The main areas to watch are: {negatives}. Management should treat this "
            "as a decision-support signal and validate it with market traction, customer retention, "
            "unit economics, and funding runway before making major investment decisions."
        )

    if kind == "strategy":
        info = context.get("startup_info", {})
        return (
            "1. Strengthen the evidence behind the score by tracking revenue growth, repeat usage, "
            "customer acquisition cost, and runway on a monthly basis.\n"
            f"2. For a {info.get('category', 'startup')} operating in {info.get('country', 'its target market')}, "
            "prioritize the customer segment with the clearest willingness to pay before expanding into adjacent markets.\n"
            "3. Use the next funding or operating cycle to reduce the largest model risk factors, especially weak traction, "
            "low funding efficiency, or limited commercial partnerships."
        )

    if kind == "whatif":
        original_prob = float(context.get("original_prob", 0.0)) * 100
        new_prob = float(context.get("new_prob", 0.0)) * 100
        changed_feature = context.get("changed_feature", "the selected factor")
        change_direction = context.get("change_direction", "changed")
        delta = new_prob - original_prob
        direction = "improves" if delta >= 0 else "reduces"
        return (
            f"When {changed_feature} is {change_direction}, the estimated success probability moves "
            f"from {original_prob:.1f}% to {new_prob:.1f}%. This {direction} the outlook by "
            f"{abs(delta):.1f} percentage points, so the change is worth prioritizing if it is realistic "
            "and supported by execution capacity."
        )

    allocation = context.get("allocation", {})
    reserve = allocation.get("reserve", 0)
    product = allocation.get("product", 0)
    marketing = allocation.get("marketing", 0)
    return (
        f"The allocation produces an estimated success probability of {probability:.1f}%. "
        f"Product receives {product}% and marketing receives {marketing}%, which should be checked against "
        "the startup's current stage: early ventures usually need stronger product validation, while scaling "
        f"ventures need disciplined acquisition. The {reserve}% reserve is "
        f"{'healthy' if reserve >= 10 else 'thin'}, so keep enough runway for unexpected delays."
    )


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 600,
    fallback_kind: str = "business",
    fallback_context: dict[str, Any] | None = None,
) -> str:
    api_key = _get_api_key()
    if not api_key:
        return _fallback_text(fallback_kind, fallback_context or {})

    payload: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.35,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=45)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.HTTPError:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            detail = resp.text[:250]
        return _fallback_text(fallback_kind, fallback_context or {})
    except Exception as exc:
        return _fallback_text(fallback_kind, fallback_context or {})


def get_business_explanation(
    startup_info: dict,
    prediction: int,
    probability: float,
    top_positive: str,
    top_negative: str,
    currency: str = "USD",
) -> str:
    system = (
        "You are SmartStartup AI, a professional startup analyst. "
        "Write concise, practical business analysis in a serious consulting tone. "
        "Do not use emojis. Avoid hype. Max 180 words."
    )
    user = f"""
Startup Profile:
- Name: {startup_info.get('name', 'This Startup')}
- Industry: {startup_info.get('category', 'Unknown')}
- Country: {startup_info.get('country', 'Unknown')}
- Total Funding: {startup_info.get('funding_display', 'Unknown')} ({currency})
- Funding Rounds: {startup_info.get('funding_rounds', 'Unknown')}
- Startup Age: {startup_info.get('age', 'Unknown')} years

ML Prediction: {'SUCCESS' if prediction == 1 else 'HIGH RISK'} ({probability*100:.1f}% success probability)

Top factors increasing success probability:
{top_positive or 'None significant'}

Top factors reducing success probability:
{top_negative or 'None significant'}

Explain the result in 3-4 sentences. Be specific about strengths, risks, and what management should watch next.
"""
    return _call_groq(
        system,
        user,
        fallback_kind="business",
        fallback_context={
            "startup_info": startup_info,
            "prediction": prediction,
            "probability": probability,
            "top_positive": top_positive,
            "top_negative": top_negative,
        },
    )


def get_strategy_advice(
    startup_info: dict, prediction: int, probability: float, currency: str = "USD"
) -> str:
    system = (
        "You are a senior startup strategist. Give exactly three numbered recommendations. "
        "Each recommendation should be practical, specific, and 1-2 sentences. "
        "Do not use emojis or generic motivational language."
    )
    user = f"""
Startup: {startup_info.get('name', 'This Startup')}
Industry: {startup_info.get('category', 'Unknown')}
Country/Market: {startup_info.get('country', 'Unknown')}
Current Funding: {startup_info.get('funding_display', 'Unknown')} ({currency})
Success Probability: {probability*100:.1f}%
Outcome: {'Predicted successful' if prediction == 1 else 'Predicted high risk'}

Provide three specific strategic recommendations to improve or maintain success.
"""
    return _call_groq(
        system,
        user,
        fallback_kind="strategy",
        fallback_context={
            "startup_info": startup_info,
            "prediction": prediction,
            "probability": probability,
        },
    )


def get_whatif_explanation(
    original_prob: float,
    new_prob: float,
    changed_feature: str,
    change_direction: str,
    currency: str = "USD",
) -> str:
    system = (
        "You are a startup analyst. Explain what-if results in 2-3 concise sentences. "
        "Do not use emojis."
    )
    user = f"""
Original success probability: {original_prob*100:.1f}%
After changing {changed_feature} ({change_direction}): {new_prob*100:.1f}%
Change in probability: {(new_prob - original_prob)*100:+.1f}%
Currency context: {currency}

Explain what this change means for the startup's outlook in plain business language.
"""
    return _call_groq(
        system,
        user,
        max_tokens=220,
        fallback_kind="whatif",
        fallback_context={
            "original_prob": original_prob,
            "new_prob": new_prob,
            "changed_feature": changed_feature,
            "change_direction": change_direction,
        },
    )


def get_gamified_feedback(
    scenario: str, allocation: dict, prediction: int, probability: float, currency: str = "USD"
) -> str:
    system = (
        "You are a professional startup simulation coach. Give educational feedback on "
        "resource allocation. Keep it concise, direct, and free of emojis. Max 150 words."
    )
    budget_fmt = allocation.get("budget_display", "Unknown")
    user = f"""
Startup Simulation Scenario: {scenario}
Budget: {budget_fmt} ({currency})
Resource Allocation:
- Product Development: {allocation.get('product', 0)}%
- Marketing: {allocation.get('marketing', 0)}%
- Operations: {allocation.get('operations', 0)}%
- Team/HR: {allocation.get('team', 0)}%
- Reserve: {allocation.get('reserve', 0)}%

ML Prediction: {'Success' if prediction == 1 else 'High risk'} ({probability*100:.1f}% success probability)

Give feedback on the allocation strategy and the business implications.
"""
    return _call_groq(
        system,
        user,
        max_tokens=300,
        fallback_kind="game",
        fallback_context={
            "scenario": scenario,
            "allocation": allocation,
            "prediction": prediction,
            "probability": probability,
        },
    )
