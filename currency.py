"""
currency.py — Currency conversion and display formatting.
Supports USD, PKR, EUR, GBP, AED.
"""

# Approximate exchange rates relative to USD (as of early 2025)
EXCHANGE_RATES = {
    "USD": 1.0,
    "PKR": 278.5,    # Pakistani Rupee
    "EUR": 0.92,     # Euro
    "GBP": 0.79,     # British Pound
    "AED": 3.67,     # UAE Dirham (common for Pakistani diaspora)
    "SAR": 3.75,     # Saudi Riyal (Pakistani diaspora)
    "CAD": 1.36,     # Canadian Dollar
}

CURRENCY_SYMBOLS = {
    "USD": "$",
    "PKR": "₨",
    "EUR": "€",
    "GBP": "£",
    "AED": "د.إ",
    "SAR": "﷼",
    "CAD": "CA$",
}

CURRENCY_NAMES = {
    "USD": "US Dollar",
    "PKR": "Pakistani Rupee (PKR)",
    "EUR": "Euro",
    "GBP": "British Pound",
    "AED": "UAE Dirham",
    "SAR": "Saudi Riyal",
    "CAD": "Canadian Dollar",
}


def convert_from_usd(amount_usd: float, target_currency: str) -> float:
    """Convert a USD amount to target currency."""
    rate = EXCHANGE_RATES.get(target_currency, 1.0)
    return amount_usd * rate


def convert_to_usd(amount: float, source_currency: str) -> float:
    """Convert from source currency back to USD for ML model input."""
    rate = EXCHANGE_RATES.get(source_currency, 1.0)
    return amount / rate


def format_currency(amount_usd: float, currency: str, compact: bool = True) -> str:
    """Format a USD amount displayed in the chosen currency."""
    converted = convert_from_usd(amount_usd, currency)
    symbol = CURRENCY_SYMBOLS.get(currency, "$")

    if compact:
        if currency == "PKR":
            if converted >= 1_000_000_000:
                return f"{symbol}{converted/1_000_000_000:.2f}B"
            elif converted >= 10_000_000:
                return f"{symbol}{converted/10_000_000:.2f} Cr"
            elif converted >= 100_000:
                return f"{symbol}{converted/100_000:.2f} Lac"
            else:
                return f"{symbol}{converted:,.0f}"
        else:
            if converted >= 1_000_000_000:
                return f"{symbol}{converted/1_000_000_000:.2f}B"
            elif converted >= 1_000_000:
                return f"{symbol}{converted/1_000_000:.2f}M"
            elif converted >= 1_000:
                return f"{symbol}{converted/1_000:.1f}K"
            else:
                return f"{symbol}{converted:,.0f}"
    else:
        return f"{symbol}{converted:,.2f}"


def get_currency_options():
    return list(EXCHANGE_RATES.keys())
