import math
from typing import Dict, Any, List

EXEMPT_ITEM_IDS = {13190}
EXEMPT_ITEM_NAMES = {"Old school bond"}

def calculate_ge_tax(sell_price: int, is_exempt: bool = False) -> int:
    """
    Calculates the 2% GE tax for a given sell price.

    Rules:
    - Rate: 2% on seller's final gross price.
    - Rounded down per item: math.floor(sell_price * 0.02)
    - Minimum threshold: Any item selling below 50 GP pays 0 GP tax.
    - Cap: Maximum tax is capped at 5,000,000 GP per item (reached at 250,000,000 GP).
    - Exemption check: Support tax-exempt items.
    """
    if is_exempt or sell_price < 50:
        return 0
    return min(5_000_000, math.floor(sell_price * 0.02))

def calculate_net_margin(buy_price: int, sell_price: int, is_exempt: bool = False) -> int:
    """
    Calculates net margin after accounting for GE tax on sell_price.
    """
    tax = calculate_ge_tax(sell_price, is_exempt)
    return (sell_price - tax) - buy_price
