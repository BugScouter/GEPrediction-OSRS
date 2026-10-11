from typing import Dict, Any, List, Optional
import pandas as pd
from ge_predictor.market_rules import calculate_ge_tax, calculate_net_margin, EXEMPT_ITEM_IDS, EXEMPT_ITEM_NAMES

def score_flip_opportunity(
    item_id: int,
    item_name: str,
    buy_price: int,
    sell_price: int,
    buy_limit: int,
    budget_gp: int,
    hourly_volume: float = 0,
    predicted_return_pct: float = 0.0,
) -> Dict[str, Any]:
    """
    Evaluates a flipping opportunity considering budget, GE tax, buy limit, and predicted return.
    """
    is_exempt = (item_id in EXEMPT_ITEM_IDS) or (item_name in EXEMPT_ITEM_NAMES)
    tax = calculate_ge_tax(sell_price, is_exempt)
    net_margin = calculate_net_margin(buy_price, sell_price, is_exempt)

    if buy_price <= 0:
        max_quantity_for_budget = 0
        roi = 0.0
    else:
        max_quantity_for_budget = budget_gp // buy_price
        roi = (net_margin / buy_price) * 100.0

    trade_quantity = min(buy_limit if buy_limit > 0 else max_quantity_for_budget, max_quantity_for_budget)
    potential_4h_profit = net_margin * trade_quantity
    capital_required = buy_price * trade_quantity

    return {
        "item_id": item_id,
        "item_name": item_name,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "tax_per_item": tax,
        "net_margin": net_margin,
        "roi_pct": round(roi, 2),
        "buy_limit": buy_limit,
        "trade_quantity": trade_quantity,
        "capital_required": capital_required,
        "potential_4h_profit": potential_4h_profit,
        "hourly_volume": hourly_volume,
        "predicted_return_pct": round(predicted_return_pct * 100, 2)
    }

def find_top_flips(
    items_data: List[Dict[str, Any]],
    budget_gp: int = 10_000_000,
    min_roi: float = 1.0,
    min_hourly_volume: float = 100.0,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Filters and ranks flipping opportunities from a list of market items.
    """
    scored = []
    for item in items_data:
        vol = item.get("hourly_volume", 0)
        if vol < min_hourly_volume:
            continue

        buy_price = item.get("buy_price", 0)
        sell_price = item.get("sell_price", 0)
        if buy_price <= 0 or sell_price <= buy_price:
            continue

        item_id = item.get("id") or item.get("item_id", 0)
        item_name = item.get("name") or item.get("item_name", "Unknown")
        buy_limit = item.get("limit") or item.get("buy_limit", 10000)
        pred_return = item.get("predicted_return_pct", 0.0)

        res = score_flip_opportunity(
            item_id=item_id,
            item_name=item_name,
            buy_price=buy_price,
            sell_price=sell_price,
            buy_limit=buy_limit,
            budget_gp=budget_gp,
            hourly_volume=vol,
            predicted_return_pct=pred_return
        )

        if res["roi_pct"] >= min_roi and res["potential_4h_profit"] > 0:
            scored.append(res)

    # Sort by potential_4h_profit descending
    scored.sort(key=lambda x: x["potential_4h_profit"], reverse=True)
    return scored[:top_n]
