from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ge_predictor.client import OSRSClient
from ge_predictor.market_rules import calculate_ge_tax, calculate_net_margin, EXEMPT_ITEM_IDS, EXEMPT_ITEM_NAMES
from ge_predictor.models import PricePredictor
from ge_predictor.scoring import find_top_flips, score_flip_opportunity

app = FastAPI(
    title="OSRS GE Price Predictor & Flipping Screener API",
    version="1.0.0",
    description="REST API for real-time OSRS Grand Exchange price prediction and tax-adjusted flipping opportunities."
)

client = OSRSClient()

_mapping_cache: Optional[List[Dict[str, Any]]] = None

def get_cached_mapping() -> List[Dict[str, Any]]:
    global _mapping_cache
    if _mapping_cache is None:
        _mapping_cache = client.get_mapping()
    return _mapping_cache

class HealthResponse(BaseModel):
    status: str
    version: str

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/items")
def search_items(query: Optional[str] = Query(None, description="Search term for item name")):
    mapping = get_cached_mapping()
    if not query:
        return mapping[:50]
    query_lower = query.lower()
    filtered = [item for item in mapping if query_lower in item.get("name", "").lower()]
    return filtered

@app.get("/predict/{item_id}")
def predict_item(item_id: int):
    mapping = get_cached_mapping()
    item_meta = next((item for item in mapping if item["id"] == item_id), None)
    if not item_meta:
        raise HTTPException(status_code=404, detail=f"Item ID {item_id} not found in GE catalog.")

    latest_data = client.get_latest(item_id)
    item_prices = latest_data.get(str(item_id)) or latest_data.get(item_id)

    if not item_prices:
        raise HTTPException(status_code=404, detail=f"Price data for Item ID {item_id} currently unavailable.")

    high_price = item_prices.get("high") or 0
    low_price = item_prices.get("low") or 0

    try:
        ts_data = client.get_timeseries(item_id, timestep="1h")
        if len(ts_data) >= 25:
            import pandas as pd
            df = pd.DataFrame(ts_data)
            predictor = PricePredictor(target_horizon=1)
            predictor.fit_with_cv(df, n_splits=3)
            pred_return = predictor.predict_next_return(df)
        else:
            pred_return = 0.0
    except Exception:
        pred_return = 0.0

    is_exempt = (item_id in EXEMPT_ITEM_IDS) or (item_meta.get("name") in EXEMPT_ITEM_NAMES)
    tax = calculate_ge_tax(high_price, is_exempt) if high_price else 0
    margin = calculate_net_margin(low_price, high_price, is_exempt) if (high_price and low_price) else 0

    if pred_return > 0.005 and margin > 0:
        action = "BUY"
    elif pred_return < -0.005:
        action = "SELL"
    else:
        action = "HOLD"

    return {
        "item_id": item_id,
        "item_name": item_meta.get("name"),
        "high_price": high_price,
        "low_price": low_price,
        "tax": tax,
        "net_margin": margin,
        "predicted_forward_return_pct": round(pred_return * 100, 2),
        "recommended_action": action
    }

@app.get("/screener")
def screener(
    budget: int = Query(10_000_000, description="Cash budget in GP"),
    min_roi: float = Query(1.0, description="Minimum net ROI %"),
    min_volume: float = Query(50.0, description="Minimum hourly trading volume")
):
    mapping = get_cached_mapping()
    latest = client.get_latest()

    items_data = []
    for m in mapping:
        item_id = m["id"]
        prices = latest.get(str(item_id)) or latest.get(item_id)
        if not prices:
            continue

        high_price = prices.get("high")
        low_price = prices.get("low")
        if not high_price or not low_price or low_price >= high_price:
            continue

        items_data.append({
            "id": item_id,
            "name": m.get("name"),
            "buy_price": low_price,
            "sell_price": high_price,
            "limit": m.get("limit") or 10000,
            "hourly_volume": 100.0
        })

    results = find_top_flips(items_data, budget_gp=budget, min_roi=min_roi, min_hourly_volume=min_volume, top_n=20)
    return results
