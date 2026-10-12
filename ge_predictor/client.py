from typing import Dict, Any, List, Optional
import requests

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"
DEFAULT_USER_AGENT = "GEPredictor-Modernized - Contact: project-maintainer@example.com"

class OSRSClient:
    """
    Client for querying OSRS Wiki Real-time Prices API.
    """
    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, session: Optional[requests.Session] = None):
        self.user_agent = user_agent
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def get_latest(self, item_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch real-time high/low snapshots and timestamps.
        If item_id is provided, returns latest prices for that item.
        """
        params = {}
        if item_id is not None:
            params["id"] = item_id
        url = f"{BASE_URL}/latest"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json().get("data", {})

    def get_timeseries(self, item_id: int, timestep: str = "5m") -> List[Dict[str, Any]]:
        """
        Fetch historical aggregates (timestep: '5m', '1h', etc.).
        Returns list of dict entries with timestamp, avgHighPrice, avgLowPrice, highPriceVolume, lowPriceVolume.
        """
        url = f"{BASE_URL}/timeseries"
        params = {"id": item_id, "timestep": timestep}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json().get("data", [])

    def get_mapping(self) -> List[Dict[str, Any]]:
        """
        Fetch item catalog metadata (IDs, names, 4-hour buy limits, high-alch values, etc.).
        """
        url = f"{BASE_URL}/mapping"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
