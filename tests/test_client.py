import pytest
from unittest.mock import MagicMock
from ge_predictor.client import OSRSClient

def test_client_latest(mocker):
    client = OSRSClient()
    mock_get = mocker.patch.object(client.session, 'get')
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": {
            "2": {"high": 200, "highTime": 1000, "low": 195, "lowTime": 990}
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    latest = client.get_latest(item_id=2)
    assert "2" in latest
    assert latest["2"]["high"] == 200
    mock_get.assert_called_once()

def test_client_timeseries(mocker):
    client = OSRSClient()
    mock_get = mocker.patch.object(client.session, 'get')
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"timestamp": 1000, "avgHighPrice": 200, "avgLowPrice": 195, "highPriceVolume": 10, "lowPriceVolume": 15}
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    ts = client.get_timeseries(item_id=2, timestep="1h")
    assert len(ts) == 1
    assert ts[0]["avgHighPrice"] == 200

def test_client_mapping(mocker):
    client = OSRSClient()
    mock_get = mocker.patch.object(client.session, 'get')
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"id": 2, "name": "Cannonball", "limit": 10000}
    ]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    mapping = client.get_mapping()
    assert len(mapping) == 1
    assert mapping[0]["name"] == "Cannonball"
