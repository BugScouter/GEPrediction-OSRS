import pytest
import pandas as pd
import numpy as np
from ge_predictor.features import build_features, FEATURE_COLUMNS
from ge_predictor.models import PricePredictor

def test_feature_engineering_shapes_and_nans():
    timestamps = [1600000000 + i * 3600 for i in range(50)]
    np.random.seed(42)
    prices_high = 1000 + np.cumsum(np.random.randn(50) * 10)
    prices_low = prices_high - 10
    vols_high = np.random.randint(10, 100, size=50)
    vols_low = np.random.randint(10, 100, size=50)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "avgHighPrice": prices_high,
        "avgLowPrice": prices_low,
        "highPriceVolume": vols_high,
        "lowPriceVolume": vols_low
    })

    feat_df = build_features(df, horizons=[1, 3])

    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns

    assert "target_forward_return_1" in feat_df.columns
    assert "target_forward_return_3" in feat_df.columns

    predictor = PricePredictor(target_horizon=1)
    clean_df = predictor.prepare_data(df)

    assert not clean_df[FEATURE_COLUMNS].isnull().any().any()
    assert not clean_df["target_forward_return_1"].isnull().any()
    assert len(clean_df) > 0

def test_model_training():
    timestamps = [1600000000 + i * 3600 for i in range(50)]
    np.random.seed(42)
    prices_high = 1000 + np.cumsum(np.random.randn(50) * 5)
    prices_low = prices_high - 10

    df = pd.DataFrame({
        "timestamp": timestamps,
        "avgHighPrice": prices_high,
        "avgLowPrice": prices_low,
        "highPriceVolume": [50]*50,
        "lowPriceVolume": [50]*50
    })

    predictor = PricePredictor(target_horizon=1)
    results = predictor.fit_with_cv(df, n_splits=3)

    assert "mean_cv_rmse" in results
    assert "mean_cv_directional_accuracy" in results
    assert predictor.model is not None

    pred = predictor.predict_next_return(df)
    assert isinstance(pred, float)
