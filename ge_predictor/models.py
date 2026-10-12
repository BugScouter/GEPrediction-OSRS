from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from ge_predictor.features import build_features, FEATURE_COLUMNS

class PricePredictor:
    """
    LightGBM model for forecasting forward price returns using walk-forward TimeSeriesSplit.
    """
    def __init__(self, target_horizon: int = 1, lgb_params: Optional[Dict[str, Any]] = None):
        self.target_horizon = target_horizon
        self.target_col = f"target_forward_return_{target_horizon}"
        self.feature_cols = FEATURE_COLUMNS
        self.params = lgb_params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "verbose": -1,
        }
        self.model: Optional[lgb.LGBMRegressor] = None

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features and drop rows where features or target are NaN.
        """
        data = build_features(df, horizons=[self.target_horizon])
        required_cols = self.feature_cols + [self.target_col]
        clean_df = data.dropna(subset=required_cols).copy()
        return clean_df

    def fit_with_cv(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
        """
        Train LightGBM using walk-forward splits (TimeSeriesSplit) to evaluate metrics without lookahead bias.
        After CV, fits the final model on all valid data.
        """
        clean_df = self.prepare_data(df)
        if len(clean_df) < n_splits + 10:
            raise ValueError(f"Not enough clean rows ({len(clean_df)}) for {n_splits} splits.")

        X = clean_df[self.feature_cols]
        y = clean_df[self.target_col]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        directional_accuracies = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fold_model = lgb.LGBMRegressor(**self.params)
            fold_model.fit(X_train, y_train)

            preds = fold_model.predict(X_test)
            try:
                rmse = mean_squared_error(y_test, preds, squared=False)
            except TypeError:
                rmse = np.sqrt(mean_squared_error(y_test, preds))
            cv_scores.append(rmse)

            # Directional accuracy: sign match
            correct_direction = np.sign(y_test) == np.sign(preds)
            directional_accuracies.append(np.mean(correct_direction))

        # Train final model on full clean dataset
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)

        return {
            "mean_cv_rmse": float(np.mean(cv_scores)),
            "mean_cv_directional_accuracy": float(np.mean(directional_accuracies)),
            "n_samples": len(clean_df)
        }

    def predict_next_return(self, df: pd.DataFrame) -> float:
        """
        Given recent timeseries data, predict the next N-step forward return.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")

        data = build_features(df, horizons=[self.target_horizon])
        latest_features = data[self.feature_cols].tail(1)
        if latest_features.isnull().values.any():
            latest_features = latest_features.ffill().bfill().fillna(0)

        pred = self.model.predict(latest_features)
        return float(pred[0])
