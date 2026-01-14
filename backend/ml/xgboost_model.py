"""
XGBoost Model for Stock Price Prediction

Features:
- Feature engineering from OHLCV data
- Time-series aware cross-validation
- Probability calibration for directional predictions
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model"""
    # Model hyperparameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    reg_alpha: float = 0.1  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization

    # Training settings
    early_stopping_rounds: int = 10
    eval_metric: str = "rmse"

    # Feature engineering
    lookback_periods: List[int] = None
    use_technical_indicators: bool = True

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [1, 2, 3, 5, 10, 20]


class XGBoostPredictor:
    """
    XGBoost-based stock price predictor with feature engineering.

    Features used:
    - Price returns (1, 2, 3, 5, 10, 20 day)
    - Moving average ratios
    - RSI
    - MACD
    - Volume changes
    - Volatility measures
    """

    def __init__(self, config: Optional[XGBoostConfig] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")

        self.config = config or XGBoostConfig()
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=df.index)

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Price returns for different lookback periods
        for period in self.config.lookback_periods:
            features[f'return_{period}d'] = close.pct_change(period)

        # Moving average ratios
        for period in [5, 10, 20, 50]:
            ma = close.rolling(period).mean()
            features[f'ma_{period}_ratio'] = close / ma - 1

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # Bollinger Bands position
        bb_ma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_position'] = (close - bb_ma) / (2 * bb_std)

        # Volume features
        features['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        features['volume_change'] = volume.pct_change()

        # Volatility
        features['volatility_10d'] = close.pct_change().rolling(10).std()
        features['volatility_20d'] = close.pct_change().rolling(20).std()

        # High-Low range
        features['hl_range'] = (high - low) / close
        features['hl_range_ma'] = features['hl_range'].rolling(10).mean()

        # Price momentum
        features['momentum_10d'] = close / close.shift(10) - 1
        features['momentum_20d'] = close / close.shift(20) - 1

        # Rate of change
        features['roc_10'] = close.pct_change(10)
        features['roc_20'] = close.pct_change(20)

        # Stochastic oscillator
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        features['stoch_k'] = (close - low_14) / (high_14 - low_14) * 100
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()

        # Day of week (cyclical)
        if hasattr(df.index, 'dayofweek'):
            dow = df.index.dayofweek
            features['day_sin'] = np.sin(2 * np.pi * dow / 5)
            features['day_cos'] = np.cos(2 * np.pi * dow / 5)

        self.feature_names = features.columns.tolist()

        return features

    def _prepare_data(
        self,
        df: pd.DataFrame,
        target_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """
        Prepare features and target for training.

        Args:
            df: OHLCV DataFrame
            target_horizon: Days ahead to predict

        Returns:
            Tuple of (X, y, valid_indices)
        """
        features = self._create_features(df)

        # Target: next day return
        target = df['Close'].pct_change(target_horizon).shift(-target_horizon)

        # Combine and drop NaN rows
        combined = pd.concat([features, target.rename('target')], axis=1)
        combined = combined.dropna()

        X = combined[self.feature_names].values
        y = combined['target'].values
        valid_indices = combined.index

        return X, y, valid_indices

    def fit(
        self,
        df: pd.DataFrame,
        target_horizon: int = 1,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.

        Args:
            df: OHLCV DataFrame
            target_horizon: Days ahead to predict
            validation_split: Fraction of data for validation

        Returns:
            Training metrics
        """
        X, y, indices = self._prepare_data(df, target_horizon)

        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)} samples")

        # Time-series split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )

        self.is_fitted = True

        # Calculate metrics
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)

        train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
        val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))

        # Directional accuracy
        train_direction_acc = np.mean(np.sign(train_pred) == np.sign(y_train))
        val_direction_acc = np.mean(np.sign(val_pred) == np.sign(y_val))

        return {
            "train_rmse": float(train_rmse),
            "val_rmse": float(val_rmse),
            "train_direction_accuracy": float(train_direction_acc),
            "val_direction_accuracy": float(val_direction_acc),
            "n_features": len(self.feature_names),
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val)
        }

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction for the next period.

        Args:
            df: Recent OHLCV data

        Returns:
            Prediction with confidence
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        features = self._create_features(df)

        # Get the last valid row
        features = features.dropna()
        if features.empty:
            raise ValueError("Insufficient data for prediction")

        X = features.iloc[-1:][self.feature_names].values
        X_scaled = self.scaler.transform(X)

        # Predict return
        predicted_return = float(self.model.predict(X_scaled)[0])

        # Get current price
        current_price = df['Close'].iloc[-1]
        predicted_price = current_price * (1 + predicted_return)

        # Calculate confidence based on feature importance alignment
        feature_importance = self.get_feature_importance()

        # Use prediction variance as confidence proxy
        # More extreme predictions = less confident
        confidence = max(0.3, 1 - abs(predicted_return) * 10)
        confidence = min(confidence, 0.95)

        return {
            "predicted_return": round(predicted_return * 100, 4),
            "predicted_price": round(predicted_price, 2),
            "current_price": round(current_price, 2),
            "direction": "UP" if predicted_return > 0 else "DOWN",
            "confidence": round(confidence, 3),
            "model": "XGBoost"
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.is_fitted or self.model is None:
            return {}

        importance = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importance))

        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_importance

    def cross_validate(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        target_horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation.

        Args:
            df: OHLCV DataFrame
            n_splits: Number of CV splits
            target_horizon: Days ahead to predict

        Returns:
            Cross-validation results
        """
        X, y, _ = self._prepare_data(df, target_horizon)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        rmse_scores = []
        direction_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = xgb.XGBRegressor(
                n_estimators=50,  # Fewer for CV speed
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=42
            )

            model.fit(X_train_scaled, y_train, verbose=False)
            y_pred = model.predict(X_val_scaled)

            rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
            direction_acc = np.mean(np.sign(y_pred) == np.sign(y_val))

            rmse_scores.append(rmse)
            direction_scores.append(direction_acc)

        return {
            "mean_rmse": float(np.mean(rmse_scores)),
            "std_rmse": float(np.std(rmse_scores)),
            "mean_direction_accuracy": float(np.mean(direction_scores)),
            "std_direction_accuracy": float(np.std(direction_scores)),
            "n_splits": n_splits
        }

    def load_pretrained(self, symbol: str = "default") -> bool:
        """
        Load pre-trained model weights.

        Args:
            symbol: Symbol to load (falls back to 'default')

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            from .persistence import get_model_persistence

            persistence = get_model_persistence()
            data = persistence.load_xgboost(symbol)

            if data is None:
                return False

            self.model = data["model"]
            self.scaler = data["scaler"]
            self.feature_names = data["feature_names"]
            self.is_fitted = True

            logger.info(f"XGBoost loaded pre-trained weights for '{symbol}'")
            return True

        except Exception as e:
            logger.warning(f"Failed to load XGBoost pre-trained model: {e}")
            return False

    def save_trained(self, symbol: str = "default", metadata: Optional[Dict] = None) -> bool:
        """
        Save current trained model.

        Args:
            symbol: Symbol identifier
            metadata: Optional training metadata

        Returns:
            True if saved successfully
        """
        if not self.is_fitted or self.model is None:
            logger.warning("XGBoost model not fitted, cannot save")
            return False

        try:
            from .persistence import get_model_persistence

            persistence = get_model_persistence()
            persistence.save_xgboost(
                self.model,
                self.scaler,
                self.feature_names,
                symbol,
                metadata
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save XGBoost model: {e}")
            return False
