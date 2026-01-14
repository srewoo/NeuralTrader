"""
Ensemble Model Coordinator for Stock Price Prediction

Combines predictions from multiple models:
- LSTM (neural network for sequence patterns)
- XGBoost (gradient boosting for feature-based prediction)
- Transformer (attention-based sequence modeling)

Uses weighted averaging with dynamic weight adjustment based on
recent prediction accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble predictor"""
    # Model weights (LSTM, XGBoost, Transformer)
    initial_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])

    # Dynamic weight adjustment
    use_dynamic_weights: bool = True
    weight_lookback: int = 10  # Days to evaluate for weight adjustment

    # Model availability
    use_lstm: bool = True
    use_xgboost: bool = True
    use_transformer: bool = True

    # Minimum samples for training
    min_training_samples: int = 100

    # Confidence thresholds
    high_confidence_threshold: float = 0.7
    low_confidence_threshold: float = 0.4


class EnsemblePredictor:
    """
    Ensemble predictor combining LSTM, XGBoost, and Transformer models.

    Features:
    - Pre-trained model loading at initialization
    - Parallel model predictions
    - Weighted averaging with configurable weights
    - Dynamic weight adjustment based on recent accuracy
    - Confidence aggregation
    - Fallback to available models if some fail
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.weights = self.config.initial_weights.copy()

        # Model instances (loaded with pre-training support)
        self._lstm_model = None
        self._xgboost_model = None
        self._transformer_model = None

        # Track which models have pre-trained weights
        self._models_pretrained = {
            "lstm": False,
            "xgboost": False,
            "transformer": False
        }

        # Track prediction history for weight adjustment
        self.prediction_history: List[Dict[str, Any]] = []

        # Thread pool for parallel predictions
        self._executor = ThreadPoolExecutor(max_workers=3)

        # Attempt to load pre-trained models at initialization
        self._load_pretrained_models()

    def _load_pretrained_models(self):
        """Load all available pre-trained models."""
        logger.info("Loading pre-trained ML models...")

        # Load XGBoost (with pre-trained weights)
        if self.config.use_xgboost:
            try:
                from .xgboost_model import XGBoostPredictor
                xgb = XGBoostPredictor()
                if xgb.load_pretrained("default"):
                    self._xgboost_model = xgb
                    self._models_pretrained["xgboost"] = True
                    logger.info("XGBoost pre-trained model loaded")
                else:
                    self._xgboost_model = xgb  # Keep instance for on-the-fly training
                    logger.info("XGBoost will train on-the-fly (no pre-trained weights)")
            except Exception as e:
                logger.warning(f"Failed to load XGBoost: {e}")

        # Load Transformer (with pre-trained weights)
        if self.config.use_transformer:
            try:
                from .transformer_model import TransformerPredictor
                transformer = TransformerPredictor()
                if transformer.load_pretrained("default"):
                    self._transformer_model = transformer
                    self._models_pretrained["transformer"] = True
                    logger.info("Transformer pre-trained model loaded")
                else:
                    self._transformer_model = transformer  # Keep instance for on-the-fly training
                    logger.info("Transformer will train on-the-fly (no pre-trained weights)")
            except Exception as e:
                logger.warning(f"Failed to load Transformer: {e}")

        # LSTM is loaded via MLService (it handles pre-training internally)
        if self.config.use_lstm:
            try:
                from .inference import get_ml_service
                ml_service = get_ml_service()
                self._lstm_model = ml_service
                self._models_pretrained["lstm"] = ml_service.is_pretrained
                if ml_service.is_pretrained:
                    logger.info("LSTM pre-trained model loaded")
                else:
                    logger.info("LSTM will train on-the-fly (no pre-trained weights)")
            except Exception as e:
                logger.warning(f"Failed to load LSTM: {e}")

        pretrained_count = sum(self._models_pretrained.values())
        logger.info(f"Loaded {pretrained_count}/3 pre-trained models")

    def get_pretrained_status(self) -> Dict[str, bool]:
        """
        Return which models have pre-trained weights loaded.

        Returns:
            Dict mapping model name to whether it's pre-trained
        """
        return self._models_pretrained.copy()

    @property
    def lstm_model(self):
        """Get LSTM model (pre-trained or lazy-loaded)"""
        if self._lstm_model is None and self.config.use_lstm:
            try:
                from .inference import get_ml_service
                self._lstm_model = get_ml_service()
                self._models_pretrained["lstm"] = self._lstm_model.is_pretrained
            except Exception as e:
                logger.warning(f"LSTM model not available: {e}")
        return self._lstm_model

    @property
    def xgboost_model(self):
        """Get XGBoost model (pre-trained or lazy-loaded)"""
        if self._xgboost_model is None and self.config.use_xgboost:
            try:
                from .xgboost_model import XGBoostPredictor
                self._xgboost_model = XGBoostPredictor()
                # Try to load pre-trained
                if self._xgboost_model.load_pretrained("default"):
                    self._models_pretrained["xgboost"] = True
            except Exception as e:
                logger.warning(f"XGBoost model not available: {e}")
        return self._xgboost_model

    @property
    def transformer_model(self):
        """Get Transformer model (pre-trained or lazy-loaded)"""
        if self._transformer_model is None and self.config.use_transformer:
            try:
                from .transformer_model import TransformerPredictor
                self._transformer_model = TransformerPredictor()
                # Try to load pre-trained
                if self._transformer_model.load_pretrained("default"):
                    self._models_pretrained["transformer"] = True
            except Exception as e:
                logger.warning(f"Transformer model not available: {e}")
        return self._transformer_model

    async def _predict_lstm(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Get LSTM prediction"""
        try:
            if self.lstm_model is None:
                return None
            return await self.lstm_model.predict_next_price(symbol, df)
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
            return None

    def _predict_xgboost(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get XGBoost prediction"""
        try:
            if self.xgboost_model is None:
                return None

            # Train on-the-fly if not fitted
            if not self.xgboost_model.is_fitted:
                train_df = df.iloc[:-30] if len(df) > 130 else df.iloc[:-10]
                if len(train_df) >= self.config.min_training_samples:
                    self.xgboost_model.fit(train_df)
                else:
                    return None

            return self.xgboost_model.predict(df)
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")
            return None

    def _predict_transformer(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Get Transformer prediction"""
        try:
            if self.transformer_model is None:
                return None

            # Train on-the-fly if not fitted
            if not self.transformer_model.is_fitted:
                train_df = df.iloc[:-30] if len(df) > 130 else df.iloc[:-10]
                if len(train_df) >= self.config.min_training_samples:
                    self.transformer_model.fit(train_df)
                else:
                    return None

            return self.transformer_model.predict(df)
        except Exception as e:
            logger.warning(f"Transformer prediction failed: {e}")
            return None

    async def predict(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate ensemble prediction.

        Args:
            symbol: Stock symbol
            df: Historical OHLCV DataFrame

        Returns:
            Ensemble prediction with individual model outputs
        """
        predictions = {}
        prediction_values = []
        confidences = []
        active_weights = []

        # Run predictions in parallel where possible
        # LSTM is async, XGBoost and Transformer are sync

        # Run sync models in thread pool
        loop = asyncio.get_event_loop()

        xgb_future = loop.run_in_executor(
            self._executor,
            self._predict_xgboost,
            df
        )
        transformer_future = loop.run_in_executor(
            self._executor,
            self._predict_transformer,
            df
        )

        # Get LSTM prediction (async)
        lstm_pred = await self._predict_lstm(symbol, df)
        if lstm_pred and self.config.use_lstm:
            predictions["lstm"] = lstm_pred
            # Extract predicted return percentage
            if "change_pct" in lstm_pred:
                prediction_values.append(lstm_pred["change_pct"])
            elif "predicted_return" in lstm_pred:
                prediction_values.append(lstm_pred["predicted_return"])
            else:
                current = lstm_pred.get("current_price", 0)
                predicted = lstm_pred.get("predicted_price", 0)
                if current > 0:
                    prediction_values.append(((predicted - current) / current) * 100)

            confidences.append(lstm_pred.get("confidence", 0.5))
            active_weights.append(self.weights[0])

        # Get XGBoost prediction
        xgb_pred = await xgb_future
        if xgb_pred and self.config.use_xgboost:
            predictions["xgboost"] = xgb_pred
            prediction_values.append(xgb_pred.get("predicted_return", 0))
            confidences.append(xgb_pred.get("confidence", 0.5))
            active_weights.append(self.weights[1])

        # Get Transformer prediction
        transformer_pred = await transformer_future
        if transformer_pred and self.config.use_transformer:
            predictions["transformer"] = transformer_pred
            prediction_values.append(transformer_pred.get("predicted_return", 0))
            confidences.append(transformer_pred.get("confidence", 0.5))
            active_weights.append(self.weights[2])

        # Check if we have any predictions
        if not prediction_values:
            raise ValueError("All models failed to produce predictions")

        # Normalize weights
        weight_sum = sum(active_weights)
        normalized_weights = [w / weight_sum for w in active_weights]

        # Weighted average of predictions
        ensemble_return = sum(
            p * w for p, w in zip(prediction_values, normalized_weights)
        )

        # Weighted average of confidence
        ensemble_confidence = sum(
            c * w for c, w in zip(confidences, normalized_weights)
        )

        # Determine direction and signal strength
        direction = "UP" if ensemble_return > 0 else "DOWN"
        signal_strength = self._calculate_signal_strength(
            prediction_values,
            confidences,
            normalized_weights
        )

        # Current price
        current_price = float(df['Close'].iloc[-1])
        predicted_price = current_price * (1 + ensemble_return / 100)

        result = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "predicted_return_pct": round(ensemble_return, 4),
            "direction": direction,
            "confidence": round(ensemble_confidence, 3),
            "signal_strength": signal_strength,
            "models_used": list(predictions.keys()),
            "model_weights": dict(zip(predictions.keys(), normalized_weights)),
            "individual_predictions": predictions,
            "method": "Ensemble (LSTM + XGBoost + Transformer)"
        }

        # Store for weight adjustment
        self._record_prediction(result)

        return result

    def _calculate_signal_strength(
        self,
        predictions: List[float],
        confidences: List[float],
        weights: List[float]
    ) -> str:
        """
        Calculate signal strength based on model agreement.

        Returns:
            "STRONG", "MODERATE", or "WEAK"
        """
        if len(predictions) < 2:
            return "WEAK"

        # Check if all models agree on direction
        directions = [1 if p > 0 else -1 for p in predictions]
        agreement = all(d == directions[0] for d in directions)

        # Average confidence
        avg_confidence = sum(c * w for c, w in zip(confidences, weights))

        if agreement and avg_confidence >= self.config.high_confidence_threshold:
            return "STRONG"
        elif agreement or avg_confidence >= self.config.low_confidence_threshold:
            return "MODERATE"
        else:
            return "WEAK"

    def _record_prediction(self, prediction: Dict[str, Any]):
        """Record prediction for weight adjustment"""
        self.prediction_history.append({
            "timestamp": pd.Timestamp.now(),
            "prediction": prediction
        })

        # Keep only recent predictions
        max_history = self.config.weight_lookback * 2
        if len(self.prediction_history) > max_history:
            self.prediction_history = self.prediction_history[-max_history:]

    def update_weights_from_actuals(
        self,
        actuals: pd.Series
    ) -> Dict[str, float]:
        """
        Update model weights based on actual results.

        Args:
            actuals: Series of actual returns matching prediction dates

        Returns:
            Updated weights dictionary
        """
        if not self.config.use_dynamic_weights:
            return dict(zip(["lstm", "xgboost", "transformer"], self.weights))

        if len(self.prediction_history) < self.config.weight_lookback:
            return dict(zip(["lstm", "xgboost", "transformer"], self.weights))

        # Calculate accuracy for each model
        model_errors = {"lstm": [], "xgboost": [], "transformer": []}

        for record in self.prediction_history[-self.config.weight_lookback:]:
            pred = record["prediction"]
            ts = record["timestamp"]

            # Find matching actual
            if ts.date() in actuals.index:
                actual = actuals.loc[ts.date()]

                for model_name, model_pred in pred.get("individual_predictions", {}).items():
                    if model_name in model_errors:
                        pred_return = model_pred.get("predicted_return", 0)
                        error = abs(pred_return - actual)
                        model_errors[model_name].append(error)

        # Calculate inverse error weights (lower error = higher weight)
        new_weights = []
        for model_name in ["lstm", "xgboost", "transformer"]:
            errors = model_errors[model_name]
            if errors:
                avg_error = np.mean(errors)
                # Inverse weight: 1 / (error + epsilon)
                new_weights.append(1 / (avg_error + 0.01))
            else:
                # Use default weight if no data
                idx = ["lstm", "xgboost", "transformer"].index(model_name)
                new_weights.append(self.config.initial_weights[idx])

        # Normalize
        weight_sum = sum(new_weights)
        self.weights = [w / weight_sum for w in new_weights]

        return dict(zip(["lstm", "xgboost", "transformer"], self.weights))

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from models that support it.

        Returns:
            Dict with feature importance per model
        """
        importance = {}

        if self.xgboost_model and self.xgboost_model.is_fitted:
            importance["xgboost"] = self.xgboost_model.get_feature_importance()

        return importance

    async def fit_all_models(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fit all models on the provided data.

        Args:
            df: Training DataFrame with OHLCV data
            validation_split: Fraction for validation

        Returns:
            Training metrics for each model
        """
        results = {}

        # Fit XGBoost
        if self.config.use_xgboost and self.xgboost_model:
            try:
                xgb_metrics = self.xgboost_model.fit(
                    df,
                    validation_split=validation_split
                )
                results["xgboost"] = xgb_metrics
                logger.info(f"XGBoost fitted: val_rmse={xgb_metrics['val_rmse']:.4f}")
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}")
                results["xgboost"] = {"error": str(e)}

        # Fit Transformer
        if self.config.use_transformer and self.transformer_model:
            try:
                transformer_metrics = self.transformer_model.fit(
                    df,
                    validation_split=validation_split
                )
                results["transformer"] = transformer_metrics
                logger.info(
                    f"Transformer fitted: val_rmse={transformer_metrics['val_rmse']:.4f}"
                )
            except Exception as e:
                logger.error(f"Transformer training failed: {e}")
                results["transformer"] = {"error": str(e)}

        # LSTM trains on-the-fly, so we just note it's available
        if self.config.use_lstm and self.lstm_model:
            results["lstm"] = {"status": "on-demand training", "available": True}

        return results


# Singleton instance
_ensemble_instance: Optional[EnsemblePredictor] = None


def get_ensemble_predictor() -> EnsemblePredictor:
    """Get singleton instance of ensemble predictor"""
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = EnsemblePredictor()
    return _ensemble_instance


async def predict_with_ensemble(
    symbol: str,
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Convenience function for ensemble prediction.

    Args:
        symbol: Stock symbol
        df: Historical OHLCV data

    Returns:
        Ensemble prediction
    """
    predictor = get_ensemble_predictor()
    return await predictor.predict(symbol, df)
