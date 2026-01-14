"""
ML Inference Service with Pre-trained Model Support

Provides LSTM-based price predictions with:
- Pre-trained model loading at initialization
- Fallback to on-the-fly training if weights not available
- Model persistence for saving newly trained models
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

from .dataset import DataProcessor, StockDataset
from .model import PricePredictor
from .trainer import train_model

logger = logging.getLogger(__name__)


class MLService:
    """
    ML Service with pre-trained model support.

    Attempts to load pre-trained LSTM weights on initialization.
    Falls back to on-the-fly training if no weights available.
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[PricePredictor] = None
        self.scaler = None
        self.is_pretrained = False
        self.sequence_length = 60

        # Try to load pre-trained model at initialization
        self._load_pretrained_model()

    def _load_pretrained_model(self) -> bool:
        """Attempt to load pre-trained LSTM model."""
        try:
            from .persistence import get_model_persistence

            persistence = get_model_persistence()

            # Initialize model architecture
            self.model = PricePredictor(
                input_dim=1,
                hidden_dim=32,
                num_layers=2
            )

            success, scaler, metadata = persistence.load_lstm(
                self.model, "default", str(self.device)
            )

            if success:
                self.scaler = scaler
                self.is_pretrained = True
                logger.info("Loaded pre-trained LSTM model")
                return True
            else:
                self.model = None
                logger.info("No pre-trained LSTM model found, will train on-the-fly")
                return False

        except Exception as e:
            logger.warning(f"Failed to load pre-trained LSTM: {e}")
            self.model = None
            return False

    async def predict_next_price(
        self,
        symbol: str,
        df: pd.DataFrame,
        lookback_days: int = 60
    ) -> Dict[str, Any]:
        """
        Predict next day price.

        Uses pre-trained model if available, otherwise trains on-the-fly.

        Args:
            symbol: Stock symbol
            df: Historical DataFrame with 'Close' column
            lookback_days: Sequence length for LSTM

        Returns:
            Dict with prediction and confidence
        """
        try:
            # Use last 300 days of data
            train_df = df.iloc[-300:] if len(df) > 300 else df

            if self.is_pretrained and self.model is not None and self.scaler is not None:
                # Use pre-trained model - inference only
                return await self._predict_with_pretrained(symbol, train_df, lookback_days)
            else:
                # Fall back to on-the-fly training
                return await self._predict_with_training(symbol, train_df, lookback_days)

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            raise e

    async def _predict_with_pretrained(
        self,
        symbol: str,
        df: pd.DataFrame,
        lookback_days: int
    ) -> Dict[str, Any]:
        """Make prediction using pre-trained model (fast path)."""
        self.model.eval()

        # Prepare the inference sequence using the stored scaler
        data = df[['Close']].values
        scaled_data = self.scaler.transform(data)

        last_sequence = scaled_data[-lookback_days:]
        if len(last_sequence) < lookback_days:
            raise ValueError(f"Insufficient data: need {lookback_days} days, got {len(last_sequence)}")

        input_seq = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction_scaled = self.model(input_seq)
            prediction = self.scaler.inverse_transform(prediction_scaled.cpu().numpy())

        predicted_price = float(prediction[0][0])
        current_price = float(df['Close'].iloc[-1])

        # Calculate confidence based on volatility
        volatility = df['Close'].pct_change().std()
        confidence_interval = [
            round(predicted_price * (1 - volatility), 2),
            round(predicted_price * (1 + volatility), 2)
        ]

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "change_pct": round(((predicted_price - current_price) / current_price) * 100, 2),
            "confidence_interval": confidence_interval,
            "method": "LSTM (Pre-trained)",
            "is_pretrained": True
        }

    async def _predict_with_training(
        self,
        symbol: str,
        df: pd.DataFrame,
        lookback_days: int
    ) -> Dict[str, Any]:
        """Make prediction with on-the-fly training (slow path)."""
        processor = DataProcessor(sequence_length=lookback_days)
        dataset, scaler = processor.prepare_data(df)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize fresh model
        model = PricePredictor(input_dim=1, hidden_dim=32, num_layers=1)

        # Train (keeping epochs low for responsiveness)
        train_model(model, loader, num_epochs=15, device=self.device)

        # Predict
        model.eval()
        with torch.no_grad():
            input_seq = processor.prepare_inference_data(df)
            input_seq = input_seq.to(self.device)
            prediction_scaled = model(input_seq)
            prediction = scaler.inverse_transform(prediction_scaled.cpu().numpy())

        predicted_price = float(prediction[0][0])
        current_price = float(df['Close'].iloc[-1])

        volatility = df['Close'].pct_change().std()
        confidence_interval = [
            round(predicted_price * (1 - volatility), 2),
            round(predicted_price * (1 + volatility), 2)
        ]

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "change_pct": round(((predicted_price - current_price) / current_price) * 100, 2),
            "confidence_interval": confidence_interval,
            "method": "LSTM (On-the-fly Training)",
            "is_pretrained": False
        }

    def save_model(self, symbol: str = "default", metadata: Optional[Dict] = None) -> bool:
        """
        Save current model (if trained).

        Args:
            symbol: Symbol identifier
            metadata: Optional training metadata

        Returns:
            True if saved successfully
        """
        if self.model is None or self.scaler is None:
            logger.warning("No model to save")
            return False

        try:
            from .persistence import get_model_persistence

            persistence = get_model_persistence()
            persistence.save_lstm(self.model, self.scaler, symbol, metadata)
            return True

        except Exception as e:
            logger.error(f"Failed to save LSTM model: {e}")
            return False


# Global instance
_ml_service: Optional[MLService] = None


def get_ml_service() -> MLService:
    """Get or create ML service singleton."""
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service


def reload_ml_service() -> MLService:
    """
    Force reload of ML service.

    Useful after retraining to pick up new weights.

    Returns:
        New MLService instance
    """
    global _ml_service
    _ml_service = MLService()
    return _ml_service
