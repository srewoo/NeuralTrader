
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

from .dataset import DataProcessor
from .model import PricePredictor
from .trainer import train_model

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def predict_next_price(
        self, 
        symbol: str, 
        df: pd.DataFrame, 
        lookback_days: int = 60
    ) -> Dict[str, Any]:
        """
        Train a quick model on recent history and predict next day.
        
        Args:
            symbol: Stock symbol
            df: Historical DataFrame with 'Close' column
            lookback_days: Sequence length for LSTM
            
        Returns:
            Dict with prediction and confidence
        """
        try:
            # 1. Prepare Data
            processor = DataProcessor(sequence_length=lookback_days)
            # Use last year of data for training to be fast but relevant
            train_df = df.iloc[-300:] if len(df) > 300 else df
            
            dataset, scaler = processor.prepare_data(train_df)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # 2. Initialize Model
            model = PricePredictor(input_dim=1, hidden_dim=32, num_layers=1)
            
            # 3. Train (On-the-fly) - keeping epochs low for responsiveness
            # In production, we would load pre-trained models
            train_model(model, loader, num_epochs=15, device=self.device)
            
            # 4. Predict
            model.eval()
            with torch.no_grad():
                # Prepare input sequence (last window)
                input_seq = processor.prepare_inference_data(train_df)
                input_seq = input_seq.to(self.device)
                
                prediction_scaled = model(input_seq)
                prediction = scaler.inverse_transform(prediction_scaled.cpu().numpy())
                
                predicted_price = float(prediction[0][0])
                current_price = float(df['Close'].iloc[-1])
                
                # Simple confidence calculation (based on recent volatility)
                # In real ML, this would use Monte Carlo dropout or ensemble
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
                    "method": "LSTM (On-the-fly Training)"
                }
                
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            raise e

# Global instance
_ml_service = None

def get_ml_service() -> MLService:
    global _ml_service
    if _ml_service is None:
        _ml_service = MLService()
    return _ml_service
