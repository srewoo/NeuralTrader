"""
Transformer Model for Stock Price Prediction

A lightweight transformer encoder for time-series prediction.
Uses positional encoding and self-attention for sequence modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for Transformer model"""
    # Model architecture
    input_dim: int = 6  # OHLCV + returns
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # Sequence settings
    sequence_length: int = 60  # 60 days lookback

    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 5

    # Output
    prediction_horizon: int = 1  # Days ahead to predict


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, features)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class StockTransformer(nn.Module):
    """
    Transformer encoder for stock price prediction.

    Architecture:
    - Input projection
    - Positional encoding
    - Transformer encoder layers
    - Output projection for prediction
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.hidden_dim,
            max_len=config.sequence_length + 10,
            dropout=config.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # Output layers
        self.fc_out = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)  # Predict return
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, features)

        Returns:
            Predictions of shape (batch, 1)
        """
        # Project input
        x = self.input_projection(x)  # (batch, seq, hidden)

        # Transpose for transformer (seq, batch, hidden)
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Use last position for prediction
        x = x[-1]  # (batch, hidden)

        # Output projection
        return self.fc_out(x)


class TransformerPredictor:
    """
    Transformer-based stock price predictor.

    Features used:
    - Open, High, Low, Close, Volume (normalized)
    - Price returns
    """

    def __init__(self, config: Optional[TransformerConfig] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Run: pip install torch")

        self.config = config or TransformerConfig()
        self.model: Optional[StockTransformer] = None
        self.scaler = StandardScaler()
        self.price_scaler = StandardScaler()
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            Feature array of shape (n_samples, n_features)
        """
        features = pd.DataFrame(index=df.index)

        # Price features (normalized by close)
        features['open_ratio'] = df['Open'] / df['Close']
        features['high_ratio'] = df['High'] / df['Close']
        features['low_ratio'] = df['Low'] / df['Close']
        features['close_return'] = df['Close'].pct_change()

        # Volume (log-normalized)
        features['volume_log'] = np.log1p(df['Volume'])

        # Range
        features['hl_range'] = (df['High'] - df['Low']) / df['Close']

        return features.values

    def _create_sequences(
        self,
        data: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for transformer input.

        Args:
            data: Feature array
            targets: Target array

        Returns:
            Tuple of (X, y) sequences
        """
        seq_len = self.config.sequence_length
        X, y = [], []

        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(targets[i + seq_len])

        return np.array(X), np.array(y)

    def fit(
        self,
        df: pd.DataFrame,
        target_horizon: int = 1,
        validation_split: float = 0.2,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Train the transformer model.

        Args:
            df: OHLCV DataFrame
            target_horizon: Days ahead to predict
            validation_split: Fraction for validation
            verbose: Print training progress

        Returns:
            Training metrics
        """
        logger.info("Preparing data for transformer training")

        # Prepare features
        features = self._prepare_features(df)

        # Target: next-day return
        target = df['Close'].pct_change(target_horizon).shift(-target_horizon).values

        # Remove NaN rows
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        features = features[valid_mask]
        target = target[valid_mask]

        if len(features) < self.config.sequence_length + 100:
            raise ValueError(f"Insufficient data: {len(features)} samples")

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Create sequences
        X, y = self._create_sequences(features_scaled, target)

        if len(X) < 50:
            raise ValueError(f"Insufficient sequences: {len(X)}")

        # Time-series split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # Initialize model
        self.config.input_dim = features.shape[1]
        self.model = StockTransformer(self.config).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.config.epochs):
            self.model.train()

            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            epoch_loss = 0
            n_batches = 0

            for i in range(0, len(X_train_t), self.config.batch_size):
                batch_idx = indices[i:i + self.config.batch_size]
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
                val_losses.append(val_loss)

            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.6f}, Val Loss={val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        self.is_fitted = True

        # Calculate metrics
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train_t).cpu().numpy().flatten()
            val_pred = self.model(X_val_t).cpu().numpy().flatten()

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
            "epochs_trained": len(train_losses),
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "device": str(self.device)
        }

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make prediction for the next period.

        Args:
            df: Recent OHLCV data (at least sequence_length rows)

        Returns:
            Prediction with confidence
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if len(df) < self.config.sequence_length:
            raise ValueError(f"Need at least {self.config.sequence_length} rows")

        # Prepare features
        features = self._prepare_features(df)

        # Remove NaN
        valid_start = np.where(~np.isnan(features).any(axis=1))[0]
        if len(valid_start) == 0:
            raise ValueError("All rows contain NaN")

        features = features[valid_start[0]:]

        if len(features) < self.config.sequence_length:
            raise ValueError("Insufficient valid data after removing NaN")

        # Scale
        features_scaled = self.scaler.transform(features)

        # Get last sequence
        X = features_scaled[-self.config.sequence_length:]
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            predicted_return = float(self.model(X_tensor).cpu().numpy()[0, 0])

        # Current price
        current_price = df['Close'].iloc[-1]
        predicted_price = current_price * (1 + predicted_return)

        # Confidence based on prediction magnitude
        confidence = max(0.3, 1 - abs(predicted_return) * 10)
        confidence = min(confidence, 0.95)

        return {
            "predicted_return": round(predicted_return * 100, 4),
            "predicted_price": round(predicted_price, 2),
            "current_price": round(current_price, 2),
            "direction": "UP" if predicted_return > 0 else "DOWN",
            "confidence": round(confidence, 3),
            "model": "Transformer"
        }

    def get_attention_weights(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get attention weights for interpretability.
        (Requires model modification to expose attention)

        Returns:
            None (placeholder for future implementation)
        """
        # This would require modifying StockTransformer to return attention weights
        logger.warning("Attention weight extraction not yet implemented")
        return None
