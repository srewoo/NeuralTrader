"""
ML Training Configuration

Defines stocks for pre-training and model hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List

# NIFTY 50 stocks for pre-training (most liquid Indian stocks)
NIFTY_50_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "BAJFINANCE",
    "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "HCLTECH", "WIPRO", "TITAN", "ULTRACEMCO", "NESTLEIND",
    "SUNPHARMA", "ONGC", "NTPC", "POWERGRID", "M&M",
    "TATAMOTORS", "TECHM", "ADANIPORTS", "DIVISLAB", "BRITANNIA",
    "BAJAJFINSV", "TATASTEEL", "HINDALCO", "JSWSTEEL", "INDUSINDBK",
    "GRASIM", "CIPLA", "DRREDDY", "BPCL", "COALINDIA",
    "SBILIFE", "EICHERMOT", "APOLLOHOSP", "TATACONSUM", "HEROMOTOCO",
    "ADANIENT", "SHREECEM", "HDFCLIFE", "BAJAJ-AUTO", "UPL"
]

# Popular additional stocks
ADDITIONAL_STOCKS = [
    "ZOMATO", "PAYTM", "DELHIVERY", "NYKAA", "IRCTC",
    "TATAPOWER", "VEDL", "PFC", "RECLTD", "NHPC",
    "BIOCON", "LUPIN", "AUROPHARMA", "TORNTPHARM",
    "BANDHANBNK", "RBLBANK", "IDFC", "IDFCFIRSTB", "PNB",
]


@dataclass
class LSTMTrainingConfig:
    """LSTM training configuration."""
    input_dim: int = 1  # Single feature (Close price)
    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 60
    epochs: int = 50  # More epochs for pre-training (vs 15 on-the-fly)
    batch_size: int = 32
    learning_rate: float = 0.001
    train_days: int = 365 * 2  # 2 years of data


@dataclass
class XGBoostTrainingConfig:
    """XGBoost training configuration."""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    train_days: int = 365 * 3  # 3 years of data
    validation_split: float = 0.2
    min_samples: int = 100


@dataclass
class TransformerTrainingConfig:
    """Transformer training configuration."""
    input_dim: int = 6  # OHLCV features
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    sequence_length: int = 60
    epochs: int = 100  # More epochs for pre-training
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    train_days: int = 365 * 2


@dataclass
class PreTrainingConfig:
    """Overall pre-training configuration."""
    symbols: List[str] = field(default_factory=lambda: NIFTY_50_SYMBOLS)
    train_default_model: bool = True  # Train one "default" model on combined data
    train_symbol_specific: bool = False  # Train per-symbol models (optional, slower)
    lstm: LSTMTrainingConfig = field(default_factory=LSTMTrainingConfig)
    xgboost: XGBoostTrainingConfig = field(default_factory=XGBoostTrainingConfig)
    transformer: TransformerTrainingConfig = field(default_factory=TransformerTrainingConfig)


def get_training_symbols(count: int = 50, include_additional: bool = False) -> List[str]:
    """
    Get list of symbols for training.

    Args:
        count: Number of symbols to return
        include_additional: Include additional stocks beyond NIFTY 50

    Returns:
        List of stock symbols
    """
    symbols = NIFTY_50_SYMBOLS.copy()

    if include_additional:
        symbols.extend(ADDITIONAL_STOCKS)

    return symbols[:count]
