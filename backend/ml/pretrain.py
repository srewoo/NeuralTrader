#!/usr/bin/env python
"""
Model Pre-Training Script

Trains and saves ML models on NIFTY 50 stocks for fast inference.

Usage:
    python -m ml.pretrain --all
    python -m ml.pretrain --lstm
    python -m ml.pretrain --xgboost
    python -m ml.pretrain --transformer
    python -m ml.pretrain --all --symbols 20
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd
import numpy as np

# Add backend to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import torch
from torch.utils.data import DataLoader

from ml.persistence import get_model_persistence
from ml.training_config import (
    PreTrainingConfig, NIFTY_50_SYMBOLS,
    LSTMTrainingConfig, XGBoostTrainingConfig, TransformerTrainingConfig
)
from ml.model import PricePredictor
from ml.dataset import DataProcessor, StockDataset
from ml.trainer import train_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_stock_data(symbol: str, days: int = 730) -> pd.DataFrame:
    """
    Fetch historical data for a single symbol.

    Args:
        symbol: Stock symbol (without .NS suffix)
        days: Number of days of history

    Returns:
        DataFrame with OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Try NSE first, then BSE
    for suffix in [".NS", ".BO"]:
        try:
            yf_symbol = f"{symbol}{suffix}"
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date)

            if not df.empty and len(df) >= 100:
                df['Symbol'] = symbol
                return df

        except Exception as e:
            logger.debug(f"Failed to fetch {yf_symbol}: {e}")

    return pd.DataFrame()


def fetch_training_data(symbols: List[str], days: int = 730) -> pd.DataFrame:
    """
    Fetch historical data for multiple symbols and combine.

    Args:
        symbols: List of stock symbols
        days: Number of days of history

    Returns:
        Combined DataFrame
    """
    all_data = []

    for i, symbol in enumerate(symbols):
        logger.info(f"Fetching data for {symbol} ({i+1}/{len(symbols)})...")

        df = fetch_stock_data(symbol, days)

        if df.empty:
            logger.warning(f"No data for {symbol}, skipping")
            continue

        all_data.append(df)
        logger.info(f"  Got {len(df)} rows for {symbol}")

    if not all_data:
        raise ValueError("No data fetched for any symbol")

    combined = pd.concat(all_data)
    logger.info(f"Combined dataset: {len(combined)} total rows from {len(all_data)} symbols")
    return combined


def train_lstm_default(config: LSTMTrainingConfig, symbols: List[str]) -> Dict[str, Any]:
    """
    Train a default LSTM model on combined data.

    Args:
        config: LSTM training configuration
        symbols: List of symbols to train on

    Returns:
        Training result dict
    """
    logger.info("=" * 50)
    logger.info("Training LSTM Model")
    logger.info("=" * 50)

    persistence = get_model_persistence()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Fetch combined data
    combined_df = fetch_training_data(symbols, config.train_days)

    # Process each symbol's data to create sequences
    all_sequences = []
    all_targets = []
    processor = DataProcessor(sequence_length=config.sequence_length)
    last_scaler = None

    for symbol in symbols:
        symbol_df = combined_df[combined_df['Symbol'] == symbol].copy()
        if len(symbol_df) < config.sequence_length + 10:
            continue

        try:
            dataset, scaler = processor.prepare_data(symbol_df)
            last_scaler = scaler

            for i in range(len(dataset)):
                seq, target = dataset[i]
                all_sequences.append(seq.numpy())
                all_targets.append(target.numpy())

        except Exception as e:
            logger.warning(f"Failed to process {symbol} for LSTM: {e}")

    if not all_sequences:
        raise ValueError("No sequences generated for LSTM training")

    logger.info(f"Created {len(all_sequences)} training sequences")

    # Create combined dataset
    combined_dataset = StockDataset(np.array(all_sequences), np.array(all_targets))
    loader = DataLoader(combined_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize and train model
    model = PricePredictor(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    )

    logger.info(f"Training LSTM for {config.epochs} epochs...")
    history = train_model(
        model, loader,
        num_epochs=config.epochs,
        learning_rate=config.learning_rate,
        device=str(device)
    )

    # Save model
    metadata = {
        "symbols_trained": symbols,
        "num_sequences": len(all_sequences),
        "final_loss": history['loss'][-1] if history.get('loss') else None,
        "epochs": config.epochs,
        "config": {
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "sequence_length": config.sequence_length
        }
    }

    persistence.save_lstm(model, last_scaler, "default", metadata)

    logger.info(f"LSTM training complete. Final loss: {history['loss'][-1]:.6f}")
    return {"model": "lstm", "status": "success", "metadata": metadata}


def train_xgboost_default(config: XGBoostTrainingConfig, symbols: List[str]) -> Dict[str, Any]:
    """
    Train a default XGBoost model on combined data.

    Args:
        config: XGBoost training configuration
        symbols: List of symbols to train on

    Returns:
        Training result dict
    """
    logger.info("=" * 50)
    logger.info("Training XGBoost Model")
    logger.info("=" * 50)

    from ml.xgboost_model import XGBoostPredictor, XGBoostConfig

    # Fetch combined data
    combined_df = fetch_training_data(symbols, config.train_days)

    # Create predictor with config
    xgb_config = XGBoostConfig(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate
    )
    predictor = XGBoostPredictor(xgb_config)

    # Prepare features for each symbol
    all_features = []
    all_targets = []

    for symbol in symbols:
        symbol_df = combined_df[combined_df['Symbol'] == symbol].copy()
        if len(symbol_df) < 150:
            continue

        try:
            X, y, _ = predictor._prepare_data(symbol_df)
            all_features.append(X)
            all_targets.append(y)
        except Exception as e:
            logger.warning(f"Failed to process {symbol} for XGBoost: {e}")

    if not all_features:
        raise ValueError("No features generated for XGBoost training")

    # Combine all data
    X_combined = np.vstack(all_features)
    y_combined = np.concatenate(all_targets)

    logger.info(f"XGBoost training data: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")

    # Split for validation
    split_idx = int(len(X_combined) * (1 - config.validation_split))
    X_train, X_val = X_combined[:split_idx], X_combined[split_idx:]
    y_train, y_val = y_combined[:split_idx], y_combined[split_idx:]

    # Scale features
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_val_scaled = predictor.scaler.transform(X_val)

    # Train XGBoost
    import xgboost as xgb
    predictor.model = xgb.XGBRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        random_state=42,
        n_jobs=-1
    )

    logger.info(f"Training XGBoost with {config.n_estimators} trees...")
    predictor.model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=True
    )
    predictor.is_fitted = True

    # Calculate metrics
    val_pred = predictor.model.predict(X_val_scaled)
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    val_direction_acc = np.mean(np.sign(val_pred) == np.sign(y_val))

    # Save model
    metadata = {
        "symbols_trained": symbols,
        "num_samples": len(X_combined),
        "val_rmse": float(val_rmse),
        "val_direction_accuracy": float(val_direction_acc),
        "config": {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "learning_rate": config.learning_rate
        }
    }

    predictor.save_trained("default", metadata)

    logger.info(f"XGBoost training complete. Val RMSE: {val_rmse:.6f}, Direction Acc: {val_direction_acc:.2%}")
    return {"model": "xgboost", "status": "success", "metadata": metadata}


def train_transformer_default(config: TransformerTrainingConfig, symbols: List[str]) -> Dict[str, Any]:
    """
    Train a default Transformer model on combined data.

    Args:
        config: Transformer training configuration
        symbols: List of symbols to train on

    Returns:
        Training result dict
    """
    logger.info("=" * 50)
    logger.info("Training Transformer Model")
    logger.info("=" * 50)

    from ml.transformer_model import TransformerPredictor, TransformerConfig, StockTransformer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Fetch combined data
    combined_df = fetch_training_data(symbols, config.train_days)

    # Create transformer config
    transformer_config = TransformerConfig(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
        sequence_length=config.sequence_length,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate
    )
    predictor = TransformerPredictor(transformer_config)

    # Prepare sequences for each symbol
    all_X = []
    all_y = []

    # Use subset of symbols for transformer (training is slow)
    sample_symbols = symbols[:min(20, len(symbols))]
    logger.info(f"Using {len(sample_symbols)} symbols for Transformer training")

    for symbol in sample_symbols:
        symbol_df = combined_df[combined_df['Symbol'] == symbol].copy()
        if len(symbol_df) < config.sequence_length + 50:
            continue

        try:
            features = predictor._prepare_features(symbol_df)
            target = symbol_df['Close'].pct_change(1).shift(-1).values

            # Remove NaN
            valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
            features = features[valid_mask]
            target = target[valid_mask]

            if len(features) < config.sequence_length + 10:
                continue

            # Scale (fit scaler on first symbol, transform on rest)
            if len(all_X) == 0:
                features_scaled = predictor.scaler.fit_transform(features)
            else:
                features_scaled = predictor.scaler.transform(features)

            X, y = predictor._create_sequences(features_scaled, target)
            all_X.append(X)
            all_y.append(y)

        except Exception as e:
            logger.warning(f"Failed to process {symbol} for Transformer: {e}")

    if not all_X:
        raise ValueError("No sequences generated for Transformer training")

    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)

    logger.info(f"Transformer training data: {X_combined.shape[0]} sequences")

    # Split and prepare tensors
    split_idx = int(len(X_combined) * 0.8)
    X_train, X_val = X_combined[:split_idx], X_combined[split_idx:]
    y_train, y_val = y_combined[:split_idx], y_combined[split_idx:]

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

    # Initialize model
    transformer_config.input_dim = X_combined.shape[2]
    predictor.config = transformer_config
    predictor.model = StockTransformer(transformer_config).to(device)

    # Training setup
    import torch.nn as nn
    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    logger.info(f"Training Transformer for {config.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        predictor.model.train()
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(X_train_t), config.batch_size):
            batch_idx = indices[i:i + config.batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]

            optimizer.zero_grad()
            predictions = predictor.model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        predictor.model.eval()
        with torch.no_grad():
            val_pred = predictor.model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={epoch_loss/n_batches:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    predictor.is_fitted = True

    # Calculate final metrics
    predictor.model.eval()
    with torch.no_grad():
        val_pred = predictor.model(X_val_t).cpu().numpy().flatten()

    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    val_direction_acc = np.mean(np.sign(val_pred) == np.sign(y_val))

    # Save model
    metadata = {
        "symbols_trained": sample_symbols,
        "num_sequences": len(X_combined),
        "val_rmse": float(val_rmse),
        "val_direction_accuracy": float(val_direction_acc),
        "epochs_trained": epoch + 1,
        "config": {
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "sequence_length": config.sequence_length
        }
    }

    predictor.save_trained("default", metadata)

    logger.info(f"Transformer training complete. Val RMSE: {val_rmse:.6f}, Direction Acc: {val_direction_acc:.2%}")
    return {"model": "transformer", "status": "success", "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description="Pre-train ML models for NeuralTrader")
    parser.add_argument("--all", action="store_true", help="Train all models")
    parser.add_argument("--lstm", action="store_true", help="Train LSTM model")
    parser.add_argument("--xgboost", action="store_true", help="Train XGBoost model")
    parser.add_argument("--transformer", action="store_true", help="Train Transformer model")
    parser.add_argument("--symbols", type=int, default=30, help="Number of symbols to use (default: 30)")

    args = parser.parse_args()

    # Default to all if no specific model selected
    if not any([args.all, args.lstm, args.xgboost, args.transformer]):
        args.all = True

    config = PreTrainingConfig()
    symbols = NIFTY_50_SYMBOLS[:args.symbols]

    logger.info("=" * 60)
    logger.info("NeuralTrader Model Pre-Training")
    logger.info("=" * 60)
    logger.info(f"Training on {len(symbols)} symbols: {', '.join(symbols[:5])}...")

    results = []

    if args.all or args.lstm:
        try:
            result = train_lstm_default(config.lstm, symbols)
            results.append(result)
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            results.append({"model": "lstm", "status": "failed", "error": str(e)})

    if args.all or args.xgboost:
        try:
            result = train_xgboost_default(config.xgboost, symbols)
            results.append(result)
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            results.append({"model": "xgboost", "status": "failed", "error": str(e)})

    if args.all or args.transformer:
        try:
            result = train_transformer_default(config.transformer, symbols)
            results.append(result)
        except Exception as e:
            logger.error(f"Transformer training failed: {e}")
            results.append({"model": "transformer", "status": "failed", "error": str(e)})

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pre-training Complete!")
    logger.info("=" * 60)
    for r in results:
        status = "SUCCESS" if r["status"] == "success" else "FAILED"
        logger.info(f"  {r['model'].upper()}: {status}")
        if r["status"] == "success" and "metadata" in r:
            meta = r["metadata"]
            if "val_rmse" in meta:
                logger.info(f"    Val RMSE: {meta['val_rmse']:.6f}")
            if "val_direction_accuracy" in meta:
                logger.info(f"    Direction Accuracy: {meta['val_direction_accuracy']:.2%}")

    # Check saved models
    persistence = get_model_persistence()
    available = persistence.list_available_models()
    logger.info("")
    logger.info("Available pre-trained models:")
    for model_type, symbols_list in available.items():
        if symbols_list:
            logger.info(f"  {model_type}: {', '.join(symbols_list)}")

    return results


if __name__ == "__main__":
    main()
