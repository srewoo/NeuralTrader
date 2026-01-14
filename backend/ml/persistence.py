"""
Model Persistence Utilities

Handles saving and loading of pre-trained ML models for:
- LSTM (PyTorch state_dict + MinMaxScaler)
- XGBoost (joblib for model + StandardScaler + feature names)
- Transformer (PyTorch state_dict + StandardScaler + config)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import json

import torch
import joblib

logger = logging.getLogger(__name__)

# Base directory for saved models
MODELS_DIR = Path(__file__).parent.parent / "models"


class ModelPersistence:
    """Handles model save/load operations for all ML model types."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or MODELS_DIR
        self._ensure_directories()

    def _ensure_directories(self):
        """Create model directories if they don't exist."""
        for subdir in ["lstm", "xgboost", "transformer"]:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

    # --- LSTM (PyTorch) ---
    def save_lstm(
        self,
        model: torch.nn.Module,
        scaler: Any,
        symbol: str = "default",
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save LSTM model and scaler.

        Args:
            model: PyTorch LSTM model (PricePredictor)
            scaler: MinMaxScaler used for data normalization
            symbol: Symbol identifier (use 'default' for general model)
            metadata: Optional training metadata

        Returns:
            Path to saved model
        """
        model_path = self.base_dir / "lstm" / f"lstm_{symbol}.pt"
        scaler_path = self.base_dir / "lstm" / f"lstm_{symbol}_scaler.joblib"
        meta_path = self.base_dir / "lstm" / f"lstm_{symbol}_meta.json"

        # Save model state
        torch.save(model.state_dict(), model_path)

        # Save scaler
        joblib.dump(scaler, scaler_path)

        # Save metadata
        if metadata is None:
            metadata = {}
        metadata["saved_at"] = datetime.now().isoformat()
        metadata["model_type"] = "lstm"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"LSTM model saved: {model_path}")
        return model_path

    def load_lstm(
        self,
        model: torch.nn.Module,
        symbol: str = "default",
        device: str = "cpu"
    ) -> Tuple[bool, Any, Optional[Dict]]:
        """
        Load LSTM model weights and scaler.

        Args:
            model: Initialized PyTorch model to load weights into
            symbol: Symbol identifier
            device: Device to load model onto

        Returns:
            Tuple of (success, scaler, metadata)
        """
        model_path = self.base_dir / "lstm" / f"lstm_{symbol}.pt"
        scaler_path = self.base_dir / "lstm" / f"lstm_{symbol}_scaler.joblib"
        meta_path = self.base_dir / "lstm" / f"lstm_{symbol}_meta.json"

        # Fall back to default if symbol-specific not found
        if not model_path.exists() and symbol != "default":
            logger.info(f"No model for {symbol}, falling back to default")
            return self.load_lstm(model, "default", device)

        if not model_path.exists():
            logger.warning(f"LSTM model not found: {model_path}")
            return False, None, None

        try:
            # Load model weights
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()

            # Load scaler
            scaler = None
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)

            # Load metadata
            metadata = None
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)

            logger.info(f"LSTM model loaded: {model_path}")
            return True, scaler, metadata

        except Exception as e:
            logger.error(f"Failed to load LSTM: {e}")
            return False, None, None

    # --- XGBoost (joblib) ---
    def save_xgboost(
        self,
        model: Any,  # XGBRegressor
        scaler: Any,  # StandardScaler
        feature_names: list,
        symbol: str = "default",
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save XGBoost model with scaler and feature names.

        Args:
            model: Trained XGBRegressor
            scaler: StandardScaler for features
            feature_names: List of feature column names
            symbol: Symbol identifier
            metadata: Optional training metadata

        Returns:
            Path to saved model
        """
        model_path = self.base_dir / "xgboost" / f"xgboost_{symbol}.joblib"

        save_obj = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat()
        }

        joblib.dump(save_obj, model_path)
        logger.info(f"XGBoost model saved: {model_path}")
        return model_path

    def load_xgboost(self, symbol: str = "default") -> Optional[Dict]:
        """
        Load XGBoost model.

        Args:
            symbol: Symbol identifier

        Returns:
            Dict with model, scaler, feature_names, metadata or None
        """
        model_path = self.base_dir / "xgboost" / f"xgboost_{symbol}.joblib"

        # Fall back to default
        if not model_path.exists() and symbol != "default":
            logger.info(f"No XGBoost model for {symbol}, falling back to default")
            return self.load_xgboost("default")

        if not model_path.exists():
            logger.warning(f"XGBoost model not found: {model_path}")
            return None

        try:
            data = joblib.load(model_path)
            logger.info(f"XGBoost model loaded: {model_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load XGBoost: {e}")
            return None

    # --- Transformer (PyTorch) ---
    def save_transformer(
        self,
        model: torch.nn.Module,
        scaler: Any,
        config: Any,  # TransformerConfig
        symbol: str = "default",
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save Transformer model with config and scaler.

        Args:
            model: Trained StockTransformer
            scaler: StandardScaler for features
            config: TransformerConfig dataclass
            symbol: Symbol identifier
            metadata: Optional training metadata

        Returns:
            Path to saved model
        """
        model_path = self.base_dir / "transformer" / f"transformer_{symbol}.pt"
        scaler_path = self.base_dir / "transformer" / f"transformer_{symbol}_scaler.joblib"
        config_path = self.base_dir / "transformer" / f"transformer_{symbol}_config.json"

        # Save model state
        torch.save(model.state_dict(), model_path)

        # Save scaler
        joblib.dump(scaler, scaler_path)

        # Save config as dict
        config_dict = {
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "sequence_length": config.sequence_length,
            "metadata": metadata or {},
            "saved_at": datetime.now().isoformat()
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Transformer model saved: {model_path}")
        return model_path

    def load_transformer(
        self,
        symbol: str = "default",
        device: str = "cpu"
    ) -> Optional[Dict]:
        """
        Load Transformer model.

        Args:
            symbol: Symbol identifier
            device: Device to load onto

        Returns:
            Dict with state_dict, scaler, config or None
        """
        model_path = self.base_dir / "transformer" / f"transformer_{symbol}.pt"
        scaler_path = self.base_dir / "transformer" / f"transformer_{symbol}_scaler.joblib"
        config_path = self.base_dir / "transformer" / f"transformer_{symbol}_config.json"

        # Fall back to default
        if not model_path.exists() and symbol != "default":
            logger.info(f"No Transformer model for {symbol}, falling back to default")
            return self.load_transformer("default", device)

        if not model_path.exists():
            logger.warning(f"Transformer model not found: {model_path}")
            return None

        try:
            # Load config
            with open(config_path) as f:
                config_dict = json.load(f)

            # Load scaler
            scaler = joblib.load(scaler_path)

            # Load model state
            state_dict = torch.load(model_path, map_location=device, weights_only=True)

            logger.info(f"Transformer model loaded: {model_path}")
            return {
                "state_dict": state_dict,
                "scaler": scaler,
                "config": config_dict,
                "device": device
            }

        except Exception as e:
            logger.error(f"Failed to load Transformer: {e}")
            return None

    def list_available_models(self) -> Dict[str, list]:
        """
        List all available pre-trained models.

        Returns:
            Dict mapping model type to list of available symbols
        """
        models = {"lstm": [], "xgboost": [], "transformer": []}

        for model_type in models.keys():
            model_dir = self.base_dir / model_type
            if model_dir.exists():
                if model_type == "xgboost":
                    pattern = "xgboost_*.joblib"
                else:
                    pattern = f"{model_type}_*.pt"

                for f in model_dir.glob(pattern):
                    # Extract symbol from filename
                    stem = f.stem
                    if model_type == "lstm":
                        symbol = stem.replace("lstm_", "")
                    elif model_type == "xgboost":
                        symbol = stem.replace("xgboost_", "")
                    else:
                        symbol = stem.replace("transformer_", "")

                    # Skip scaler files
                    if "_scaler" not in symbol and "_meta" not in symbol and "_config" not in symbol:
                        models[model_type].append(symbol)

        return models

    def get_model_info(self, model_type: str, symbol: str = "default") -> Optional[Dict]:
        """
        Get metadata/info about a saved model.

        Args:
            model_type: 'lstm', 'xgboost', or 'transformer'
            symbol: Symbol identifier

        Returns:
            Metadata dict or None
        """
        if model_type == "lstm":
            meta_path = self.base_dir / "lstm" / f"lstm_{symbol}_meta.json"
        elif model_type == "xgboost":
            model_path = self.base_dir / "xgboost" / f"xgboost_{symbol}.joblib"
            if model_path.exists():
                data = joblib.load(model_path)
                return {
                    "saved_at": data.get("saved_at"),
                    "metadata": data.get("metadata", {}),
                    "n_features": len(data.get("feature_names", []))
                }
            return None
        elif model_type == "transformer":
            meta_path = self.base_dir / "transformer" / f"transformer_{symbol}_config.json"
        else:
            return None

        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return None


# Singleton instance
_persistence: Optional[ModelPersistence] = None


def get_model_persistence() -> ModelPersistence:
    """Get or create singleton ModelPersistence instance."""
    global _persistence
    if _persistence is None:
        _persistence = ModelPersistence()
    return _persistence
