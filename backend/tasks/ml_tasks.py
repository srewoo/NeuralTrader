"""
ML Model Training Celery Tasks

Handles periodic retraining of ML models.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=1)
def retrain_all_models(self, symbols_count: int = 30):
    """
    Retrain all ML models with latest data.

    Should run weekly during off-hours (e.g., Sunday 2 AM).

    Args:
        symbols_count: Number of NIFTY symbols to train on
    """
    try:
        from ml.pretrain import (
            train_lstm_default,
            train_xgboost_default,
            train_transformer_default
        )
        from ml.training_config import PreTrainingConfig, NIFTY_50_SYMBOLS

        logger.info(f"Starting model retraining with {symbols_count} symbols...")

        config = PreTrainingConfig()
        symbols = NIFTY_50_SYMBOLS[:symbols_count]

        results = []

        # Train LSTM
        try:
            result = train_lstm_default(config.lstm, symbols)
            results.append(result)
            logger.info("LSTM retraining complete")
        except Exception as e:
            logger.error(f"LSTM retraining failed: {e}")
            results.append({"model": "lstm", "status": "failed", "error": str(e)})

        # Train XGBoost
        try:
            result = train_xgboost_default(config.xgboost, symbols)
            results.append(result)
            logger.info("XGBoost retraining complete")
        except Exception as e:
            logger.error(f"XGBoost retraining failed: {e}")
            results.append({"model": "xgboost", "status": "failed", "error": str(e)})

        # Train Transformer
        try:
            result = train_transformer_default(config.transformer, symbols)
            results.append(result)
            logger.info("Transformer retraining complete")
        except Exception as e:
            logger.error(f"Transformer retraining failed: {e}")
            results.append({"model": "transformer", "status": "failed", "error": str(e)})

        # Reload models in ensemble after retraining
        try:
            from ml.inference import reload_ml_service
            from ml.ensemble import get_ensemble_predictor

            # Reload ML service (LSTM)
            reload_ml_service()

            # Force reload of ensemble by creating new instance
            # The singleton will be recreated on next access
            import ml.ensemble as ensemble_module
            ensemble_module._ensemble_instance = None

            logger.info("Models reloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to reload models: {e}")

        # Store retraining results in MongoDB
        try:
            from database.mongo_client import get_mongo_client
            mongo = get_mongo_client()
            db = mongo.get_database()

            db.model_training_history.insert_one({
                "trained_at": datetime.now(),
                "symbols_count": symbols_count,
                "results": results,
                "task_id": self.request.id
            })
        except Exception as e:
            logger.warning(f"Failed to store training history: {e}")

        return {
            "status": "success",
            "results": results,
            "trained_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise self.retry(exc=e, countdown=3600)  # Retry in 1 hour


@celery_app.task(bind=True)
def retrain_single_model(self, model_type: str, symbols_count: int = 30):
    """
    Retrain a single model type.

    Args:
        model_type: 'lstm', 'xgboost', or 'transformer'
        symbols_count: Number of symbols to train on
    """
    try:
        from ml.training_config import PreTrainingConfig, NIFTY_50_SYMBOLS

        config = PreTrainingConfig()
        symbols = NIFTY_50_SYMBOLS[:symbols_count]

        logger.info(f"Retraining {model_type.upper()} model with {symbols_count} symbols...")

        if model_type == "lstm":
            from ml.pretrain import train_lstm_default
            result = train_lstm_default(config.lstm, symbols)
        elif model_type == "xgboost":
            from ml.pretrain import train_xgboost_default
            result = train_xgboost_default(config.xgboost, symbols)
        elif model_type == "transformer":
            from ml.pretrain import train_transformer_default
            result = train_transformer_default(config.transformer, symbols)
        else:
            return {"status": "error", "message": f"Unknown model type: {model_type}"}

        logger.info(f"{model_type.upper()} retraining complete")
        return result

    except Exception as e:
        logger.error(f"{model_type} retraining failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True)
def check_model_freshness(self):
    """
    Check if models need retraining based on age.

    Triggers retraining if models are older than threshold (7 days).
    Runs daily at 6 AM.
    """
    try:
        from ml.persistence import get_model_persistence
        from datetime import timedelta
        import json

        persistence = get_model_persistence()
        models = persistence.list_available_models()

        stale_models = []
        max_age = timedelta(days=7)  # Retrain if older than 7 days

        for model_type, symbols in models.items():
            if "default" in symbols:
                # Check metadata for training date
                info = persistence.get_model_info(model_type, "default")
                if info:
                    saved_at = info.get("saved_at")
                    if saved_at:
                        try:
                            saved_date = datetime.fromisoformat(saved_at)
                            age = datetime.now() - saved_date
                            if age > max_age:
                                stale_models.append(model_type)
                                logger.info(f"{model_type} model is {age.days} days old (stale)")
                        except (ValueError, TypeError):
                            pass

        if stale_models:
            logger.info(f"Stale models detected: {stale_models}, triggering retraining")
            retrain_all_models.delay(30)
            return {"status": "retraining_triggered", "stale_models": stale_models}

        logger.info("All models are fresh")
        return {"status": "models_fresh", "checked_at": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Model freshness check failed: {e}")
        return {"status": "error", "message": str(e)}


@celery_app.task(bind=True)
def get_model_status(self):
    """
    Get status of all pre-trained models.

    Useful for monitoring and debugging.
    """
    try:
        from ml.persistence import get_model_persistence

        persistence = get_model_persistence()
        available = persistence.list_available_models()

        status = {}
        for model_type, symbols in available.items():
            if "default" in symbols:
                info = persistence.get_model_info(model_type, "default")
                status[model_type] = {
                    "available": True,
                    "info": info
                }
            else:
                status[model_type] = {
                    "available": False,
                    "info": None
                }

        return {
            "status": "success",
            "models": status,
            "checked_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        return {"status": "error", "message": str(e)}
