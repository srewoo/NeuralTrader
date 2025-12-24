"""
MongoDB Client for Celery Tasks (Synchronous)

This module provides synchronous MongoDB client for use in Celery background tasks.
The main server.py uses AsyncIOMotorClient for async FastAPI endpoints,
but Celery tasks need synchronous operations.
"""

import os
import logging
from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global MongoDB client instance
_mongo_client: Optional[MongoClient] = None
_database: Optional[Database] = None


def get_mongo_client() -> MongoClient:
    """
    Get or create synchronous MongoDB client for Celery tasks

    Returns:
        MongoClient: Synchronous MongoDB client instance
    """
    global _mongo_client

    if _mongo_client is None:
        mongo_url = os.getenv("MONGO_URL")
        if not mongo_url:
            raise ValueError("MONGO_URL environment variable not set")

        _mongo_client = MongoClient(mongo_url)
        logger.info("MongoDB synchronous client initialized for Celery tasks")

    return _mongo_client


def get_database(db_name: Optional[str] = None) -> Database:
    """
    Get MongoDB database instance

    Args:
        db_name: Database name (defaults to DB_NAME from env)

    Returns:
        Database: MongoDB database instance
    """
    global _database

    if _database is None:
        client = get_mongo_client()
        db_name = db_name or os.getenv("DB_NAME", "neuraltrader")
        _database = client[db_name]
        logger.info(f"MongoDB database '{db_name}' accessed")

    return _database


def close_mongo_client():
    """Close MongoDB client connection"""
    global _mongo_client, _database

    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        _database = None
        logger.info("MongoDB client closed")


# Monkey patch get_database method on MongoClient for compatibility
# This allows code like: mongo.get_database() to work
MongoClient.get_database = lambda self, name=None: self[name or os.getenv("DB_NAME", "neuraltrader")]
