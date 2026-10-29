"""
NoSQL persistence package for Real-Time Food Delivery Big Data Analytics.
Provides MongoDB integration with transparent SQLite/JSON fallback for local demonstration.
"""
from src.nosql.mongo_manager import MongoManager, get_database_manager

__all__ = ["MongoManager", "get_database_manager"]
