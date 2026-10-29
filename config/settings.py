"""
Configuration module for Real-Time Food Delivery Big Data Analytics System.
"""
from pathlib import Path

# Base Paths (Computed dynamically to avoid hardcoded absolute paths)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
DOCS_DIR = BASE_DIR / "docs"

# Standard Filenames
SYNTHETIC_ORDERS_FILE = SYNTHETIC_DATA_DIR / "synthetic_orders.csv"
SYNTHETIC_MARKET_BASKETS_FILE = SYNTHETIC_DATA_DIR / "synthetic_market_baskets.json"
SYNTHETIC_COORDINATES_FILE = SYNTHETIC_DATA_DIR / "synthetic_delivery_coordinates.csv"
RAW_ORDERS_FILE = RAW_DATA_DIR / "raw_orders.csv"
PROCESSED_ORDERS_FILE = PROCESSED_DATA_DIR / "cleaned_orders.csv"
DATA_QUALITY_REPORT_FILE = PROCESSED_DATA_DIR / "data_quality_report.json"
MODELS_DIR = PROCESSED_DATA_DIR / "models"
DELAY_CLASSIFIER_MODEL_FILE = MODELS_DIR / "delay_classifier.joblib"

# Pipeline Configurations
RANDOM_SEED = 42
DEFAULT_NUM_RECORDS = 2500

# Domain Metadata and Categorical Values (aligned with BDA_34_EXP1 case study)
DELIVERY_ZONES = [
    "Downtown",
    "North Suburbs",
    "South Suburbs",
    "East Tech Park",
    "West University Campus",
    "Central Waterfront"
]

RESTAURANT_TYPES = [
    "Fast Food",
    "Casual Dining",
    "Cafe",
    "Cloud Kitchen",
    "Bakery & Desserts",
    "Fine Dining"
]

ORDER_STATUSES = [
    "Delivered",
    "Delivered",
    "Delivered",
    "Delivered",
    "Delivered",
    "Cancelled",
    "In Transit"
]

WEATHER_CONDITIONS = [
    "Clear",
    "Rainy",
    "Windy",
    "Foggy",
    "Heavy Rain"
]

TRAFFIC_LEVELS = [
    "Low",
    "Medium",
    "High",
    "Jam"
]

# Numeric Validation Thresholds
NUMERIC_RANGES = {
    "order_amount": {"min": 50.0, "max": 10000.0},
    "item_count": {"min": 1, "max": 50},
    "preparation_time_min": {"min": 5.0, "max": 120.0},
    "delivery_distance_km": {"min": 0.1, "max": 40.0},
    "delivery_duration_min": {"min": 5.0, "max": 180.0}
}
