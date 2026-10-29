"""
Unit tests for data preprocessing and cleaning pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing.clean_data import (
    load_dataset,
    inspect_schema_and_quality,
    clean_dataset
)


@pytest.fixture
def sample_dirty_data():
    """Provides a synthetic dirty dataframe with duplicates, nulls, and anomalies."""
    return pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-2", "ORD-3", "ORD-4"],  # ORD-2 is duplicated
        "order_timestamp": ["2026-10-01 12:00:00", "2026-10-01 13:00:00", "2026-10-01 13:00:00", " 2026-10-01 14:00:00 ", "invalid-date"],
        "restaurant_id": ["REST-1", "REST-1", "REST-1", "REST-2", "REST-2"],
        "restaurant_type": ["Fast Food", None, None, "Cafe", "Fast Food"],  # Contains None
        "delivery_zone": ["Downtown", "Downtown", "Downtown", "Suburbs", "Downtown"],
        "order_amount": [250.0, 300.0, 300.0, -50.0, 400.0],  # -50 is an anomaly
        "item_count": [2, 3, 3, 0, 4],  # 0 items is an anomaly
        "preparation_time_min": [15.0, 20.0, 20.0, None, 25.0],  # Contains None
        "delivery_distance_km": [3.5, 4.0, 4.0, 5.0, None],  # Contains None
        "delivery_duration_min": [30.0, 40.0, 40.0, 50.0, 45.0],
        "order_status": ["Delivered", "Delivered", "Delivered", "Delivered", "Delivered"],
        "weather_condition": ["Clear", None, None, "Rainy", "Clear"],  # Contains None
        "traffic_density": ["Low", "Medium", "Medium", "High", "Low"],
        "is_delayed": [0, 0, 0, 1, 0]
    })


def test_inspect_schema_and_quality(sample_dirty_data):
    """Verify inspection audit accurately counts missing items and duplicates."""
    audit = inspect_schema_and_quality(sample_dirty_data)
    assert audit["total_records"] == 5
    assert audit["exact_duplicate_rows"] == 1
    assert audit["duplicate_order_ids"] == 1
    assert audit["total_missing_values"] > 0
    assert "order_id" in audit["columns"]


def test_clean_dataset_deduplication(sample_dirty_data):
    """Verify duplicates are dropped based on order_id."""
    cleaned_df, report = clean_dataset(sample_dirty_data)
    assert report["duplicates_removed"] == 1
    assert cleaned_df["order_id"].nunique() == len(cleaned_df)


def test_clean_dataset_anomaly_correction(sample_dirty_data):
    """Verify negative amounts and zero item counts are corrected."""
    cleaned_df, report = clean_dataset(sample_dirty_data)
    # Order amount should now be all positive
    assert (cleaned_df["order_amount"] > 0).all()
    # Item count should now be at least 1
    assert (cleaned_df["item_count"] >= 1).all()
    assert "order_amount" in report["anomalies_corrected"]
    assert "item_count" in report["anomalies_corrected"]


def test_clean_dataset_null_imputation(sample_dirty_data):
    """Verify missing categorical and numeric fields are imputed."""
    cleaned_df, report = clean_dataset(sample_dirty_data)
    # No nulls should remain in these columns
    assert cleaned_df["restaurant_type"].isnull().sum() == 0
    assert cleaned_df["weather_condition"].isnull().sum() == 0
    assert cleaned_df["preparation_time_min"].isnull().sum() == 0
    assert cleaned_df["delivery_distance_km"].isnull().sum() == 0


def test_clean_dataset_invalid_timestamp_handling(sample_dirty_data):
    """Verify unparseable timestamp record is safely handled."""
    cleaned_df, report = clean_dataset(sample_dirty_data)
    # The 'invalid-date' row should be removed
    assert report["invalid_timestamps_removed"] == 1
    assert pd.api.types.is_datetime64_any_dtype(cleaned_df["order_timestamp"])


def test_load_dataset_missing_file_error():
    """Verify load_dataset raises FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        load_dataset(Path("data/non_existent_folder/missing_file.csv"))
