"""
Unit tests for synthetic data generation module.
"""
import pytest
import pandas as pd
from src.data_generation.generate_orders import generate_synthetic_orders

REQUIRED_COLUMNS = [
    "order_id",
    "order_timestamp",
    "restaurant_id",
    "restaurant_type",
    "delivery_zone",
    "order_amount",
    "item_count",
    "preparation_time_min",
    "delivery_distance_km",
    "delivery_duration_min",
    "order_status",
    "weather_condition",
    "traffic_density",
    "is_delayed"
]


def test_generation_record_count_clean():
    """Verify exact record count generated when noise is disabled."""
    df = generate_synthetic_orders(num_records=150, seed=42, include_noise=False)
    assert len(df) == 150
    assert isinstance(df, pd.DataFrame)


def test_generation_required_columns_present():
    """Verify all 14 domain-specified columns are present."""
    df = generate_synthetic_orders(num_records=50, seed=42, include_noise=False)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Required column '{col}' is missing in generated dataset."


def test_generation_reproducibility():
    """Verify fixed seed produces identical output across calls."""
    df1 = generate_synthetic_orders(num_records=100, seed=99, include_noise=False)
    df2 = generate_synthetic_orders(num_records=100, seed=99, include_noise=False)
    pd.testing.assert_frame_equal(df1, df2)


def test_generation_value_ranges_clean():
    """Verify clean dataset numbers are positive and logical."""
    df = generate_synthetic_orders(num_records=100, seed=42, include_noise=False)
    assert (df["order_amount"] > 0).all()
    assert (df["item_count"] >= 1).all()
    assert (df["preparation_time_min"] > 0).all()
    assert (df["delivery_distance_km"] > 0).all()
    assert (df["delivery_duration_min"] > 0).all()
    assert set(df["is_delayed"].unique()).issubset({0, 1})


def test_generation_with_noise_injected():
    """Verify noise injection creates duplicates and null values as designed."""
    base_records = 200
    df_noisy = generate_synthetic_orders(num_records=base_records, seed=42, include_noise=True)
    
    # Duplicate records should make total rows greater than base_records
    assert len(df_noisy) > base_records
    # There should be missing values injected
    assert df_noisy.isnull().sum().sum() > 0
    # There should be duplicate order_ids
    assert df_noisy.duplicated(subset=["order_id"]).sum() > 0
