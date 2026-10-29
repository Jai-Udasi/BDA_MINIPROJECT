"""
Unit tests for Phase 2 Exploratory Data Analytics, KPI formulations,
deterministic math assertions, and filter mechanisms.
"""
import pytest
import pandas as pd
import numpy as np

from src.analytics.exploratory_analysis import (
    compute_executive_kpis,
    get_hourly_demand,
    get_weekday_demand,
    get_daily_trend,
    get_zone_performance,
    get_restaurant_performance,
    detect_numeric_outliers,
    inspect_suspicious_records,
    apply_dashboard_filters
)
from src.analytics.spark_compat import check_spark_environment


@pytest.fixture
def deterministic_sample_orders():
    """
    Deterministic dataset with known ground-truth metrics for exact cross-validation:
    Total Orders: 6
    Completed: 4 (ORD-1, ORD-2, ORD-4, ORD-6)
    Cancelled: 1 (ORD-3)
    In Transit: 1 (ORD-5)
    Non-cancelled Revenue: 200 + 300 + 450 + 100 + 900 = 1950.0
    Completed Durations: 25.0, 50.0, 60.0, 45.0 -> Mean: 45.0, Median: 47.5
    Delayed: ORD-2 (50 > 45), ORD-4 (60 > 50) -> 2 out of 4 -> 50.0%
    """
    return pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4", "ORD-5", "ORD-6"],
        "order_timestamp": [
            "2026-10-01 12:00:00",
            "2026-10-01 13:00:00",
            "2026-10-01 19:00:00",
            "2026-10-02 20:00:00",
            "2026-10-02 21:00:00",
            "2026-10-03 14:00:00"
        ],
        "restaurant_id": ["REST-1", "REST-1", "REST-2", "REST-2", "REST-1", "REST-3"],
        "restaurant_type": ["Fast Food", "Fast Food", "Cafe", "Cafe", "Fast Food", "Fine Dining"],
        "delivery_zone": ["Downtown", "Downtown", "Suburbs", "Suburbs", "Downtown", "Waterfront"],
        "order_amount": [200.0, 300.0, 150.0, 450.0, 100.0, 900.0],
        "item_count": [2, 3, 1, 4, 1, 5],
        "preparation_time_min": [15.0, 20.0, 10.0, 25.0, 10.0, 35.0],
        "delivery_distance_km": [2.0, 4.0, 3.0, 6.0, 1.0, 5.0],
        "delivery_duration_min": [25.0, 50.0, np.nan, 60.0, np.nan, 45.0],
        "order_status": ["Delivered", "Delivered", "Cancelled", "Delivered", "In Transit", "Delivered"],
        "weather_condition": ["Clear", "Rainy", "Clear", "Clear", "Windy", "Clear"],
        "traffic_density": ["Low", "High", "Low", "Medium", "Low", "Medium"],
        "is_delayed": [0, 1, 0, 1, 0, 0]
    })


def test_deterministic_kpis_accuracy(deterministic_sample_orders):
    """Verify KPI computations strictly match deterministic ground-truth values."""
    kpis = compute_executive_kpis(deterministic_sample_orders)
    
    assert kpis["total_unique_orders"] == 6
    assert kpis["completed_orders"] == 4
    assert kpis["cancelled_orders"] == 1
    assert kpis["in_transit_orders"] == 1
    assert kpis["completion_rate_pct"] == round(4 / 6 * 100.0, 2)
    assert kpis["cancellation_rate_pct"] == round(1 / 6 * 100.0, 2)
    assert kpis["total_revenue"] == 1950.0
    assert kpis["avg_delivery_duration_min"] == 45.0
    assert kpis["median_delivery_duration_min"] == 47.5
    assert kpis["eligible_delay_orders_count"] == 4
    assert kpis["delayed_orders_count"] == 2
    assert kpis["delay_rate_pct"] == 50.0
    assert kpis["unique_restaurants"] == 3
    assert kpis["unique_zones"] == 3


def test_kpi_computation_on_empty_dataset():
    """Verify KPI calculations do not fail on empty dataframe."""
    empty_df = pd.DataFrame()
    kpis = compute_executive_kpis(empty_df)
    assert kpis["total_unique_orders"] == 0
    assert kpis["total_revenue"] == 0.0
    assert kpis["avg_delivery_duration_min"] == 0.0
    assert kpis["delay_rate_pct"] == 0.0


def test_kpi_missing_optional_columns():
    """Verify KPI calculations handle data missing optional columns."""
    minimal_df = pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2"],
        "order_amount": [150.0, 250.0]
    })
    kpis = compute_executive_kpis(minimal_df)
    assert kpis["total_unique_orders"] == 2
    assert kpis["total_revenue"] == 400.0
    assert kpis["avg_delivery_duration_min"] == 0.0


def test_hourly_demand_aggregation(deterministic_sample_orders):
    """Verify hourly demand returns 24 hours with exact counts."""
    hourly = get_hourly_demand(deterministic_sample_orders)
    assert len(hourly) == 24
    assert hourly.loc[hourly["hour"] == 12, "order_count"].iloc[0] == 1
    assert hourly.loc[hourly["hour"] == 13, "order_count"].iloc[0] == 1
    assert hourly.loc[hourly["hour"] == 0, "order_count"].iloc[0] == 0


def test_weekday_demand_aggregation(deterministic_sample_orders):
    """Verify weekday demand correctly identifies days and weekend flags."""
    weekday = get_weekday_demand(deterministic_sample_orders)
    assert not weekday.empty
    assert "weekday_name" in weekday.columns
    assert "is_weekend" in weekday.columns


def test_zone_performance_metrics(deterministic_sample_orders):
    """Verify zone performance computes sample sizes and averages correctly."""
    zone_perf = get_zone_performance(deterministic_sample_orders)
    assert len(zone_perf) == 3
    
    # Downtown has ORD-1, ORD-2, ORD-5 (3 orders, 2 delivered)
    downtown = zone_perf.loc[zone_perf["delivery_zone"] == "Downtown"].iloc[0]
    assert downtown["sample_size_orders"] == 3
    assert downtown["completed_orders"] == 2
    # Completed durations for Downtown: 25.0 and 50.0 -> avg = 37.5
    assert downtown["avg_duration_min"] == 37.5


def test_restaurant_performance_ranking(deterministic_sample_orders):
    """Verify restaurant ranking groups by restaurant_id and sorts by order volume."""
    rest_perf = get_restaurant_performance(deterministic_sample_orders, top_n=5)
    assert len(rest_perf) == 3
    # REST-1 has 3 orders (highest)
    assert rest_perf.iloc[0]["restaurant_id"] == "REST-1"
    assert rest_perf.iloc[0]["sample_size_orders"] == 3


def test_outlier_detection_iqr():
    """Verify Tukey's IQR detects extreme outliers without crashing."""
    normal_values = [100.0] * 20
    extreme_values = [50000.0, -1000.0]
    df = pd.DataFrame({"order_amount": normal_values + extreme_values})
    
    outliers = detect_numeric_outliers(df, target_cols=["order_amount"])
    assert len(outliers) == 1
    assert outliers[0]["column"] == "order_amount"
    assert outliers[0]["outlier_count"] >= 2


def test_suspicious_records_detection():
    """Verify identification of illogical records."""
    df = pd.DataFrame({
        "order_status": ["Cancelled", "Delivered"],
        "delivery_duration_min": [45.0, 30.0],
        "preparation_time_min": [15.0, 40.0],  # 40 > 30 duration
        "order_amount": [-100.0, 200.0]
    })
    suspicious = inspect_suspicious_records(df)
    assert suspicious["cancelled_with_duration"] == 1
    assert suspicious["prep_exceeds_duration"] == 1
    assert suspicious["non_positive_amount"] == 1


def test_apply_dashboard_filters(deterministic_sample_orders):
    """Verify multi-criteria filtering works properly."""
    # Filter by zone 'Downtown'
    filtered = apply_dashboard_filters(deterministic_sample_orders, selected_zones=["Downtown"])
    assert len(filtered) == 3
    assert (filtered["delivery_zone"] == "Downtown").all()

    # Combined filter: Downtown + Status 'Delivered'
    filtered_combo = apply_dashboard_filters(
        deterministic_sample_orders,
        selected_zones=["Downtown"],
        selected_statuses=["Delivered"]
    )
    assert len(filtered_combo) == 2

    # Filter with no matches
    filtered_empty = apply_dashboard_filters(
        deterministic_sample_orders,
        selected_zones=["NonExistentZone"]
    )
    assert len(filtered_empty) == 0


def test_spark_compat_checker():
    """Verify spark environment checker executes cleanly and reports dictionary."""
    compat = check_spark_environment()
    assert isinstance(compat, dict)
    assert "java_installed" in compat
    assert "pyspark_installed" in compat
    assert "recommendation" in compat
