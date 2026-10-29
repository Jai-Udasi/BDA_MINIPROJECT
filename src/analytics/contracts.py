"""
Analytical Output Contracts and Schema Specifications.
Defines standard interfaces for analytics outputs, ensuring that both
Phase 2 (Pandas baseline) and Phase 3 (PySpark DataFrames) produce identical,
interchangeable data structures consumed by the dashboard.
"""
from typing import TypedDict, List, Dict, Any, Optional


class DashboardKPIs(TypedDict, total=False):
    """Data contract for top-level operational and executive KPIs."""
    total_records: int
    total_unique_orders: int
    completed_orders: int
    cancelled_orders: int
    in_transit_orders: int
    completion_rate_pct: float
    cancellation_rate_pct: float
    total_revenue: float
    avg_order_value: float
    median_order_value: float
    avg_prep_time_min: float
    avg_delivery_duration_min: float
    median_delivery_duration_min: float
    delayed_orders_count: int
    eligible_delay_orders_count: int
    delay_rate_pct: float
    unique_restaurants: int
    unique_zones: int


class TemporalAggregationSchema:
    """Expected schema for temporal order demand aggregations."""
    HOURLY_COLUMNS = ["hour", "order_count", "order_share_pct", "is_peak"]
    WEEKDAY_COLUMNS = ["weekday_num", "weekday_name", "order_count", "avg_duration_min", "is_weekend"]
    DAILY_COLUMNS = ["date", "order_count", "total_amount", "delay_rate_pct"]


class SpatialAggregationSchema:
    """Expected schema for zone and restaurant comparisons."""
    ZONE_COLUMNS = [
        "delivery_zone",
        "sample_size_orders",
        "completed_orders",
        "cancellation_rate_pct",
        "avg_order_amount",
        "avg_distance_km",
        "avg_duration_min",
        "median_duration_min",
        "delay_rate_pct"
    ]
    RESTAURANT_COLUMNS = [
        "restaurant_id",
        "restaurant_type",
        "sample_size_orders",
        "completed_orders",
        "avg_order_amount",
        "avg_prep_time_min",
        "avg_duration_min",
        "delay_rate_pct"
    ]


class OutlierSummary(TypedDict):
    """Data contract for numeric outlier detection results."""
    column: str
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    outlier_count: int
    outlier_percentage: float
