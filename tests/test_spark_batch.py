"""
Unit tests for Apache Spark (PySpark) Batch Analytics Engine.
Verifies SparkSession, DataFrame schema enforcement, aggregations,
and Spark SQL queries against ground-truth fixtures using pure JVM CSV ingestion.
"""
import pytest
from pathlib import Path
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType

from src.analytics.spark_batch import (
    get_spark_schema,
    get_or_create_spark_session,
    load_orders_spark_df,
    compute_spark_executive_kpis,
    compute_zone_demand_spark,
    run_spark_sql_analytics
)


@pytest.fixture(scope="session")
def spark_session():
    """Provides a shared local SparkSession for tests."""
    spark = get_or_create_spark_session(app_name="PyTest-Spark")
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def deterministic_csv_path(tmp_path_factory):
    """Creates a temporary deterministic CSV file for Spark ingestion."""
    fn = tmp_path_factory.mktemp("data") / "test_orders.csv"
    df = pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4", "ORD-5"],
        "order_timestamp": [
            "2026-10-01 12:00:00",
            "2026-10-01 13:00:00",
            "2026-10-01 14:00:00",
            "2026-10-01 19:00:00",
            "2026-10-01 20:00:00"
        ],
        "restaurant_id": ["REST-1", "REST-1", "REST-2", "REST-2", "REST-1"],
        "restaurant_type": ["Fast Food", "Fast Food", "Cafe", "Cafe", "Fast Food"],
        "delivery_zone": ["Downtown", "Downtown", "Suburbs", "Suburbs", "Downtown"],
        "order_amount": [200.0, 300.0, 150.0, 450.0, 100.0],
        "item_count": [2, 3, 1, 4, 1],
        "preparation_time_min": [15.0, 20.0, 10.0, 25.0, 10.0],
        "delivery_distance_km": [2.0, 4.0, 3.0, 6.0, 1.0],
        "delivery_duration_min": [25.0, 50.0, None, 60.0, None],
        "order_status": ["Delivered", "Delivered", "Cancelled", "Delivered", "In Transit"],
        "weather_condition": ["Clear", "Rainy", "Clear", "Clear", "Windy"],
        "traffic_density": ["Low", "High", "Low", "Medium", "Low"],
        "is_delayed": [0, 1, 0, 1, 0]
    })
    df.to_csv(fn, index=False)
    return fn


@pytest.fixture
def deterministic_spark_df(spark_session, deterministic_csv_path):
    """Loads deterministic CSV into a native PySpark DataFrame."""
    return load_orders_spark_df(spark_session, deterministic_csv_path)


def test_spark_schema_validity():
    """Verify PySpark StructType schema defines all 14 domain columns."""
    schema = get_spark_schema()
    assert isinstance(schema, StructType)
    assert len(schema.fields) == 14
    field_names = [f.name for f in schema.fields]
    assert "order_id" in field_names
    assert "delivery_zone" in field_names
    assert "order_amount" in field_names
    assert "is_delayed" in field_names


def test_spark_session_creation(spark_session):
    """Verify active local SparkSession is operational."""
    assert spark_session is not None
    assert spark_session.sparkContext.master.startswith("local")


def test_spark_kpi_computation(deterministic_spark_df):
    """Verify PySpark aggregations calculate exact ground-truth KPIs."""
    kpis = compute_spark_executive_kpis(deterministic_spark_df)

    assert kpis["total_records"] == 5
    assert kpis["total_unique_orders"] == 5
    assert kpis["completed_orders"] == 3
    assert kpis["cancelled_orders"] == 1
    # Non-cancelled revenue: 200 + 300 + 450 + 100 = 1050.0
    assert kpis["total_revenue"] == 1050.0
    assert kpis["delayed_orders_count"] == 2
    assert kpis["eligible_delay_orders_count"] == 3
    assert kpis["delay_rate_pct"] == round(2 / 3 * 100.0, 2)
    assert kpis["unique_zones"] == 2


def test_spark_zone_aggregation(deterministic_spark_df):
    """Verify PySpark groupBy and agg produce accurate zone-level metrics."""
    zone_df = compute_zone_demand_spark(deterministic_spark_df)
    results = {row["delivery_zone"]: row for row in zone_df.collect()}

    assert "Downtown" in results
    assert "Suburbs" in results
    assert results["Downtown"]["total_orders"] == 3
    assert results["Downtown"]["completed_orders"] == 2
    assert results["Downtown"]["total_gmv"] == 600.0


def test_spark_sql_analytics(spark_session, deterministic_spark_df):
    """Verify Spark SQL query execution over temporary table view."""
    sql_res = run_spark_sql_analytics(spark_session, deterministic_spark_df)
    assert "weather_traffic_cross_tab" in sql_res
    assert "peak_rush_comparison" in sql_res
    assert len(sql_res["weather_traffic_cross_tab"]) > 0
