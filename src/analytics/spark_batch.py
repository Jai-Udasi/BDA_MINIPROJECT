"""
Apache Spark (PySpark) Batch Analytics and Distributed Data Processing Engine.
Implements Syllabus Module 2 (Working with Apache Spark) and Module 3 (Relational Algebra & Aggregations).
Uses native PySpark DataFrames, Spark SQL, and Window functions.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Automatically configure Java & Python environment variables for PySpark on Windows
JDK_PATH = r"C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot"
if os.path.exists(JDK_PATH):
    os.environ["JAVA_HOME"] = JDK_PATH

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import PROCESSED_ORDERS_FILE, PROCESSED_DATA_DIR

# PySpark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType
)
import pyspark.sql.functions as F
from pyspark.sql.window import Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SparkBatchEngine")

SPARK_OUTPUT_DIR = PROCESSED_DATA_DIR / "spark_output"


def get_spark_schema() -> StructType:
    """
    Defines the explicit Big Data schema for Food Delivery Orders.
    Enforces strong typing for distributed PySpark processing.
    """
    return StructType([
        StructField("order_id", StringType(), False),
        StructField("order_timestamp", TimestampType(), True),
        StructField("restaurant_id", StringType(), True),
        StructField("restaurant_type", StringType(), True),
        StructField("delivery_zone", StringType(), True),
        StructField("order_amount", DoubleType(), True),
        StructField("item_count", IntegerType(), True),
        StructField("preparation_time_min", DoubleType(), True),
        StructField("delivery_distance_km", DoubleType(), True),
        StructField("delivery_duration_min", DoubleType(), True),
        StructField("order_status", StringType(), True),
        StructField("weather_condition", StringType(), True),
        StructField("traffic_density", StringType(), True),
        StructField("is_delayed", IntegerType(), True),
    ])


def get_or_create_spark_session(app_name: str = "FoodDeliveryBigData-Spark") -> SparkSession:
    """
    Creates or retrieves an optimized local SparkSession configured for Windows.
    """
    builder = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.ui.enabled", "false")  # Avoid port binding conflicts in test environments
        .config("spark.sql.shuffle.partitions", "4")  # Optimized for local multi-core execution
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_orders_spark_df(spark: SparkSession, csv_path: Path = PROCESSED_ORDERS_FILE) -> DataFrame:
    """
    Loads order transactions into a PySpark DataFrame with strict schema enforcement.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Source dataset not found at: {csv_path}")

    schema = get_spark_schema()
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
        .schema(schema)
        .load(str(csv_path))
    )
    return df


def compute_spark_executive_kpis(df: DataFrame) -> Dict[str, Any]:
    """
    Computes top-level Big Data operational KPIs in a SINGLE consolidated Spark aggregation pass.
    Drastically reduces job scheduling latency and execution time.
    """
    row = df.agg(
        F.count("*").alias("total_records"),
        F.countDistinct("order_id").alias("unique_orders"),
        F.countDistinct("restaurant_id").alias("unique_restaurants"),
        F.countDistinct("delivery_zone").alias("unique_zones"),
        F.sum(F.when(F.lower(F.col("order_status")) == "delivered", 1).otherwise(0)).alias("completed_orders"),
        F.sum(F.when(F.lower(F.col("order_status")) == "cancelled", 1).otherwise(0)).alias("cancelled_orders"),
        F.sum(F.when(F.lower(F.col("order_status")) != "cancelled", F.col("order_amount")).otherwise(0.0)).alias("total_gmv"),
        F.avg(F.when(F.lower(F.col("order_status")) != "cancelled", F.col("order_amount"))).alias("avg_order_value"),
        F.avg(F.when(F.lower(F.col("order_status")) == "delivered", F.col("delivery_duration_min"))).alias("avg_dur"),
        F.percentile_approx(F.when(F.lower(F.col("order_status")) == "delivered", F.col("delivery_duration_min")), 0.5).alias("median_dur"),
        F.avg(F.when(F.lower(F.col("order_status")) == "delivered", F.col("preparation_time_min"))).alias("avg_prep"),
        F.sum(F.when((F.lower(F.col("order_status")) == "delivered") & F.col("delivery_duration_min").isNotNull(), 1).otherwise(0)).alias("eligible_count"),
        F.sum(F.when((F.lower(F.col("order_status")) == "delivered") & (F.col("is_delayed") == 1), 1).otherwise(0)).alias("delayed_count")
    ).first()

    total_records = int(row["total_records"] or 0)
    unique_orders = int(row["unique_orders"] or 0)
    completed_orders = int(row["completed_orders"] or 0)
    cancelled_orders = int(row["cancelled_orders"] or 0)

    completion_rate_pct = round((completed_orders / unique_orders * 100.0), 2) if unique_orders > 0 else 0.0
    cancellation_rate_pct = round((cancelled_orders / unique_orders * 100.0), 2) if unique_orders > 0 else 0.0

    total_gmv = round(float(row["total_gmv"] or 0.0), 2)
    avg_order_val = round(float(row["avg_order_value"] or 0.0), 2)

    avg_dur = round(float(row["avg_dur"] or 0.0), 1)
    median_dur = round(float(row["median_dur"] or 0.0), 1)
    avg_prep = round(float(row["avg_prep"] or 0.0), 1)

    eligible_count = int(row["eligible_count"] or 0)
    delayed_count = int(row["delayed_count"] or 0)
    delay_rate_pct = round((delayed_count / eligible_count * 100.0), 2) if eligible_count > 0 else 0.0

    unique_restaurants = int(row["unique_restaurants"] or 0)
    unique_zones = int(row["unique_zones"] or 0)

    return {
        "engine": "Apache Spark (PySpark DataFrames)",
        "total_records": total_records,
        "total_unique_orders": unique_orders,
        "completed_orders": completed_orders,
        "cancelled_orders": cancelled_orders,
        "completion_rate_pct": completion_rate_pct,
        "cancellation_rate_pct": cancellation_rate_pct,
        "total_revenue": total_gmv,
        "avg_order_value": avg_order_val,
        "avg_prep_time_min": avg_prep,
        "avg_delivery_duration_min": avg_dur,
        "median_delivery_duration_min": median_dur,
        "delayed_orders_count": delayed_count,
        "eligible_delay_orders_count": eligible_count,
        "delay_rate_pct": delay_rate_pct,
        "unique_restaurants": unique_restaurants,
        "unique_zones": unique_zones
    }


def compute_zone_demand_spark(df: DataFrame) -> DataFrame:
    """
    Spatial grouping and aggregation via PySpark DataFrame API.
    Computes volume, GMV, avg delivery duration, and delay percentage per zone.
    """
    zone_agg = (
        df.groupBy("delivery_zone")
        .agg(
            F.count("*").alias("total_orders"),
            F.sum(F.when(F.lower(F.col("order_status")) == "delivered", 1).otherwise(0)).alias("completed_orders"),
            F.round(F.sum("order_amount"), 2).alias("total_gmv"),
            F.round(F.avg("order_amount"), 2).alias("avg_order_amount"),
            F.round(F.avg("delivery_distance_km"), 2).alias("avg_distance_km"),
            F.round(F.avg("delivery_duration_min"), 1).alias("avg_duration_min"),
            F.round(F.mean("is_delayed") * 100.0, 2).alias("delay_rate_pct")
        )
        .orderBy(F.col("total_orders").desc())
    )
    return zone_agg


def compute_hourly_velocity_spark(df: DataFrame) -> DataFrame:
    """
    Temporal demand aggregation using PySpark SQL hour() function.
    """
    hourly_agg = (
        df.filter(F.col("order_timestamp").isNotNull())
        .withColumn("order_hour", F.hour("order_timestamp"))
        .groupBy("order_hour")
        .agg(
            F.count("*").alias("order_count"),
            F.round(F.avg("delivery_duration_min"), 1).alias("avg_duration_min"),
            F.round(F.mean("is_delayed") * 100.0, 2).alias("delay_rate_pct")
        )
        .orderBy("order_hour")
    )
    return hourly_agg


def compute_restaurant_rankings_window_spark(df: DataFrame) -> DataFrame:
    """
    Demonstrates PySpark Window Functions:
    Ranks top restaurants within each delivery zone by order volume.
    """
    rest_zone = (
        df.groupBy("delivery_zone", "restaurant_id", "restaurant_type")
        .agg(
            F.count("*").alias("order_count"),
            F.round(F.sum("order_amount"), 2).alias("revenue")
        )
    )

    zone_window = Window.partitionBy("delivery_zone").orderBy(F.col("order_count").desc())
    ranked = rest_zone.withColumn("rank_in_zone", F.dense_rank().over(zone_window))
    return ranked.filter(F.col("rank_in_zone") <= 3).orderBy("delivery_zone", "rank_in_zone")


def run_spark_sql_analytics(spark: SparkSession, df: DataFrame) -> Dict[str, Any]:
    """
    Demonstrates Spark SQL query execution via temporary views.
    Executes analytical queries to uncover high-delay bottlenecks.
    """
    df.createOrReplaceTempView("orders")

    # 1. Weather and traffic impact cross-tabulation
    sql_weather_traffic = """
        SELECT
            weather_condition,
            traffic_density,
            COUNT(*) as order_count,
            ROUND(AVG(delivery_duration_min), 1) as avg_duration,
            ROUND(AVG(is_delayed) * 100.0, 2) as delay_rate_pct
        FROM orders
        WHERE order_status = 'Delivered'
        GROUP BY weather_condition, traffic_density
        ORDER BY delay_rate_pct DESC
    """
    weather_traffic_df = spark.sql(sql_weather_traffic).toPandas()

    # 2. Peak rush hours comparison
    sql_peak_hours = """
        SELECT
            CASE
                WHEN HOUR(order_timestamp) IN (12, 13, 14) THEN 'Lunch Rush (12-14h)'
                WHEN HOUR(order_timestamp) IN (19, 20, 21) THEN 'Dinner Rush (19-21h)'
                ELSE 'Regular / Off-Peak'
            END as period_window,
            COUNT(*) as order_volume,
            ROUND(SUM(order_amount), 2) as total_revenue,
            ROUND(AVG(delivery_duration_min), 1) as avg_delivery_time,
            ROUND(AVG(is_delayed) * 100.0, 2) as delay_pct
        FROM orders
        GROUP BY 1
        ORDER BY order_volume DESC
    """
    peak_hours_df = spark.sql(sql_peak_hours).toPandas()

    return {
        "weather_traffic_cross_tab": weather_traffic_df.to_dict(orient="records"),
        "peak_rush_comparison": peak_hours_df.to_dict(orient="records")
    }


def run_spark_batch_pipeline(csv_path: Path = PROCESSED_ORDERS_FILE) -> Dict[str, Any]:
    """
    Orchestrates the complete PySpark batch processing pipeline:
    1. Local SparkSession Initialization.
    2. DataFrame Loading with Schema.
    3. Executive KPI Computation.
    4. Spatial & Temporal Aggregations.
    5. Windowed Restaurant Rankings.
    6. Spark SQL Query Executions.
    7. Storage of Curated Big Data Outputs.
    """
    logger.info("Initializing Apache Spark batch processing pipeline...")
    spark = get_or_create_spark_session()
    
    df = load_orders_spark_df(spark, csv_path)
    
    kpis = compute_spark_executive_kpis(df)
    zone_df = compute_zone_demand_spark(df)
    hourly_df = compute_hourly_velocity_spark(df)
    ranked_rest_df = compute_restaurant_rankings_window_spark(df)
    sql_results = run_spark_sql_analytics(spark, df)

    # Capture Spark Physical Execution Plan
    plan_output = df.groupBy("delivery_zone").count()._jdf.queryExecution().simpleString()

    # Convert Spark DataFrames to Python dictionaries for clean dashboard consumption
    zone_records = [row.asDict() for row in zone_df.collect()]
    hourly_records = [row.asDict() for row in hourly_df.collect()]
    ranked_records = [row.asDict() for row in ranked_rest_df.collect()]

    # Export to spark_output directory
    SPARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SPARK_OUTPUT_DIR / "spark_batch_summary.json"
    
    output_payload = {
        "spark_version": spark.version,
        "spark_master": spark.sparkContext.master,
        "spark_app_name": spark.sparkContext.appName,
        "execution_plan_sample": plan_output,
        "kpis": kpis,
        "zone_aggregations": zone_records,
        "hourly_aggregations": hourly_records,
        "ranked_restaurants_window": ranked_records,
        "spark_sql_analytics": sql_results
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=4)

    logger.info(f"PySpark batch processing complete. Exported results to: {summary_path}")
    return output_payload


if __name__ == "__main__":
    print("=" * 60)
    print("  APACHE SPARK (PySpark) BATCH ANALYTICS ENGINE")
    print("=" * 60)
    
    results = run_spark_batch_pipeline()
    print(f"Spark Version: {results['spark_version']} | Master: {results['spark_master']}")
    print(f"Total Unique Orders Processed: {results['kpis']['total_unique_orders']:,}")
    print(f"Total Revenue (GMV): Rs. {results['kpis']['total_revenue']:,.2f}")
    print(f"Average Delivery Duration: {results['kpis']['avg_delivery_duration_min']} min")
    print(f"Delayed Orders Rate: {results['kpis']['delay_rate_pct']}%")
    print("\nTop Zones by Volume (PySpark GroupBy & Agg):")
    for z in results['zone_aggregations'][:3]:
        print(f"   {z['delivery_zone']:<25} Orders: {z['total_orders']:<6} GMV: Rs. {z['total_gmv']:<10.2f}")
    print("=" * 60)
