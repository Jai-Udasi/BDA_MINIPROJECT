"""
Module for generating reproducible, realistic synthetic food delivery orders.
Used for initial development, testing, and pipeline verification.
"""
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add project root to sys.path if executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import (
    SYNTHETIC_ORDERS_FILE,
    RAW_ORDERS_FILE,
    RANDOM_SEED,
    DEFAULT_NUM_RECORDS,
    DELIVERY_ZONES,
    RESTAURANT_TYPES,
    WEATHER_CONDITIONS,
    TRAFFIC_LEVELS
)

# Configure module logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataGeneration")


def generate_synthetic_orders(
    num_records: int = DEFAULT_NUM_RECORDS,
    seed: int = RANDOM_SEED,
    include_noise: bool = True
) -> pd.DataFrame:
    """
    Generates a synthetic dataset of food delivery orders with realistic distributions,
    temporal patterns, and controlled data-quality edge cases.

    Parameters:
    -----------
    num_records : int
        Number of order records to synthesize.
    seed : int
        Random seed for full reproducibility.
    include_noise : bool
        If True, injects controlled anomalies (missing values, duplicates, outliers)
        to demonstrate and test the data cleaning pipeline.

    Returns:
    --------
    pd.DataFrame
        Synthesized food delivery order records.
    """
    logger.info(f"Generating {num_records} synthetic order records (seed={seed}, include_noise={include_noise})...")
    np.random.seed(seed)

    # 1. Generate Order IDs
    order_ids = [f"ORD-{10000 + i}" for i in range(num_records)]

    # 2. Generate Temporal Distribution with Peak Hours (Lunch: 12-14, Dinner: 19-22)
    start_date = datetime(2026, 10, 1, 8, 0, 0)
    timestamps = []
    
    # Hour probability distribution mirroring real food delivery rush periods
    hour_weights = [
        0.01, 0.01, 0.005, 0.005, 0.005, 0.01,  # 00:00 - 05:00 (late night / early dawn)
        0.02, 0.03, 0.04, 0.04, 0.05, 0.08,    # 06:00 - 11:00 (breakfast / pre-lunch)
        0.12, 0.11, 0.06, 0.04, 0.04, 0.05,    # 12:00 - 17:00 (lunch rush & afternoon)
        0.08, 0.12, 0.13, 0.08, 0.04, 0.02     # 18:00 - 23:00 (dinner rush)
    ]
    hour_weights = np.array(hour_weights) / sum(hour_weights)

    for _ in range(num_records):
        day_offset = np.random.randint(0, 10)  # 10 days of data
        chosen_hour = np.random.choice(range(24), p=hour_weights)
        chosen_minute = np.random.randint(0, 60)
        chosen_second = np.random.randint(0, 60)
        order_time = start_date + timedelta(
            days=int(day_offset),
            hours=int(chosen_hour),
            minutes=int(chosen_minute),
            seconds=int(chosen_second)
        )
        timestamps.append(order_time.strftime("%Y-%m-%d %H:%M:%S"))

    # 3. Generate Restaurant Attributes
    restaurants = [f"REST-{100 + i}" for i in range(30)]
    restaurant_ids = np.random.choice(restaurants, size=num_records)
    
    # Associate restaurant types probabilistically
    rest_type_map = {
        r: np.random.choice(RESTAURANT_TYPES) for r in restaurants
    }
    restaurant_types = [rest_type_map[r] for r in restaurant_ids]

    # 4. Generate Spatial Attributes
    zone_weights = [0.28, 0.18, 0.16, 0.20, 0.12, 0.06]  # Downtown & Tech Park higher volume
    delivery_zones = np.random.choice(DELIVERY_ZONES, size=num_records, p=zone_weights)

    # 5. Operational Metrics: Items, Amount, Prep Time, Distance
    item_counts = np.random.choice([1, 2, 3, 4, 5, 6, 7], size=num_records, p=[0.25, 0.35, 0.20, 0.10, 0.05, 0.03, 0.02])
    
    # Base amounts influenced by restaurant type and item count
    base_price_map = {
        "Fast Food": 120.0,
        "Casual Dining": 220.0,
        "Cafe": 150.0,
        "Cloud Kitchen": 180.0,
        "Bakery & Desserts": 140.0,
        "Fine Dining": 450.0
    }
    order_amounts = []
    for r_type, items in zip(restaurant_types, item_counts):
        unit_price = base_price_map[r_type] * np.random.uniform(0.85, 1.25)
        total_amt = round(unit_price * items + np.random.uniform(20.0, 50.0), 2)
        order_amounts.append(total_amt)

    # Prep time depends on item count and kitchen type
    prep_times = []
    for r_type, items in zip(restaurant_types, item_counts):
        base_prep = 12.0 if r_type in ["Fast Food", "Cloud Kitchen"] else 22.0
        prep = base_prep + (items * np.random.uniform(1.8, 3.2)) + np.random.normal(0, 3)
        prep_times.append(round(max(6.0, prep), 1))

    # Delivery distance (km)
    distances = np.random.gamma(shape=2.5, scale=1.8, size=num_records)
    distances = [round(max(0.5, min(d, 28.0)), 2) for d in distances]

    # Weather & Traffic
    weather_choices = np.random.choice(
        WEATHER_CONDITIONS,
        size=num_records,
        p=[0.60, 0.18, 0.10, 0.07, 0.05]
    )
    traffic_choices = np.random.choice(
        TRAFFIC_LEVELS,
        size=num_records,
        p=[0.25, 0.40, 0.25, 0.10]
    )

    # Calculate Realistic Delivery Duration (Physics & Operations simulation)
    # Transit speed varies by traffic and weather
    speed_factor = {"Low": 1.1, "Medium": 1.4, "High": 2.0, "Jam": 2.8}
    weather_delay = {"Clear": 0.0, "Windy": 2.0, "Foggy": 4.0, "Rainy": 6.0, "Heavy Rain": 12.0}

    delivery_durations = []
    is_delayed_flags = []

    for prep, dist, traffic, weather in zip(prep_times, distances, traffic_choices, weather_choices):
        # Transit time in minutes: distance * speed factor (mins/km) + pickup/drop buffer
        transit_time = (dist * 2.2 * speed_factor[traffic]) + weather_delay[weather] + np.random.normal(4.0, 1.5)
        total_duration = round(prep + max(3.0, transit_time), 1)
        delivery_durations.append(total_duration)

        # Standard Service Level Agreement (SLA): 40 mins base + 2.5 mins per km
        sla_threshold = 35.0 + (dist * 2.5)
        is_delayed_flags.append(1 if total_duration > sla_threshold else 0)

    # Order Status
    order_statuses = np.random.choice(
        ["Delivered", "Delivered", "Delivered", "Cancelled", "In Transit"],
        size=num_records,
        p=[0.88, 0.05, 0.03, 0.02, 0.02]
    )

    df = pd.DataFrame({
        "order_id": order_ids,
        "order_timestamp": timestamps,
        "restaurant_id": restaurant_ids,
        "restaurant_type": restaurant_types,
        "delivery_zone": delivery_zones,
        "order_amount": order_amounts,
        "item_count": item_counts,
        "preparation_time_min": prep_times,
        "delivery_distance_km": distances,
        "delivery_duration_min": delivery_durations,
        "order_status": order_statuses,
        "weather_condition": weather_choices,
        "traffic_density": traffic_choices,
        "is_delayed": is_delayed_flags
    })

    # 6. Inject controlled data-quality noise if requested
    if include_noise:
        # A. Introduce small number of duplicate records (~1.5%)
        num_dupes = max(5, int(num_records * 0.015))
        dupe_indices = np.random.choice(df.index, size=num_dupes, replace=False)
        dupes_df = df.loc[dupe_indices].copy()
        df = pd.concat([df, dupes_df], ignore_index=True)

        # B. Introduce missing values in optional fields (~2%)
        cols_with_nulls = ["restaurant_type", "weather_condition", "preparation_time_min", "delivery_distance_km"]
        for col in cols_with_nulls:
            null_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
            df.loc[null_indices, col] = np.nan

        # C. Introduce small anomalies (e.g. negative amount or 0 item count) (~0.5%)
        anomaly_indices = np.random.choice(df.index, size=max(2, int(len(df) * 0.005)), replace=False)
        for idx in anomaly_indices[:len(anomaly_indices)//2]:
            df.loc[idx, "order_amount"] = -150.0  # Erroneous negative value
        for idx in anomaly_indices[len(anomaly_indices)//2:]:
            df.loc[idx, "item_count"] = 0  # Invalid zero items

        # D. Introduce occasional date string variations / whitespace (~1%)
        date_var_indices = np.random.choice(df.index, size=max(2, int(len(df) * 0.01)), replace=False)
        for idx in date_var_indices:
            orig = str(df.loc[idx, "order_timestamp"])
            df.loc[idx, "order_timestamp"] = f"  {orig}  "  # Surrounding whitespace

    logger.info(f"Dataset generated successfully with {len(df)} total rows and {len(df.columns)} columns.")
    return df


def save_dataset(df: pd.DataFrame, file_path: Path) -> Path:
    """
    Safely saves DataFrame to CSV, ensuring parent directories exist.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info(f"Saved dataset to: {file_path} ({len(df)} rows)")
    return file_path


if __name__ == "__main__":
    logger.info("Running standalone synthetic data generation script...")
    dataset = generate_synthetic_orders(
        num_records=DEFAULT_NUM_RECORDS,
        seed=RANDOM_SEED,
        include_noise=True
    )
    # Save to both synthetic archive and raw input path
    save_dataset(dataset, SYNTHETIC_ORDERS_FILE)
    save_dataset(dataset, RAW_ORDERS_FILE)
    print(f"Data generation complete: {len(dataset)} records written to {RAW_ORDERS_FILE}")
