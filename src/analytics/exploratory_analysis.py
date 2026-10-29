"""
Exploratory Data Analytics (EDA) Module for Food Delivery Operations.
Provides reusable, pure-function analytical transformations, KPIs,
temporal distributions, spatial comparisons, and data-quality auditing.

Designed with clean data contracts to ensure compatibility with Phase 3 PySpark.
"""
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


def compute_executive_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level operational and executive KPIs.
    
    Metrics & Formulas:
    - Total Unique Orders: count of distinct non-null order_ids (or total rows if order_id is absent).
    - Completed Orders: count where order_status == 'Delivered' (case-insensitive).
    - Cancelled Orders: count where order_status == 'Cancelled' (case-insensitive).
    - Completion Rate (%): (Completed Orders / Total Unique Orders) * 100
    - Cancellation Rate (%): (Cancelled Orders / Total Unique Orders) * 100
    - Total Revenue (₹): Sum of order_amount for non-cancelled orders.
    - Average / Median Delivery Duration (min): Computed exclusively on completed orders
      with non-null delivery durations. Never counts missing durations as zero.
    - Delayed-Order Percentage (%):
        Denominator: Count of completed orders with valid delivery duration.
        Numerator: Count of those completed orders where is_delayed == 1.
    """
    if df.empty:
        return {
            "total_records": 0,
            "total_unique_orders": 0,
            "completed_orders": 0,
            "cancelled_orders": 0,
            "in_transit_orders": 0,
            "completion_rate_pct": 0.0,
            "cancellation_rate_pct": 0.0,
            "total_revenue": 0.0,
            "avg_order_value": 0.0,
            "median_order_value": 0.0,
            "avg_prep_time_min": 0.0,
            "avg_delivery_duration_min": 0.0,
            "median_delivery_duration_min": 0.0,
            "delayed_orders_count": 0,
            "eligible_delay_orders_count": 0,
            "delay_rate_pct": 0.0,
            "unique_restaurants": 0,
            "unique_zones": 0
        }

    total_records = len(df)
    unique_orders = int(df["order_id"].nunique()) if "order_id" in df.columns else total_records

    # Status distribution
    completed_mask = pd.Series(True, index=df.index)
    cancelled_count = 0
    in_transit_count = 0
    if "order_status" in df.columns:
        status_clean = df["order_status"].astype(str).str.strip().str.lower()
        completed_mask = status_clean == "delivered"
        cancelled_count = int((status_clean == "cancelled").sum())
        in_transit_count = int((status_clean == "in transit").sum())
    completed_count = int(completed_mask.sum())

    completion_rate = round((completed_count / unique_orders * 100.0), 2) if unique_orders > 0 else 0.0
    cancellation_rate = round((cancelled_count / unique_orders * 100.0), 2) if unique_orders > 0 else 0.0

    # Revenue calculation (non-cancelled orders)
    non_cancelled_df = df.loc[~df["order_status"].astype(str).str.strip().str.lower().eq("cancelled")] if "order_status" in df.columns else df
    total_rev = float(non_cancelled_df["order_amount"].sum()) if "order_amount" in df.columns else 0.0
    avg_order_val = float(df["order_amount"].mean()) if "order_amount" in df.columns else 0.0
    med_order_val = float(df["order_amount"].median()) if "order_amount" in df.columns else 0.0

    # Delivery durations: strictly completed orders with non-null duration
    completed_durations = pd.Series(dtype=float)
    if "delivery_duration_min" in df.columns:
        valid_dur_mask = completed_mask & df["delivery_duration_min"].notna()
        completed_durations = df.loc[valid_dur_mask, "delivery_duration_min"]

    avg_duration = float(completed_durations.mean()) if not completed_durations.empty else 0.0
    med_duration = float(completed_durations.median()) if not completed_durations.empty else 0.0

    # Prep time calculation
    valid_prep = df["preparation_time_min"].dropna() if "preparation_time_min" in df.columns else pd.Series(dtype=float)
    avg_prep = float(valid_prep.mean()) if not valid_prep.empty else 0.0

    # Delay rate calculation
    delayed_count = 0
    eligible_delay_count = 0
    delay_rate = 0.0
    if "is_delayed" in df.columns and not completed_durations.empty:
        eligible_mask = completed_mask & df["delivery_duration_min"].notna() & df["is_delayed"].notna()
        eligible_delay_count = int(eligible_mask.sum())
        delayed_count = int((df.loc[eligible_mask, "is_delayed"] == 1).sum())
        delay_rate = round((delayed_count / eligible_delay_count * 100.0), 2) if eligible_delay_count > 0 else 0.0

    # Counts of entities
    unique_rests = int(df["restaurant_id"].nunique()) if "restaurant_id" in df.columns else 0
    unique_zones = int(df["delivery_zone"].nunique()) if "delivery_zone" in df.columns else 0

    return {
        "total_records": total_records,
        "total_unique_orders": unique_orders,
        "completed_orders": completed_count,
        "cancelled_orders": cancelled_count,
        "in_transit_orders": in_transit_count,
        "completion_rate_pct": completion_rate,
        "cancellation_rate_pct": cancellation_rate,
        "total_revenue": round(total_rev, 2),
        "avg_order_value": round(avg_order_val, 2),
        "median_order_value": round(med_order_val, 2),
        "avg_prep_time_min": round(avg_prep, 1),
        "avg_delivery_duration_min": round(avg_duration, 1),
        "median_delivery_duration_min": round(med_duration, 1),
        "delayed_orders_count": delayed_count,
        "eligible_delay_orders_count": eligible_delay_count,
        "delay_rate_pct": delay_rate,
        "unique_restaurants": unique_rests,
        "unique_zones": unique_zones
    }


def get_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes hourly order volume, percentage share, and identifies peak-demand rush periods.
    """
    if df.empty or "order_timestamp" not in df.columns:
        return pd.DataFrame(columns=["hour", "order_count", "order_share_pct", "is_peak"])

    temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp["order_timestamp"]):
        temp["order_timestamp"] = pd.to_datetime(temp["order_timestamp"], errors="coerce")

    valid_ts = temp.dropna(subset=["order_timestamp"]).copy()
    if valid_ts.empty:
        return pd.DataFrame(columns=["hour", "order_count", "order_share_pct", "is_peak"])

    valid_ts["hour"] = valid_ts["order_timestamp"].dt.hour
    hourly = valid_ts.groupby("hour").size().reset_index(name="order_count")

    # Complete 24 hours grid
    grid = pd.DataFrame({"hour": list(range(24))})
    merged = pd.merge(grid, hourly, on="hour", how="left").fillna(0)
    merged["order_count"] = merged["order_count"].astype(int)

    total_orders = merged["order_count"].sum()
    merged["order_share_pct"] = (merged["order_count"] / total_orders * 100.0).round(2) if total_orders > 0 else 0.0
    
    # Peak periods defined as lunch (12-14) or dinner (19-21) or top 25% order volume
    volume_threshold = merged["order_count"].quantile(0.75) if total_orders > 0 else 0
    merged["is_peak"] = (merged["order_count"] >= volume_threshold) | (merged["hour"].isin([12, 13, 19, 20]))
    return merged


def get_weekday_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes order demand by day of week (Monday-Sunday) and compares against delivery duration.
    """
    if df.empty or "order_timestamp" not in df.columns:
        return pd.DataFrame(columns=["weekday_num", "weekday_name", "order_count", "avg_duration_min", "is_weekend"])

    temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp["order_timestamp"]):
        temp["order_timestamp"] = pd.to_datetime(temp["order_timestamp"], errors="coerce")

    valid_ts = temp.dropna(subset=["order_timestamp"]).copy()
    if valid_ts.empty:
        return pd.DataFrame(columns=["weekday_num", "weekday_name", "order_count", "avg_duration_min", "is_weekend"])

    valid_ts["weekday_num"] = valid_ts["order_timestamp"].dt.weekday
    valid_ts["weekday_name"] = valid_ts["order_timestamp"].dt.day_name()

    agg_rules: Dict[str, Any] = {"order_count": ("order_timestamp", "count")}
    if "delivery_duration_min" in valid_ts.columns:
        agg_rules["avg_duration_min"] = ("delivery_duration_min", "mean")

    weekday_summary = valid_ts.groupby(["weekday_num", "weekday_name"]).agg(**agg_rules).reset_index()
    if "avg_duration_min" in weekday_summary.columns:
        weekday_summary["avg_duration_min"] = weekday_summary["avg_duration_min"].round(1)
    else:
        weekday_summary["avg_duration_min"] = 0.0

    weekday_summary = weekday_summary.sort_values(by="weekday_num")
    weekday_summary["is_weekend"] = weekday_summary["weekday_num"].isin([5, 6])
    return weekday_summary


def get_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily volume, daily revenue, and daily delay rates over the operational calendar.
    """
    if df.empty or "order_timestamp" not in df.columns:
        return pd.DataFrame(columns=["date", "order_count", "total_amount", "delay_rate_pct"])

    temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(temp["order_timestamp"]):
        temp["order_timestamp"] = pd.to_datetime(temp["order_timestamp"], errors="coerce")

    valid_ts = temp.dropna(subset=["order_timestamp"]).copy()
    if valid_ts.empty:
        return pd.DataFrame(columns=["date", "order_count", "total_amount", "delay_rate_pct"])

    valid_ts["date"] = valid_ts["order_timestamp"].dt.date
    
    agg_rules: Dict[str, Any] = {"order_count": ("order_timestamp", "count")}
    if "order_amount" in valid_ts.columns:
        agg_rules["total_amount"] = ("order_amount", "sum")
    if "is_delayed" in valid_ts.columns:
        agg_rules["delay_rate_pct"] = ("is_delayed", lambda s: round(s.mean() * 100.0, 2))

    daily = valid_ts.groupby("date").agg(**agg_rules).reset_index()
    daily["date"] = daily["date"].astype(str)
    if "total_amount" in daily.columns:
        daily["total_amount"] = daily["total_amount"].round(2)
    return daily.sort_values(by="date")


def get_zone_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates delivery performance, sample size (N), cancellation rates,
    average distances, and delay percentages per delivery zone.
    """
    if df.empty or "delivery_zone" not in df.columns:
        return pd.DataFrame(columns=[
            "delivery_zone", "sample_size_orders", "completed_orders",
            "cancellation_rate_pct", "avg_order_amount", "avg_distance_km",
            "avg_duration_min", "median_duration_min", "delay_rate_pct"
        ])

    def calc_zone_stats(grp):
        n_orders = len(grp)
        status_clean = grp["order_status"].astype(str).str.lower() if "order_status" in grp.columns else pd.Series("delivered", index=grp.index)
        completed = int((status_clean == "delivered").sum())
        cancelled = int((status_clean == "cancelled").sum())
        cancel_rate = round((cancelled / n_orders * 100.0), 2) if n_orders > 0 else 0.0

        avg_amt = round(float(grp["order_amount"].mean()), 2) if "order_amount" in grp.columns else 0.0
        avg_dist = round(float(grp["delivery_distance_km"].mean()), 2) if "delivery_distance_km" in grp.columns else 0.0
        
        # Duration for completed orders with non-null duration
        durations = grp.loc[status_clean == "delivered", "delivery_duration_min"].dropna() if "delivery_duration_min" in grp.columns else pd.Series(dtype=float)
        avg_dur = round(float(durations.mean()), 1) if not durations.empty else 0.0
        med_dur = round(float(durations.median()), 1) if not durations.empty else 0.0

        delay_pct = 0.0
        if "is_delayed" in grp.columns and not durations.empty:
            delays = grp.loc[status_clean == "delivered", "is_delayed"].dropna()
            delay_pct = round(float(delays.mean() * 100.0), 2) if not delays.empty else 0.0

        return pd.Series({
            "sample_size_orders": n_orders,
            "completed_orders": completed,
            "cancellation_rate_pct": cancel_rate,
            "avg_order_amount": avg_amt,
            "avg_distance_km": avg_dist,
            "avg_duration_min": avg_dur,
            "median_duration_min": med_dur,
            "delay_rate_pct": delay_pct
        })

    summary = df.groupby("delivery_zone", group_keys=False).apply(calc_zone_stats).reset_index()
    return summary.sort_values(by="sample_size_orders", ascending=False)


def get_restaurant_performance(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Computes restaurant-level volume, food prep speed, and delay rate with sample size (N).
    """
    if df.empty or "restaurant_id" not in df.columns:
        return pd.DataFrame(columns=[
            "restaurant_id", "restaurant_type", "sample_size_orders",
            "completed_orders", "avg_order_amount", "avg_prep_time_min",
            "avg_duration_min", "delay_rate_pct"
        ])

    def calc_rest_stats(grp):
        n_orders = len(grp)
        rest_type = grp["restaurant_type"].iloc[0] if "restaurant_type" in grp.columns and not grp["restaurant_type"].dropna().empty else "Unknown"
        status_clean = grp["order_status"].astype(str).str.lower() if "order_status" in grp.columns else pd.Series("delivered", index=grp.index)
        completed = int((status_clean == "delivered").sum())

        avg_amt = round(float(grp["order_amount"].mean()), 2) if "order_amount" in grp.columns else 0.0
        avg_prep = round(float(grp["preparation_time_min"].mean()), 1) if "preparation_time_min" in grp.columns else 0.0
        
        durations = grp.loc[status_clean == "delivered", "delivery_duration_min"].dropna() if "delivery_duration_min" in grp.columns else pd.Series(dtype=float)
        avg_dur = round(float(durations.mean()), 1) if not durations.empty else 0.0

        delay_pct = 0.0
        if "is_delayed" in grp.columns and not durations.empty:
            delays = grp.loc[status_clean == "delivered", "is_delayed"].dropna()
            delay_pct = round(float(delays.mean() * 100.0), 2) if not delays.empty else 0.0

        return pd.Series({
            "restaurant_type": rest_type,
            "sample_size_orders": n_orders,
            "completed_orders": completed,
            "avg_order_amount": avg_amt,
            "avg_prep_time_min": avg_prep,
            "avg_duration_min": avg_dur,
            "delay_rate_pct": delay_pct
        })

    summary = df.groupby("restaurant_id", group_keys=False).apply(calc_rest_stats).reset_index()
    return summary.sort_values(by="sample_size_orders", ascending=False).head(top_n)


def detect_numeric_outliers(df: pd.DataFrame, target_cols: List[str] = None) -> List[Dict[str, Any]]:
    """
    Detects potential outliers using Tukey's IQR rule:
      Lower Bound = Q1 - 1.5 * IQR
      Upper Bound = Q3 + 1.5 * IQR
    Flags potential outliers rather than automatically deleting them.
    """
    if df.empty:
        return []

    if target_cols is None:
        target_cols = ["order_amount", "item_count", "preparation_time_min", "delivery_distance_km", "delivery_duration_min"]

    outliers_summary = []
    for col in target_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            valid_vals = df[col].dropna()
            if len(valid_vals) < 4:
                continue

            q1 = float(valid_vals.quantile(0.25))
            q3 = float(valid_vals.quantile(0.75))
            iqr = q3 - q1
            lower_bound = round(q1 - 1.5 * iqr, 2)
            upper_bound = round(q3 + 1.5 * iqr, 2)

            outlier_mask = (valid_vals < lower_bound) | (valid_vals > upper_bound)
            count = int(outlier_mask.sum())
            pct = round((count / len(valid_vals) * 100.0), 2)

            outliers_summary.append({
                "column": col,
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": count,
                "outlier_percentage": pct
            })

    return outliers_summary


def inspect_suspicious_records(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifies logically inconsistent or suspicious records in the dataset:
    - Cancelled orders that report a completed delivery duration.
    - Orders where food prep time exceeds total delivery duration.
    - Zero or negative transactional values.
    """
    if df.empty:
        return {"cancelled_with_duration": 0, "prep_exceeds_duration": 0, "non_positive_amount": 0}

    cancelled_with_dur = 0
    if "order_status" in df.columns and "delivery_duration_min" in df.columns:
        mask = (df["order_status"].astype(str).str.lower() == "cancelled") & (df["delivery_duration_min"] > 0)
        cancelled_with_dur = int(mask.sum())

    prep_exceeds_dur = 0
    if "preparation_time_min" in df.columns and "delivery_duration_min" in df.columns:
        mask = (df["preparation_time_min"] > df["delivery_duration_min"]) & (df["delivery_duration_min"] > 0)
        prep_exceeds_dur = int(mask.sum())

    non_pos_amt = 0
    if "order_amount" in df.columns:
        non_pos_amt = int((df["order_amount"] <= 0).sum())

    return {
        "cancelled_with_duration": cancelled_with_dur,
        "prep_exceeds_duration": prep_exceeds_dur,
        "non_positive_amount": non_pos_amt
    }


def apply_dashboard_filters(
    df: pd.DataFrame,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None,
    selected_zones: Optional[List[str]] = None,
    selected_restaurant_types: Optional[List[str]] = None,
    selected_restaurant_ids: Optional[List[str]] = None,
    selected_statuses: Optional[List[str]] = None,
    selected_weather: Optional[List[str]] = None,
    selected_traffic: Optional[List[str]] = None,
    day_type_filter: Optional[str] = "All",
    hour_range: Optional[Tuple[int, int]] = None
) -> pd.DataFrame:
    """
    Safely filters the dataframe based on multiple simultaneous criteria.
    Gracefully adapts to missing columns.
    """
    if df.empty:
        return df

    filtered = df.copy()

    # 1. Date Range Filter
    if "order_timestamp" in filtered.columns and (start_date or end_date):
        dt_series = pd.to_datetime(filtered["order_timestamp"], errors="coerce")
        if start_date:
            filtered = filtered.loc[dt_series.dt.date >= start_date]
            dt_series = dt_series.loc[dt_series.dt.date >= start_date]
        if end_date:
            filtered = filtered.loc[dt_series.dt.date <= end_date]
            dt_series = dt_series.loc[dt_series.dt.date <= end_date]

    # 2. Day of Week / Weekend Filter
    if "order_timestamp" in filtered.columns and day_type_filter and day_type_filter != "All":
        dt_series = pd.to_datetime(filtered["order_timestamp"], errors="coerce")
        is_weekend = dt_series.dt.weekday.isin([5, 6])
        if day_type_filter == "Weekdays (Mon-Fri)":
            filtered = filtered.loc[~is_weekend]
        elif day_type_filter == "Weekends (Sat-Sun)":
            filtered = filtered.loc[is_weekend]

    # 3. Hour of Day Filter
    if "order_timestamp" in filtered.columns and hour_range and len(hour_range) == 2:
        dt_series = pd.to_datetime(filtered["order_timestamp"], errors="coerce")
        filtered = filtered.loc[(dt_series.dt.hour >= hour_range[0]) & (dt_series.dt.hour <= hour_range[1])]

    # 4. Categorical Filters
    filter_mappings = [
        ("delivery_zone", selected_zones),
        ("restaurant_type", selected_restaurant_types),
        ("restaurant_id", selected_restaurant_ids),
        ("order_status", selected_statuses),
        ("weather_condition", selected_weather),
        ("traffic_density", selected_traffic),
    ]

    for col, selected_list in filter_mappings:
        if col in filtered.columns and selected_list:
            filtered = filtered.loc[filtered[col].isin(selected_list)]

    return filtered
