"""
Data Preprocessing and Quality Verification Pipeline.
Handles data loading, schema inspection, deduplication, missing value imputation,
anomaly correction, and quality auditing.
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Add project root to sys.path if executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import (
    RAW_ORDERS_FILE,
    PROCESSED_ORDERS_FILE,
    DATA_QUALITY_REPORT_FILE,
    NUMERIC_RANGES,
    DELIVERY_ZONES,
    RESTAURANT_TYPES,
    WEATHER_CONDITIONS,
    TRAFFIC_LEVELS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataPreprocessing")


def load_dataset(file_path: Path | str) -> pd.DataFrame:
    """
    Safely loads a CSV dataset with error handling and logging.

    Parameters:
    -----------
    file_path : Path | str
        Path to the dataset CSV file.

    Returns:
    --------
    pd.DataFrame
        Loaded dataset.
    """
    path = Path(file_path)
    if not path.exists():
        err_msg = f"Dataset file not found at: {path.resolve()}. Please ensure data has been generated or placed there."
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"Loading dataset from: {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns.")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file at {path}: {str(e)}")
        raise


def inspect_schema_and_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects dataset schema, missingness, duplicates, and data boundaries.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe to inspect.

    Returns:
    --------
    Dict[str, Any]
        Data quality audit dictionary.
    """
    total_records = int(len(df))
    missing_by_col = {col: int(count) for col, count in df.isnull().sum().items()}
    total_missing = sum(missing_by_col.values())
    
    # Check duplicate rows and duplicate order IDs
    exact_duplicates = int(df.duplicated().sum())
    duplicate_order_ids = int(df.duplicated(subset=["order_id"]).sum()) if "order_id" in df.columns else 0

    # Date range inspection if available
    date_range = {"start": None, "end": None}
    if "order_timestamp" in df.columns:
        parsed_dates = pd.to_datetime(df["order_timestamp"], errors="coerce")
        valid_dates = parsed_dates.dropna()
        if not valid_dates.empty:
            date_range["start"] = str(valid_dates.min())
            date_range["end"] = str(valid_dates.max())

    schema_info = {col: str(dtype) for col, dtype in df.dtypes.items()}

    summary = {
        "total_records": total_records,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "schema": schema_info,
        "total_missing_values": total_missing,
        "missing_per_column": missing_by_col,
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_order_ids": duplicate_order_ids,
        "date_range": date_range,
        "memory_usage_kb": round(float(df.memory_usage(deep=True).sum() / 1024), 2)
    }
    return summary


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans and standardizes the dataset following documented Big Data veracity rules.
    Never silently discards data; records all modifications in an audit report.

    Rules applied:
    1. Deduplication: Removes duplicate records based on `order_id` (keeping first occurrence).
    2. Datetime standardisation: Trims string whitespace and parses `order_timestamp` to ISO datetime.
    3. Categorical imputation: Imputes missing restaurant_type and weather_condition using statistical mode.
    4. Numeric validation: Corrects negative order amounts and zero item counts using medians.
    5. Operational field imputation: Imputes missing preparation times and distances using group medians.
    6. SLA validation: Recalculates `is_delayed` flag to guarantee internal consistency.

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned dataframe and a comprehensive data-quality transformations report.
    """
    initial_count = len(df)
    logger.info(f"Starting data cleaning pipeline for {initial_count} records...")
    
    # Work on an isolated copy
    cleaned = df.copy()
    report: Dict[str, Any] = {
        "initial_record_count": initial_count,
        "duplicates_removed": 0,
        "invalid_timestamps_removed": 0,
        "anomalies_corrected": {},
        "missing_values_imputed": {},
        "final_record_count": 0,
        "retention_rate_pct": 0.0
    }

    # Step 1: Deduplication based on order_id if present, else exact row
    if "order_id" in cleaned.columns:
        dupes_count = int(cleaned.duplicated(subset=["order_id"]).sum())
        cleaned = cleaned.drop_duplicates(subset=["order_id"], keep="first")
    else:
        dupes_count = int(cleaned.duplicated().sum())
        cleaned = cleaned.drop_duplicates(keep="first")
    
    report["duplicates_removed"] = dupes_count
    logger.info(f"Removed {dupes_count} duplicate records.")

    # Step 2: Timestamp Parsing and Trimming
    if "order_timestamp" in cleaned.columns:
        cleaned["order_timestamp"] = cleaned["order_timestamp"].astype(str).str.strip()
        cleaned["order_timestamp"] = pd.to_datetime(cleaned["order_timestamp"], errors="coerce")
        
        # Check for unparseable timestamps
        invalid_dates = cleaned["order_timestamp"].isna()
        invalid_date_count = int(invalid_dates.sum())
        if invalid_date_count > 0:
            cleaned = cleaned.loc[~invalid_dates].copy()
            report["invalid_timestamps_removed"] = invalid_date_count
            logger.warning(f"Dropped {invalid_date_count} records with unparseable timestamps.")

    # Step 3: Categorical Fields Imputation & Sanitization
    cat_defaults = {
        "restaurant_type": "Fast Food",
        "delivery_zone": "Downtown",
        "weather_condition": "Clear",
        "traffic_density": "Medium",
        "order_status": "Delivered"
    }
    for col, default_val in cat_defaults.items():
        if col in cleaned.columns:
            null_count = int(cleaned[col].isna().sum())
            if null_count > 0:
                mode_val = cleaned[col].mode().iloc[0] if not cleaned[col].mode().empty else default_val
                cleaned[col] = cleaned[col].fillna(mode_val)
                report["missing_values_imputed"][col] = null_count
                logger.info(f"Imputed {null_count} missing values in '{col}' with '{mode_val}'.")

    # Step 4: Numeric Anomaly Validation & Range Enforcement
    # A. Order Amount
    if "order_amount" in cleaned.columns:
        invalid_amount_mask = (cleaned["order_amount"] <= 0) | cleaned["order_amount"].isna()
        anomaly_count = int(invalid_amount_mask.sum())
        if anomaly_count > 0:
            valid_median = float(cleaned.loc[~invalid_amount_mask, "order_amount"].median())
            cleaned.loc[invalid_amount_mask, "order_amount"] = valid_median
            report["anomalies_corrected"]["order_amount"] = anomaly_count
            logger.info(f"Corrected {anomaly_count} non-positive/missing order amounts using median ({valid_median}).")

    # B. Item Count
    if "item_count" in cleaned.columns:
        invalid_items_mask = (cleaned["item_count"] <= 0) | cleaned["item_count"].isna()
        anomaly_count = int(invalid_items_mask.sum())
        if anomaly_count > 0:
            valid_median = int(cleaned.loc[~invalid_items_mask, "item_count"].median())
            cleaned.loc[invalid_items_mask, "item_count"] = valid_median
            report["anomalies_corrected"]["item_count"] = anomaly_count
            logger.info(f"Corrected {anomaly_count} invalid item counts using median ({valid_median}).")

    # C. Preparation Time
    if "preparation_time_min" in cleaned.columns:
        invalid_prep_mask = (cleaned["preparation_time_min"] <= 0) | cleaned["preparation_time_min"].isna()
        anomaly_count = int(invalid_prep_mask.sum())
        if anomaly_count > 0:
            # Impute by restaurant_type median if possible
            if "restaurant_type" in cleaned.columns:
                cleaned["preparation_time_min"] = cleaned.groupby("restaurant_type")["preparation_time_min"].transform(
                    lambda s: s.fillna(s.median())
                )
            overall_median = float(cleaned["preparation_time_min"].median())
            cleaned["preparation_time_min"] = cleaned["preparation_time_min"].fillna(overall_median)
            report["missing_values_imputed"]["preparation_time_min"] = anomaly_count
            logger.info(f"Imputed {anomaly_count} preparation times using restaurant-type medians.")

    # D. Delivery Distance
    if "delivery_distance_km" in cleaned.columns:
        invalid_dist_mask = (cleaned["delivery_distance_km"] <= 0) | cleaned["delivery_distance_km"].isna()
        anomaly_count = int(invalid_dist_mask.sum())
        if anomaly_count > 0:
            if "delivery_zone" in cleaned.columns:
                cleaned["delivery_distance_km"] = cleaned.groupby("delivery_zone")["delivery_distance_km"].transform(
                    lambda s: s.fillna(s.median())
                )
            overall_median = float(cleaned["delivery_distance_km"].median())
            cleaned["delivery_distance_km"] = cleaned["delivery_distance_km"].fillna(overall_median)
            report["missing_values_imputed"]["delivery_distance_km"] = anomaly_count
            logger.info(f"Imputed {anomaly_count} delivery distances using zone medians.")

    # E. Delivery Duration & SLA Validation
    if "delivery_duration_min" in cleaned.columns:
        invalid_dur_mask = (cleaned["delivery_duration_min"] <= 0) | cleaned["delivery_duration_min"].isna()
        anomaly_count = int(invalid_dur_mask.sum())
        if anomaly_count > 0:
            # Reconstruct duration from prep time + distance estimate
            prep = cleaned.get("preparation_time_min", pd.Series(20.0, index=cleaned.index))
            dist = cleaned.get("delivery_distance_km", pd.Series(5.0, index=cleaned.index))
            estimated_duration = prep + (dist * 2.8) + 5.0
            cleaned.loc[invalid_dur_mask, "delivery_duration_min"] = estimated_duration.loc[invalid_dur_mask].round(1)
            report["missing_values_imputed"]["delivery_duration_min"] = anomaly_count
            logger.info(f"Imputed {anomaly_count} missing delivery durations from operational formula.")

    # Recalculate is_delayed flag to maintain internal consistency
    if "delivery_duration_min" in cleaned.columns and "delivery_distance_km" in cleaned.columns:
        sla_benchmark = 35.0 + (cleaned["delivery_distance_km"] * 2.5)
        cleaned["is_delayed"] = (cleaned["delivery_duration_min"] > sla_benchmark).astype(int)

    final_count = len(cleaned)
    report["final_record_count"] = final_count
    report["retention_rate_pct"] = round((final_count / initial_count) * 100.0, 2) if initial_count > 0 else 0.0

    logger.info(f"Cleaning complete: {final_count}/{initial_count} records retained ({report['retention_rate_pct']}%).")
    return cleaned, report


def save_cleaned_data(
    df: pd.DataFrame,
    output_csv_path: Path = PROCESSED_ORDERS_FILE,
    report: Dict[str, Any] = None,
    output_report_path: Path = DATA_QUALITY_REPORT_FILE
) -> None:
    """
    Saves cleaned dataframe and JSON transformation audit report.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Cleaned dataset saved to: {output_csv_path}")

    if report is not None:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Data quality report saved to: {output_report_path}")


def run_pipeline(input_path: Path = RAW_ORDERS_FILE) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full load -> inspect -> clean -> save pipeline.
    """
    raw_df = load_dataset(input_path)
    initial_stats = inspect_schema_and_quality(raw_df)
    logger.info(f"Initial Dataset Quality: {initial_stats['total_missing_values']} missing, {initial_stats['exact_duplicate_rows']} duplicates.")
    
    cleaned_df, audit_report = clean_dataset(raw_df)
    audit_report["initial_inspection"] = initial_stats
    
    save_cleaned_data(cleaned_df, PROCESSED_ORDERS_FILE, audit_report, DATA_QUALITY_REPORT_FILE)
    return cleaned_df, audit_report


if __name__ == "__main__":
    logger.info("Executing standalone data cleaning pipeline...")
    cleaned_data, audit = run_pipeline()
    print("Pipeline Execution Complete!")
    print(f"- Initial records: {audit['initial_record_count']}")
    print(f"- Duplicates removed: {audit['duplicates_removed']}")
    print(f"- Final cleaned records: {audit['final_record_count']} ({audit['retention_rate_pct']}% retained)")
    print(f"- Imputations performed: {audit['missing_values_imputed']}")
    print(f"- Anomalies corrected: {audit['anomalies_corrected']}")
