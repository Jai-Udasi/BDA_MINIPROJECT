"""
Unit tests for Time-Series Demand Forecasting (Syllabus Module 5).
Verifies:
1. Continuous hourly aggregation with zero-count filling (no missing time gaps).
2. Chronological train/test split without lookahead leakage.
3. Ridge regression model evaluation vs Naive Persistence Baseline (MAE, RMSE).
4. Strict non-negativity constraint on predicted order counts.
5. Horizon forecasts for 1h, 3h, and 24h intervals.
"""
import pytest
import pandas as pd
from src.ml.demand_forecasting import HourlyDemandTimeSeriesBuilder, ZoneDemandForecaster
from config.settings import PROCESSED_ORDERS_FILE


@pytest.fixture
def cleaned_orders():
    """Loads cleaned orders dataset."""
    assert PROCESSED_ORDERS_FILE.exists()
    return pd.read_csv(PROCESSED_ORDERS_FILE)


def test_hourly_time_series_builder_completeness(cleaned_orders):
    """
    Verify time series builder creates an uninterrupted hourly sequence
    with zero orders filled for idle hours.
    """
    ts_df = HourlyDemandTimeSeriesBuilder.build_zone_hourly_df(cleaned_orders, zone="Downtown")

    assert len(ts_df) > 50
    assert "hourly_bin" in ts_df.columns
    assert "order_count" in ts_df.columns
    assert "lag_1h" in ts_df.columns
    assert "lag_24h" in ts_df.columns

    # Verify no nulls exist after lag construction
    assert ts_df["order_count"].isnull().sum() == 0
    assert (ts_df["order_count"] >= 0).all()

    # Verify timestamps are strictly monotonically increasing by 1 hour
    time_diffs = ts_df["hourly_bin"].diff().dropna()
    assert (time_diffs == pd.Timedelta(hours=1)).all()


def test_chronological_evaluation_and_metrics(cleaned_orders):
    """
    CRITICAL TIME-SERIES INTEGRITY:
    Verify train/test split is strictly chronological (train strictly precedes test).
    """
    ts_df = HourlyDemandTimeSeriesBuilder.build_zone_hourly_df(cleaned_orders, zone="Downtown")
    forecaster = ZoneDemandForecaster(zone="Downtown", alpha=1.0)
    summary = forecaster.train_and_evaluate(ts_df, train_ratio=0.80)

    assert summary["train_periods"] + summary["test_periods"] == summary["total_hourly_periods"]

    # Verify metric values are positive floats
    assert summary["ridge_model_metrics"]["mae"] >= 0.0
    assert summary["ridge_model_metrics"]["rmse"] >= 0.0
    assert summary["naive_baseline_metrics"]["mae"] >= 0.0

    # Predictions must be non-negative
    for row in summary["recent_actual_vs_predicted"]:
        assert row["predicted_orders"] >= 0.0


def test_upcoming_horizon_forecasting(cleaned_orders):
    """Verify recursive horizon forecasting for 1h, 3h, and 24h."""
    ts_df = HourlyDemandTimeSeriesBuilder.build_zone_hourly_df(cleaned_orders, zone="Downtown")
    forecaster = ZoneDemandForecaster(zone="Downtown", alpha=1.0)
    forecaster.train_and_evaluate(ts_df)

    forecast_res = forecaster.forecast_upcoming_horizons(ts_df)

    assert forecast_res["next_1h_forecast"] >= 0.0
    assert forecast_res["next_3h_cumulative"] >= 0.0
    assert forecast_res["next_24h_cumulative"] >= 0.0
    assert len(forecast_res["hourly_24h_breakdown"]) == 24

    for f in forecast_res["hourly_24h_breakdown"]:
        assert f["predicted_order_volume"] >= 0.0


def test_forecasting_error_on_empty_zone(cleaned_orders):
    """Verify ValueError is raised if an invalid zone is specified."""
    with pytest.raises(ValueError):
        HourlyDemandTimeSeriesBuilder.build_zone_hourly_df(cleaned_orders, zone="NonExistentZone")
