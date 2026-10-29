"""
Time-Series Demand Forecasting (Syllabus Module 5: Big Data Mining & Machine Learning).
Forecasts hourly food delivery order demand per delivery zone for upcoming 1, 3, and 24 hour horizons.

METHODOLOGY & LEAKAGE PREVENTION:
1. Complete Temporal Grid:
   Aggregates raw orders into continuous 1-hour intervals across the full date range.
   Hours with zero orders are explicitly filled with 0 (preventing sparse sample bias).
2. Chronological Train/Test Split:
   Evaluates strictly forward in time (earliest 80% hours for training, most recent 20% for testing).
   Random shuffling is strictly avoided to prevent temporal look-ahead leakage.
3. Feature Engineering (Strictly Past Information):
   - lag_1h: Demand in preceding hour (t-1).
   - lag_2h: Demand two hours ago (t-2).
   - lag_3h: Demand three hours ago (t-3).
   - lag_24h: Demand at the same hour yesterday (t-24, capturing daily seasonality).
   - rolling_mean_3h: Moving average demand over previous 3 hours.
   - hour_of_day: Cyclical hour index (0-23).
   - is_weekend: Binary indicator for weekend demand surges.
4. Benchmarking Against Naive Persistence:
   Compares Ridge Regression against a Naive Baseline (predicting lag_1h or lag_24h),
   reporting Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
"""
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


class HourlyDemandTimeSeriesBuilder:
    """
    Constructs an uninterrupted hourly time series per delivery zone with lag features.
    """
    @staticmethod
    def build_zone_hourly_df(df: pd.DataFrame, zone: str) -> pd.DataFrame:
        """
        Creates a complete hourly time series for the specified zone, filling zero-order hours.
        """
        data = df.copy()
        if "order_timestamp" in data.columns:
            ts_col = "order_timestamp"
        elif "timestamp" in data.columns:
            ts_col = "timestamp"
        else:
            raise ValueError("No timestamp column found in dataset.")

        data["dt"] = pd.to_datetime(data[ts_col], errors="coerce")
        data = data.dropna(subset=["dt"])

        # Filter by zone if specified
        zone_data = data[data["delivery_zone"] == zone].copy()
        if zone_data.empty:
            raise ValueError(f"No order records found for zone '{zone}'.")

        # Floor to nearest hour
        zone_data["hourly_bin"] = zone_data["dt"].dt.floor("h")
        hourly_counts = zone_data.groupby("hourly_bin").size().reset_index(name="order_count")

        # Build complete continuous hourly index from min to max timestamp
        min_hour = hourly_counts["hourly_bin"].min()
        max_hour = hourly_counts["hourly_bin"].max()
        full_time_index = pd.date_range(start=min_hour, end=max_hour, freq="1h", name="hourly_bin")

        complete_df = pd.DataFrame(index=full_time_index).reset_index()
        merged_df = pd.merge(complete_df, hourly_counts, on="hourly_bin", how="left")
        merged_df["order_count"] = merged_df["order_count"].fillna(0).astype(int)
        merged_df["delivery_zone"] = zone

        # Construct time and lag features
        merged_df["hour_of_day"] = merged_df["hourly_bin"].dt.hour
        merged_df["day_of_week"] = merged_df["hourly_bin"].dt.dayofweek
        merged_df["is_weekend"] = merged_df["day_of_week"].isin([5, 6]).astype(int)

        # Lags
        merged_df["lag_1h"] = merged_df["order_count"].shift(1)
        merged_df["lag_2h"] = merged_df["order_count"].shift(2)
        merged_df["lag_3h"] = merged_df["order_count"].shift(3)
        merged_df["lag_24h"] = merged_df["order_count"].shift(24)
        merged_df["rolling_mean_3h"] = merged_df["order_count"].shift(1).rolling(window=3).mean()

        # Drop the warmup rows containing NaN lags
        clean_ts_df = merged_df.dropna().reset_index(drop=True)
        return clean_ts_df


class ZoneDemandForecaster:
    """
    Forecasting model for zone order demand using Ridge Regression and chronological evaluation.
    """
    FEATURE_COLS = ["lag_1h", "lag_2h", "lag_3h", "lag_24h", "rolling_mean_3h", "hour_of_day", "is_weekend"]

    def __init__(self, zone: str = "Downtown", alpha: float = 1.0):
        self.zone: str = zone
        self.alpha: float = alpha
        self.model: Optional[Ridge] = None
        self.evaluation_summary: Optional[Dict[str, Any]] = None

    def train_and_evaluate(
        self,
        ts_df: pd.DataFrame,
        train_ratio: float = 0.80
    ) -> Dict[str, Any]:
        """
        Chronologically splits time series, trains Ridge model, and benchmarks against Naive Baseline.
        """
        if len(ts_df) < 30:
            raise ValueError(f"Time series too short for forecasting ({len(ts_df)} rows).")

        n_train = int(len(ts_df) * train_ratio)
        train_df = ts_df.iloc[:n_train].copy()
        test_df = ts_df.iloc[n_train:].copy()

        X_train = train_df[self.FEATURE_COLS]
        y_train = train_df["order_count"]

        X_test = test_df[self.FEATURE_COLS]
        y_test = test_df["order_count"]

        # 1. Fit Ridge Model
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_train, y_train)

        # Predict with non-negativity constraint
        raw_pred = self.model.predict(X_test)
        y_pred = np.maximum(0.0, raw_pred)

        # 2. Naive Persistence Baseline: Predict lag_1h (demand of preceding hour)
        y_baseline = test_df["lag_1h"].values

        # 3. Compute Metrics
        model_mae = float(mean_absolute_error(y_test, y_pred))
        model_rmse = float(root_mean_squared_error(y_test, y_pred))

        base_mae = float(mean_absolute_error(y_test, y_baseline))
        base_rmse = float(root_mean_squared_error(y_test, y_baseline))

        mae_improvement = ((base_mae - model_mae) / base_mae * 100.0) if base_mae > 0 else 0.0

        summary = {
            "zone": self.zone,
            "total_hourly_periods": len(ts_df),
            "train_periods": len(train_df),
            "test_periods": len(test_df),
            "ridge_model_metrics": {
                "mae": round(model_mae, 2),
                "rmse": round(model_rmse, 2)
            },
            "naive_baseline_metrics": {
                "mae": round(base_mae, 2),
                "rmse": round(base_rmse, 2)
            },
            "mae_improvement_pct": round(mae_improvement, 1),
            "feature_coefficients": {
                feat: round(float(coef), 3)
                for feat, coef in zip(self.FEATURE_COLS, self.model.coef_)
            },
            "recent_actual_vs_predicted": [
                {
                    "timestamp": str(ts),
                    "actual_orders": int(act),
                    "predicted_orders": round(float(prd), 1)
                }
                for ts, act, prd in zip(test_df["hourly_bin"].tail(12), y_test.tail(12), y_pred[-12:])
            ]
        }
        self.evaluation_summary = summary
        return summary

    def forecast_upcoming_horizons(self, ts_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates recursive demand forecasts for the next 1h, 3h, and 24h horizons.
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting.")

        last_known_row = ts_df.iloc[-1].copy()
        current_time = pd.to_datetime(last_known_row["hourly_bin"])

        # History buffer for lag lookups
        recent_history = list(ts_df["order_count"].values)

        forecasts = []
        for step in range(1, 25):  # 24 hours ahead
            f_time = current_time + pd.Timedelta(hours=step)
            f_hour = f_time.hour
            f_weekend = 1 if f_time.dayofweek in [5, 6] else 0

            # Lags relative to current simulated forecast step
            l1 = recent_history[-1]
            l2 = recent_history[-2] if len(recent_history) >= 2 else l1
            l3 = recent_history[-3] if len(recent_history) >= 3 else l2
            l24 = recent_history[-24] if len(recent_history) >= 24 else l1
            roll3 = (l1 + l2 + l3) / 3.0

            feats = pd.DataFrame([{
                "lag_1h": l1,
                "lag_2h": l2,
                "lag_3h": l3,
                "lag_24h": l24,
                "rolling_mean_3h": roll3,
                "hour_of_day": f_hour,
                "is_weekend": f_weekend
            }])[self.FEATURE_COLS]

            pred_val = float(max(0.0, self.model.predict(feats)[0]))
            recent_history.append(pred_val)

            forecasts.append({
                "horizon_step_hours": step,
                "forecast_time": f_time.strftime("%Y-%m-%d %H:00"),
                "predicted_order_volume": round(pred_val, 1)
            })

        # Find peak predicted periods
        peak_period = max(forecasts, key=lambda x: x["predicted_order_volume"])

        return {
            "zone": self.zone,
            "next_1h_forecast": forecasts[0]["predicted_order_volume"],
            "next_3h_cumulative": round(sum(f["predicted_order_volume"] for f in forecasts[:3]), 1),
            "next_24h_cumulative": round(sum(f["predicted_order_volume"] for f in forecasts), 1),
            "peak_predicted_period": peak_period,
            "hourly_24h_breakdown": forecasts
        }
