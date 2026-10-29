"""
Delivery Delay Risk Classifier (Syllabus Module 5: Big Data Mining & Machine Learning).
Predicts whether an order will breach its Service Level Agreement (SLA) threshold.

DATA LEAKAGE PREVENTION GUARANTEES:
1. Target Variable:
   is_delayed (binary 1 or 0), derived from SLA rule: expected_duration = 35 + 2.5 * distance_km.
2. Excluded Outcome Fields (NEVER used as features):
   - delivery_duration_min (the actual duration contains the label!)
   - order_status (completed vs cancelled occurs post-delivery)
   - is_delayed (the label itself)
3. Preprocessing Hygiene:
   - Scalers and One-Hot Encoders are fit EXCLUSIVELY on the training split (X_train).
   - Test split (X_test) is transformed using the fitted training pipeline without re-fitting.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "delay_classifier.joblib"

# Feature definitions
NUMERICAL_FEATURES = ["delivery_distance_km", "preparation_time_min", "item_count", "order_hour"]
CATEGORICAL_FEATURES = ["weather_condition", "traffic_density", "delivery_zone"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_LEAKAGE_COLUMNS = ["delivery_duration_min", "order_status", "is_delayed", "order_id", "timestamp", "order_timestamp"]


class DeliveryDelayClassifier:
    """
    Supervised Machine Learning pipeline for delivery delay prediction.
    """
    def __init__(self, random_state: int = 42):
        self.random_state: int = random_state
        self.pipeline: Optional[Pipeline] = None
        self.best_model_name: Optional[str] = None
        self.evaluation_results: Optional[Dict[str, Any]] = None
        self.feature_names_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extracts valid prediction features and target, preventing any data leakage.
        """
        data = df.copy()

        # Derive order_hour if not present
        if "order_hour" not in data.columns and "order_timestamp" in data.columns:
            ts = pd.to_datetime(data["order_timestamp"], errors="coerce")
            data["order_hour"] = ts.dt.hour.fillna(12).astype(int)
        elif "order_hour" not in data.columns and "timestamp" in data.columns:
            ts = pd.to_datetime(data["timestamp"], errors="coerce")
            data["order_hour"] = ts.dt.hour.fillna(12).astype(int)
        elif "order_hour" not in data.columns:
            data["order_hour"] = 12

        # Check required features
        for col in ALL_FEATURES:
            if col not in data.columns:
                raise ValueError(f"Required input feature '{col}' is missing from dataset.")

        if "is_delayed" not in data.columns:
            raise ValueError("Target column 'is_delayed' missing from dataset.")

        # Strict feature extraction
        X = data[ALL_FEATURES].copy()
        y = data["is_delayed"].astype(int).copy()

        # Impute any missing numerical values with medians, categoricals with mode
        for col in NUMERICAL_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype(str).fillna("Unknown")

        return X, y

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20
    ) -> Dict[str, Any]:
        """
        Trains and compares Logistic Regression and Random Forest models with zero data leakage.
        """
        X, y = self.prepare_features(df)

        # Stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Preprocessing: StandardScaler for numericals, OneHotEncoder for categoricals
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERICAL_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
            ]
        )

        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=self.random_state,
                class_weight="balanced"
            )
        }

        results = {}
        best_f1 = -1.0
        best_pipeline = None
        best_name = ""

        for name, clf in models.items():
            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf)
            ])

            # Fit ONLY on training data
            pipe.fit(X_train, y_train)

            # Predict on test data
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            try:
                auc = float(roc_auc_score(y_test, y_prob))
            except Exception:
                auc = 0.5

            cm = confusion_matrix(y_test, y_pred).tolist()

            results[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc, 4),
                "confusion_matrix": cm  # [[TN, FP], [FN, TP]]
            }

            if f1 > best_f1:
                best_f1 = f1
                best_pipeline = pipe
                best_name = name

        self.pipeline = best_pipeline
        self.best_model_name = best_name
        self.evaluation_results = results

        # Extract feature importances if Random Forest
        rf_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", models["Random Forest"])])
        rf_pipe.fit(X_train, y_train)
        fitted_prep = rf_pipe.named_steps["preprocessor"]
        ohe = fitted_prep.named_transformers_["cat"]
        encoded_cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        feature_names = NUMERICAL_FEATURES + encoded_cat_names
        self.feature_names_ = feature_names

        rf_clf = rf_pipe.named_steps["classifier"]
        importances = rf_clf.feature_importances_
        fi_dict = {name: round(float(imp), 4) for name, imp in zip(feature_names, importances)}
        self.feature_importances_ = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))

        return {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "delay_class_distribution_train": dict(pd.Series(y_train).value_counts()),
            "delay_class_distribution_test": dict(pd.Series(y_test).value_counts()),
            "models_evaluated": results,
            "best_model": best_name,
            "feature_importances_top": dict(list(self.feature_importances_.items())[:8])
        }

    def save_model(self, file_path: Optional[Path] = None):
        """Saves the fitted pipeline to disk using joblib."""
        if self.pipeline is None:
            raise ValueError("No model has been trained yet to save.")
        target_path = file_path or DEFAULT_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "best_model_name": self.best_model_name,
            "evaluation_results": self.evaluation_results,
            "feature_importances": self.feature_importances_
        }, target_path)

    @classmethod
    def load_model(cls, file_path: Optional[Path] = None) -> "DeliveryDelayClassifier":
        """Loads a previously fitted model pipeline from disk."""
        target_path = file_path or DEFAULT_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at {target_path}")

        payload = joblib.load(target_path)
        instance = cls()
        instance.pipeline = payload["pipeline"]
        instance.best_model_name = payload["best_model_name"]
        instance.evaluation_results = payload["evaluation_results"]
        instance.feature_importances_ = payload.get("feature_importances", {})
        return instance

    def predict_order_delay(self, order_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts raw order features, validates inputs, and predicts delay risk.
        """
        if self.pipeline is None:
            raise ValueError("Model pipeline is not fitted or loaded.")

        row_df = pd.DataFrame([order_dict])
        if "order_hour" not in row_df.columns:
            row_df["order_hour"] = 12

        # Filter and validate features
        X_infer = row_df[ALL_FEATURES].copy()
        for col in NUMERICAL_FEATURES:
            X_infer[col] = pd.to_numeric(X_infer[col], errors="coerce").fillna(0.0)
        for col in CATEGORICAL_FEATURES:
            X_infer[col] = X_infer[col].astype(str).fillna("Unknown")

        prob = float(self.pipeline.predict_proba(X_infer)[0, 1])
        prediction = int(prob >= 0.50)

        if prob >= 0.75:
            risk = "Critical SLA Breach Risk"
        elif prob >= 0.50:
            risk = "Elevated Delay Probability"
        elif prob >= 0.25:
            risk = "Moderate (Acceptable)"
        else:
            risk = "Low Risk (On-Time Delivery Expected)"

        return {
            "predicted_is_delayed": prediction,
            "delay_probability_pct": round(prob * 100.0, 1),
            "risk_classification": risk,
            "model_used": self.best_model_name
        }


def train_and_save_delay_model(df: pd.DataFrame) -> Dict[str, Any]:
    """Helper function training and serializing the model."""
    classifier = DeliveryDelayClassifier(random_state=42)
    summary = classifier.train_and_evaluate(df)
    classifier.save_model()
    return summary
