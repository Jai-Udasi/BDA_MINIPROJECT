"""
Unit tests for Delivery Delay Risk Classifier (Syllabus Module 5).
Verifies:
1. Zero data leakage: strict exclusion of outcome variables (duration, status).
2. Proper training and evaluation on both Logistic Regression and Random Forest.
3. Realistic evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
4. Feature importance extraction and ranking.
5. Model saving, loading, and inference on unseen single order dictionaries.
6. Error handling on missing features or corrupt schemas.
"""
import pytest
import pandas as pd
from pathlib import Path
from src.ml.delay_prediction import (
    DeliveryDelayClassifier,
    EXCLUDED_LEAKAGE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from config.settings import PROCESSED_ORDERS_FILE


@pytest.fixture
def cleaned_df():
    """Loads the actual cleaned order dataset."""
    assert PROCESSED_ORDERS_FILE.exists(), "cleaned_orders.csv must exist"
    return pd.read_csv(PROCESSED_ORDERS_FILE)


def test_zero_data_leakage_in_features(cleaned_df):
    """
    CRITICAL ML INTEGRITY TEST:
    Verifies that no outcome columns (actual duration, status, label) leak into features.
    """
    classifier = DeliveryDelayClassifier()
    X, y = classifier.prepare_features(cleaned_df)

    # Outcome columns must NEVER appear in feature matrix X
    for col in EXCLUDED_LEAKAGE_COLUMNS:
        assert col not in X.columns, f"Data leakage detected! '{col}' was found in features."

    assert len(X) == len(cleaned_df)
    assert len(y) == len(cleaned_df)
    assert set(y.unique()).issubset({0, 1})


def test_delay_classifier_training_and_metrics(cleaned_df):
    """Verify training evaluates both models and computes all prescribed metrics."""
    classifier = DeliveryDelayClassifier(random_state=42)
    summary = classifier.train_and_evaluate(cleaned_df, test_size=0.20)

    assert "Logistic Regression" in summary["models_evaluated"]
    assert "Random Forest" in summary["models_evaluated"]

    for name, metrics in summary["models_evaluated"].items():
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0
        assert 0.5 <= metrics["roc_auc"] <= 1.0  # Should beat random guessing
        assert len(metrics["confusion_matrix"]) == 2  # 2x2 matrix

    # Feature importances should be populated
    assert len(summary["feature_importances_top"]) > 0


def test_delay_model_save_and_predict(cleaned_df, tmp_path):
    """Verify saving fitted pipeline, reloading it, and predicting single order risk."""
    classifier = DeliveryDelayClassifier(random_state=42)
    classifier.train_and_evaluate(cleaned_df)

    temp_model_file = tmp_path / "test_delay_model.joblib"
    classifier.save_model(temp_model_file)
    assert temp_model_file.exists()

    # Load model in new instance
    loaded = DeliveryDelayClassifier.load_model(temp_model_file)
    assert loaded.best_model_name == classifier.best_model_name

    # Test inference on a high-risk scenario: long distance, high prep, congested traffic, rain
    high_risk_order = {
        "delivery_distance_km": 11.5,
        "preparation_time_min": 35.0,
        "item_count": 6,
        "order_hour": 20,
        "weather_condition": "Rainy",
        "traffic_density": "Congested",
        "delivery_zone": "Downtown"
    }
    pred_res = loaded.predict_order_delay(high_risk_order)
    assert pred_res["predicted_is_delayed"] in [0, 1]
    assert 0.0 <= pred_res["delay_probability_pct"] <= 100.0
    assert "risk_classification" in pred_res


def test_delay_classifier_missing_features_error():
    """Verify ValueError is raised if critical features are missing."""
    classifier = DeliveryDelayClassifier()
    corrupt_df = pd.DataFrame({"some_col": [1, 2, 3]})

    with pytest.raises(ValueError):
        classifier.prepare_features(corrupt_df)
