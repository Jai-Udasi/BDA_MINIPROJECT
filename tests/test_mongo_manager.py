"""
Unit tests for MongoManager and Fallback Store.
Ensures 100% test isolation without requiring a live MongoDB daemon.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.nosql.mongo_manager import MongoManager, SQLiteFallbackBackend


@pytest.fixture
def fallback_manager(tmp_path):
    """Provides an isolated MongoManager running in fallback SQLite mode."""
    test_db = tmp_path / "test_fallback.db"
    mgr = MongoManager(force_fallback=True)
    mgr.fallback = SQLiteFallbackBackend(test_db)
    mgr.clear_all()
    return mgr


def test_fallback_health_check_structure(fallback_manager):
    """Verifies diagnostic health check returns complete metadata in fallback mode."""
    health = fallback_manager.health_check()
    assert health["status"] == "OPERATIONAL_FALLBACK"
    assert health["backend"] == "FALLBACK_SQLITE"
    assert health["is_connected"] is False
    assert "collection_counts" in health
    assert health["collection_counts"]["active_orders"] == 0
    assert "ping_latency_ms" in health
    assert "diagnostic_message" in health


def test_active_orders_idempotent_upsert(fallback_manager):
    """Tests inserting and updating active orders without row duplication."""
    order_doc = {
        "order_id": "ORD-TEST-001",
        "customer_id": "CUST-10",
        "restaurant_id": "REST-20",
        "delivery_zone": "Downtown",
        "order_amount": 450.0,
        "status": "In Transit",
        "predicted_delay_risk": 0.42
    }
    
    # First insert
    res1 = fallback_manager.upsert_active_order(order_doc)
    assert res1 is True
    
    orders = fallback_manager.get_active_orders(zone="Downtown")
    assert len(orders) == 1
    assert orders[0]["order_id"] == "ORD-TEST-001"
    assert orders[0]["order_amount"] == 450.0

    # Second upsert (update status and amount)
    order_doc["order_amount"] = 520.0
    order_doc["status"] = "Delivered"
    res2 = fallback_manager.upsert_active_order(order_doc)
    assert res2 is True

    # Count must remain 1 (idempotent)
    updated_orders = fallback_manager.get_active_orders(zone="Downtown")
    assert len(updated_orders) == 1
    assert updated_orders[0]["order_amount"] == 520.0
    assert updated_orders[0]["status"] == "Delivered"


def test_courier_telemetry_and_geojson_formatting(fallback_manager):
    """Tests courier location updates and GeoJSON coordinate handling."""
    telemetry_doc = {
        "courier_id": "COUR-505",
        "latitude": 19.0522,
        "longitude": 72.8994,
        "status": "In Transit",
        "active_order_id": "ORD-TEST-001",
        "battery_pct": 88
    }

    res = fallback_manager.upsert_courier_telemetry(telemetry_doc)
    assert res is True

    records = fallback_manager.get_courier_telemetry(courier_id="COUR-505")
    assert len(records) == 1
    assert records[0]["courier_id"] == "COUR-505"
    assert records[0]["latitude"] == 19.0522
    assert records[0]["location"]["type"] == "Point"
    assert records[0]["location"]["coordinates"] == [72.8994, 19.0522]


def test_zone_risk_scores_upsert_and_sorting(fallback_manager):
    """Tests zone risk score persistence and sorting by risk severity."""
    zones = [
        {"delivery_zone": "Downtown", "risk_score": 0.35, "risk_level": "Low"},
        {"delivery_zone": "East Tech Park", "risk_score": 0.85, "risk_level": "High"},
        {"delivery_zone": "North Suburbs", "risk_score": 0.60, "risk_level": "Medium"}
    ]

    for z in zones:
        fallback_manager.upsert_zone_risk_score(z)

    scores = fallback_manager.get_zone_risk_scores()
    assert len(scores) == 3
    # Must be ordered by risk_score DESC
    assert scores[0]["delivery_zone"] == "East Tech Park"
    assert scores[0]["risk_score"] == 0.85
    assert scores[2]["delivery_zone"] == "Downtown"


def test_validation_errors_on_empty_keys(fallback_manager):
    """Ensures documents with missing primary keys raise ValueError."""
    with pytest.raises(ValueError, match="order_id"):
        fallback_manager.upsert_active_order({"delivery_zone": "Downtown"})

    with pytest.raises(ValueError, match="courier_id"):
        fallback_manager.upsert_courier_telemetry({"latitude": 19.05})

    with pytest.raises(ValueError, match="delivery_zone"):
        fallback_manager.upsert_zone_risk_score({"risk_score": 0.5})


def test_clear_all_resets_store(fallback_manager):
    """Verifies clear_all removes all documents."""
    fallback_manager.upsert_active_order({"order_id": "ORD-TEMP", "delivery_zone": "Downtown"})
    fallback_manager.upsert_courier_telemetry({"courier_id": "COUR-TEMP"})
    fallback_manager.upsert_zone_risk_score({"delivery_zone": "Downtown", "risk_score": 0.2})

    counts_before = fallback_manager.health_check()["collection_counts"]
    assert counts_before["active_orders"] == 1

    fallback_manager.clear_all()
    counts_after = fallback_manager.health_check()["collection_counts"]
    assert counts_after["active_orders"] == 0
    assert counts_after["courier_telemetry"] == 0
    assert counts_after["zone_risk_scores"] == 0


def test_mock_mongodb_connection_flow():
    """Mocks PyMongo to verify real MongoDB connection and index logic when reachable."""
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.__getitem__.return_value = mock_db
    mock_client.admin.command.return_value = {"ok": 1}

    with patch("src.nosql.mongo_manager.MongoClient", return_value=mock_client), \
         patch("src.nosql.mongo_manager.PYMONGO_AVAILABLE", True):
        mgr = MongoManager(force_fallback=False)
        assert mgr.is_connected is True
        assert mgr.backend_name == "MONGODB"
        # Verify index creation was called
        assert mock_db.active_orders.create_index.called
        assert mock_db.courier_telemetry.create_index.called
        assert mock_db.zone_risk_scores.create_index.called
