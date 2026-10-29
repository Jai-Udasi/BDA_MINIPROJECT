"""
MongoDB Operational Store and Local Fallback Manager.
Module: Syllabus Module 2 (NoSQL) and Module 6 Integration.

Provides unified persistence for:
1. active_orders: Live in-flight order state, zone, amounts, predicted delay risk.
2. courier_telemetry: Real-time courier coordinates, battery, and dispatch status.
3. zone_risk_scores: Calculated zone risk indicators and alert levels.

Architecture:
- Checks environment variables MONGODB_URI and MONGODB_DB_NAME.
- Attempts fast connection to MongoDB with bounded timeout (1500ms).
- If MongoDB is available: uses PyMongo with indexes and GeoJSON 2dsphere geometry.
- If MongoDB is unavailable: gracefully activates an embedded SQLite fallback store
  with identical public method signatures and JSON serialization.
- Health-check explicitly distinguishes between MONGODB and FALLBACK_SQLITE.
"""
import os
import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

# Optional PyMongo import
try:
    import pymongo
    from pymongo import MongoClient, ASCENDING, DESCENDING, GEOSPHERE
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    MongoClient = None

from config.settings import PROCESSED_DATA_DIR


class SQLiteFallbackBackend:
    """
    Self-contained SQLite fallback engine ensuring the application and tests
    function smoothly on systems without a running MongoDB daemon.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_orders (
                    order_id TEXT PRIMARY KEY,
                    delivery_zone TEXT,
                    status TEXT,
                    order_timestamp TEXT,
                    order_amount REAL,
                    predicted_delay_risk REAL,
                    payload_json TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_orders_zone 
                ON active_orders (delivery_zone)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS courier_telemetry (
                    courier_id TEXT PRIMARY KEY,
                    latitude REAL,
                    longitude REAL,
                    status TEXT,
                    active_order_id TEXT,
                    last_ping TEXT,
                    payload_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS zone_risk_scores (
                    delivery_zone TEXT PRIMARY KEY,
                    risk_score REAL,
                    risk_level TEXT,
                    calculated_at TEXT,
                    payload_json TEXT
                )
            """)
            conn.commit()

    def upsert_order(self, doc: Dict[str, Any]) -> bool:
        order_id = str(doc.get("order_id", "")).strip()
        if not order_id:
            raise ValueError("Document must contain a non-empty 'order_id'")

        zone = doc.get("delivery_zone", "Unknown")
        status = doc.get("status", "In Transit")
        timestamp = doc.get("order_timestamp", datetime.now(timezone.utc).isoformat())
        amount = float(doc.get("order_amount", 0.0))
        predicted_risk = float(doc.get("predicted_delay_risk", 0.0))
        updated_at = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(doc, default=str)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO active_orders (order_id, delivery_zone, status, order_timestamp, order_amount, predicted_delay_risk, payload_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    delivery_zone=excluded.delivery_zone,
                    status=excluded.status,
                    order_amount=excluded.order_amount,
                    predicted_delay_risk=excluded.predicted_delay_risk,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
            """, (order_id, zone, status, timestamp, amount, predicted_risk, payload, updated_at))
            conn.commit()
        return True

    def get_orders(self, limit: int = 50, zone: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if zone:
                cursor = conn.execute(
                    "SELECT payload_json FROM active_orders WHERE delivery_zone = ? ORDER BY updated_at DESC LIMIT ?",
                    (zone, limit)
                )
            else:
                cursor = conn.execute(
                    "SELECT payload_json FROM active_orders ORDER BY updated_at DESC LIMIT ?",
                    (limit,)
                )
            rows = cursor.fetchall()
            return [json.loads(row["payload_json"]) for row in rows]

    def upsert_telemetry(self, doc: Dict[str, Any]) -> bool:
        courier_id = str(doc.get("courier_id", "")).strip()
        if not courier_id:
            raise ValueError("Document must contain a non-empty 'courier_id'")

        lat = float(doc.get("latitude", 0.0))
        lng = float(doc.get("longitude", 0.0))
        status = doc.get("status", "Idle")
        active_order = doc.get("active_order_id")
        last_ping = doc.get("last_ping", datetime.now(timezone.utc).isoformat())
        payload = json.dumps(doc, default=str)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO courier_telemetry (courier_id, latitude, longitude, status, active_order_id, last_ping, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(courier_id) DO UPDATE SET
                    latitude=excluded.latitude,
                    longitude=excluded.longitude,
                    status=excluded.status,
                    active_order_id=excluded.active_order_id,
                    last_ping=excluded.last_ping,
                    payload_json=excluded.payload_json
            """, (courier_id, lat, lng, status, active_order, last_ping, payload))
            conn.commit()
        return True

    def get_telemetry(self, courier_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if courier_id:
                cursor = conn.execute(
                    "SELECT payload_json FROM courier_telemetry WHERE courier_id = ?",
                    (courier_id,)
                )
            else:
                cursor = conn.execute("SELECT payload_json FROM courier_telemetry ORDER BY last_ping DESC")
            rows = cursor.fetchall()
            return [json.loads(row["payload_json"]) for row in rows]

    def upsert_zone_risk(self, doc: Dict[str, Any]) -> bool:
        zone = str(doc.get("delivery_zone", "")).strip()
        if not zone:
            raise ValueError("Document must contain a non-empty 'delivery_zone'")

        risk_score = float(doc.get("risk_score", 0.0))
        risk_level = doc.get("risk_level", "Low")
        calc_at = doc.get("calculated_at", datetime.now(timezone.utc).isoformat())
        payload = json.dumps(doc, default=str)

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO zone_risk_scores (delivery_zone, risk_score, risk_level, calculated_at, payload_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(delivery_zone) DO UPDATE SET
                    risk_score=excluded.risk_score,
                    risk_level=excluded.risk_level,
                    calculated_at=excluded.calculated_at,
                    payload_json=excluded.payload_json
            """, (zone, risk_score, risk_level, calc_at, payload))
            conn.commit()
        return True

    def get_zone_risks(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT payload_json FROM zone_risk_scores ORDER BY risk_score DESC")
            rows = cursor.fetchall()
            return [json.loads(row["payload_json"]) for row in rows]

    def counts(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            orders_c = conn.execute("SELECT COUNT(*) FROM active_orders").fetchone()[0]
            courier_c = conn.execute("SELECT COUNT(*) FROM courier_telemetry").fetchone()[0]
            risk_c = conn.execute("SELECT COUNT(*) FROM zone_risk_scores").fetchone()[0]
            return {
                "active_orders": int(orders_c),
                "courier_telemetry": int(courier_c),
                "zone_risk_scores": int(risk_c)
            }

    def clear(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM active_orders")
            conn.execute("DELETE FROM courier_telemetry")
            conn.execute("DELETE FROM zone_risk_scores")
            conn.commit()


class MongoManager:
    """
    Unified MongoDB & Fallback Persistence Manager.
    Connects to real MongoDB instance when available; automatically
    falls back to a local SQLite persistent store when offline.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        timeout_ms: int = 1500,
        force_fallback: bool = False
    ):
        self.uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", "food_delivery_bigdata")
        self.timeout_ms = timeout_ms
        self.force_fallback = force_fallback

        self.client: Optional[Any] = None
        self.db: Optional[Any] = None
        self.is_connected = False
        self.backend_name = "UNINITIALIZED"
        self.last_error: Optional[str] = None

        fallback_db_path = PROCESSED_DATA_DIR / "nosql_fallback.db"
        self.fallback = SQLiteFallbackBackend(fallback_db_path)

        self._connect()

    def _connect(self) -> None:
        """Attempts connection to MongoDB; activates fallback on failure."""
        if self.force_fallback or not PYMONGO_AVAILABLE:
            self._activate_fallback("Fallback forced or PyMongo unavailable")
            return

        try:
            # Short timeout to avoid blocking dashboard startup
            start_t = time.perf_counter()
            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=self.timeout_ms,
                connectTimeoutMS=self.timeout_ms
            )
            # The ismaster/ping command is fast and forces a connection check
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self._ensure_indexes()
            self.is_connected = True
            self.backend_name = "MONGODB"
            self.last_error = None
        except Exception as e:
            self._activate_fallback(f"MongoDB connection failed: {str(e)}")

    def _activate_fallback(self, reason: str) -> None:
        self.is_connected = False
        self.client = None
        self.db = None
        self.backend_name = "FALLBACK_SQLITE"
        self.last_error = reason

    def _ensure_indexes(self) -> None:
        """Builds proper indexing for MongoDB collections."""
        if self.db is None:
            return
        try:
            # Active orders indexes
            self.db.active_orders.create_index([("order_id", ASCENDING)], unique=True)
            self.db.active_orders.create_index([("delivery_zone", ASCENDING)])
            self.db.active_orders.create_index([("order_timestamp", DESCENDING)])

            # Courier telemetry indexes
            self.db.courier_telemetry.create_index([("courier_id", ASCENDING)], unique=True)
            self.db.courier_telemetry.create_index([("location", GEOSPHERE)])

            # Zone risk scores indexes
            self.db.zone_risk_scores.create_index([("delivery_zone", ASCENDING)], unique=True)
            self.db.zone_risk_scores.create_index([("calculated_at", DESCENDING)])
        except Exception as e:
            # Non-fatal if index creation encounters syntax or permission issue
            pass

    # =========================================================================
    # Active Orders API
    # =========================================================================
    def upsert_active_order(self, order_dict: Dict[str, Any]) -> bool:
        """
        Idempotent upsert of an active order into the operational store.
        """
        if "order_id" not in order_dict or not str(order_dict["order_id"]).strip():
            raise ValueError("Order document missing mandatory 'order_id'")

        doc = dict(order_dict)
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()

        if self.is_connected and self.db is not None:
            try:
                self.db.active_orders.update_one(
                    {"order_id": doc["order_id"]},
                    {"$set": doc},
                    upsert=True
                )
                return True
            except Exception as e:
                self.last_error = f"MongoDB upsert error: {e}"
                # Fall through to fallback on runtime error
                return self.fallback.upsert_order(doc)
        else:
            return self.fallback.upsert_order(doc)

    def get_active_orders(self, limit: int = 50, zone: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves most recent active orders, optionally filtered by zone."""
        if self.is_connected and self.db is not None:
            try:
                query = {"delivery_zone": zone} if zone else {}
                cursor = self.db.active_orders.find(
                    query, {"_id": 0}
                ).sort("updated_at", DESCENDING).limit(limit)
                return list(cursor)
            except Exception as e:
                self.last_error = f"MongoDB retrieval error: {e}"
                return self.fallback.get_orders(limit=limit, zone=zone)
        return self.fallback.get_orders(limit=limit, zone=zone)

    # =========================================================================
    # Courier Telemetry API
    # =========================================================================
    def upsert_courier_telemetry(self, telemetry_dict: Dict[str, Any]) -> bool:
        """
        Idempotent upsert of real-time courier GPS and dispatch status.
        Adds GeoJSON 'location' Point if latitude and longitude are supplied.
        """
        if "courier_id" not in telemetry_dict or not str(telemetry_dict["courier_id"]).strip():
            raise ValueError("Telemetry document missing mandatory 'courier_id'")

        doc = dict(telemetry_dict)
        doc["last_ping"] = doc.get("last_ping") or datetime.now(timezone.utc).isoformat()

        lat = doc.get("latitude")
        lng = doc.get("longitude")
        if lat is not None and lng is not None:
            try:
                # Standard GeoJSON format: [longitude, latitude]
                doc["location"] = {
                    "type": "Point",
                    "coordinates": [float(lng), float(lat)]
                }
            except (ValueError, TypeError):
                pass

        if self.is_connected and self.db is not None:
            try:
                self.db.courier_telemetry.update_one(
                    {"courier_id": doc["courier_id"]},
                    {"$set": doc},
                    upsert=True
                )
                return True
            except Exception as e:
                self.last_error = f"MongoDB telemetry error: {e}"
                return self.fallback.upsert_telemetry(doc)
        return self.fallback.upsert_telemetry(doc)

    def get_courier_telemetry(self, courier_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves latest courier positions and availability statuses."""
        if self.is_connected and self.db is not None:
            try:
                query = {"courier_id": courier_id} if courier_id else {}
                cursor = self.db.courier_telemetry.find(query, {"_id": 0}).sort("last_ping", DESCENDING)
                return list(cursor)
            except Exception as e:
                self.last_error = f"MongoDB retrieval error: {e}"
                return self.fallback.get_telemetry(courier_id=courier_id)
        return self.fallback.get_telemetry(courier_id=courier_id)

    # =========================================================================
    # Zone Risk Scores API
    # =========================================================================
    def upsert_zone_risk_score(self, risk_dict: Dict[str, Any]) -> bool:
        """Persists evaluated risk score and alert metadata for a delivery zone."""
        if "delivery_zone" not in risk_dict or not str(risk_dict["delivery_zone"]).strip():
            raise ValueError("Risk document missing mandatory 'delivery_zone'")

        doc = dict(risk_dict)
        doc["calculated_at"] = doc.get("calculated_at") or datetime.now(timezone.utc).isoformat()

        if self.is_connected and self.db is not None:
            try:
                self.db.zone_risk_scores.update_one(
                    {"delivery_zone": doc["delivery_zone"]},
                    {"$set": doc},
                    upsert=True
                )
                return True
            except Exception as e:
                self.last_error = f"MongoDB risk score error: {e}"
                return self.fallback.upsert_zone_risk(doc)
        return self.fallback.upsert_zone_risk(doc)

    def get_zone_risk_scores(self) -> List[Dict[str, Any]]:
        """Retrieves latest risk metrics sorted by severity."""
        if self.is_connected and self.db is not None:
            try:
                cursor = self.db.zone_risk_scores.find({}, {"_id": 0}).sort("risk_score", DESCENDING)
                return list(cursor)
            except Exception as e:
                self.last_error = f"MongoDB retrieval error: {e}"
                return self.fallback.get_zone_risks()
        return self.fallback.get_zone_risks()

    # =========================================================================
    # Diagnostic Health Check
    # =========================================================================
    def health_check(self) -> Dict[str, Any]:
        """
        Executes a diagnostic health check across the persistence tier.
        Returns connection state, active storage engine, document counts, and latency.
        """
        start_t = time.perf_counter()
        if self.is_connected and self.db is not None:
            try:
                self.client.admin.command("ping")
                latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
                counts = {
                    "active_orders": self.db.active_orders.count_documents({}),
                    "courier_telemetry": self.db.courier_telemetry.count_documents({}),
                    "zone_risk_scores": self.db.zone_risk_scores.count_documents({})
                }
                return {
                    "status": "HEALTHY",
                    "backend": "MONGODB",
                    "is_connected": True,
                    "uri_configured": bool(os.getenv("MONGODB_URI")),
                    "database_name": self.db_name,
                    "collection_counts": counts,
                    "ping_latency_ms": latency_ms,
                    "diagnostic_message": "Connected to real MongoDB server instance with active collections."
                }
            except Exception as e:
                self.is_connected = False
                self.backend_name = "FALLBACK_SQLITE"
                self.last_error = str(e)

        # Fallback diagnostics
        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        counts = self.fallback.counts()
        return {
            "status": "OPERATIONAL_FALLBACK",
            "backend": "FALLBACK_SQLITE",
            "is_connected": False,
            "uri_configured": bool(os.getenv("MONGODB_URI")),
            "database_name": "local_sqlite_fallback",
            "fallback_db_path": str(self.fallback.db_path),
            "collection_counts": counts,
            "ping_latency_ms": latency_ms,
            "diagnostic_message": (
                "Running in local SQLite persistent fallback mode. "
                "All inserts, queries, and upserts function identically for academic demonstration. "
                f"Reason: {self.last_error or 'No active MongoDB daemon detected.'}"
            )
        }

    def clear_all(self) -> None:
        """Clears all records from both MongoDB and fallback store (for testing/resets)."""
        if self.is_connected and self.db is not None:
            try:
                self.db.active_orders.delete_many({})
                self.db.courier_telemetry.delete_many({})
                self.db.zone_risk_scores.delete_many({})
            except Exception:
                pass
        self.fallback.clear()


# Global Singleton instance
_INSTANCE: Optional[MongoManager] = None

def get_database_manager(force_new: bool = False, force_fallback: bool = False) -> MongoManager:
    """Returns singleton MongoManager instance."""
    global _INSTANCE
    if _INSTANCE is None or force_new:
        _INSTANCE = MongoManager(force_fallback=force_fallback)
    return _INSTANCE
