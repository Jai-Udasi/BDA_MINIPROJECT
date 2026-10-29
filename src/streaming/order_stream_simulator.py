"""
Streaming Food Delivery Event Simulator (Syllabus Module 4: Mining Data Streams).
Generates continuous, high-velocity JSON order event payloads with dynamic rush-hour rates,
duplicate injection for Bloom Filter testing, and realistic spatial-operational physics.

IMPORTANT ARCHITECTURAL NOTE:
This local simulator is not a real Apache Kafka deployment. It provides an in-memory transport
using Python's queue.Queue module with an architectural contract matching Kafka Producers
and Consumers, ensuring that the application runs locally and deterministically on Windows
without requiring heavy external infrastructure. A genuine Kafka broker can be slotted in
seamlessly when cluster deployment is requested.
"""
import time
import json
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Iterator
from queue import Queue, Empty
import threading

logger = logging.getLogger("OrderStreamSimulator")

# Constants aligned with project configuration
RESTAURANT_TYPES = [
    "Fast Food", "Casual Dining", "Cafe", "Cloud Kitchen", "Bakery & Desserts", "Fine Dining"
]
DELIVERY_ZONES = [
    "Downtown", "East Tech Park", "North Suburbs", "South Suburbs", "West University Campus", "Central Waterfront"
]
WEATHER_CONDITIONS = ["Clear", "Rainy", "Thunderstorm", "Fog", "Overcast"]
TRAFFIC_LEVELS = ["Low", "Medium", "High", "Congested"]
ORDER_STATUSES = ["Delivered", "In-Transit", "Preparing", "Cancelled"]


class OrderStreamSimulator:
    """
    Simulates an unbounded real-time stream of food delivery orders.
    Supports configurable event rates, lunch/dinner rush hour multipliers,
    synthetic duplicate injection for Bloom Filter benchmarking, and bounded test execution.
    """
    def __init__(
        self,
        event_rate: float = 10.0,
        duplicate_prob: float = 0.05,
        customer_pool_size: int = 150,
        courier_pool_size: int = 40,
        seed: Optional[int] = 42,
        max_queue_size: int = 10000
    ):
        """
        Initialize the simulator.

        Parameters:
        -----------
        event_rate : float
            Base events generated per second (e.g. 5 to 50).
        duplicate_prob : float
            Probability [0.0, 1.0] of injecting a duplicate order_id (simulating network retries).
        customer_pool_size : int
            Total distinct customer IDs to sample from (for Flajolet-Martin estimation).
        courier_pool_size : int
            Total distinct active courier IDs to sample from.
        seed : Optional[int]
            Random seed for deterministic, reproducible test streams.
        max_queue_size : int
            Maximum events held in the in-memory queue.
        """
        self.base_event_rate = float(event_rate)
        self.duplicate_prob = float(duplicate_prob)
        self.customer_pool_size = customer_pool_size
        self.courier_pool_size = courier_pool_size
        self.seed = seed
        self.max_queue_size = max_queue_size

        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()

        # Customer and Courier pools
        self._customers = [f"CUST-{1000 + i}" for i in range(customer_pool_size)]
        self._couriers = [f"CR-{100 + i}" for i in range(courier_pool_size)]
        self._restaurants = [f"REST-{100 + i}" for i in range(30)]

        # Internal state tracking
        self._event_counter = 0
        self._emitted_order_ids: List[str] = []
        self._event_queue: Queue = Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._is_running = False

    @property
    def is_running(self) -> bool:
        """Returns True if the background generator thread is currently running."""
        return self._is_running

    def get_current_event_rate(self, simulated_hour: Optional[int] = None) -> float:
        """
        Calculates the effective event rate applying rush-hour peak surges.
        Lunch (12-14) and Dinner (19-22) experience a 2.0x to 3.5x throughput surge.
        """
        if simulated_hour is None:
            simulated_hour = datetime.now().hour

        if 12 <= simulated_hour <= 14:
            surge_multiplier = 2.5  # Lunch Rush
        elif 19 <= simulated_hour <= 22:
            surge_multiplier = 3.2  # Dinner Rush
        elif 0 <= simulated_hour <= 5:
            surge_multiplier = 0.3  # Late Night Lull
        else:
            surge_multiplier = 1.0  # Normal Daytime

        return round(self.base_event_rate * surge_multiplier, 2)

    def generate_single_event(
        self,
        event_index: Optional[int] = None,
        force_duplicate: bool = False,
        simulated_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generates a single realistic, JSON-compatible food-delivery order event dictionary.
        """
        if event_index is None:
            self._event_counter += 1
            idx = self._event_counter
        else:
            idx = event_index

        now = simulated_time or datetime.now()

        # Check for deliberate duplicate injection (simulating network retries)
        is_duplicate = False
        if (force_duplicate or (self._rng.random() < self.duplicate_prob)) and len(self._emitted_order_ids) > 0:
            order_id = self._rng.choice(self._emitted_order_ids)
            is_duplicate = True
        else:
            order_id = f"ORD-STREAM-{100000 + idx}"
            self._emitted_order_ids.append(order_id)
            if len(self._emitted_order_ids) > 2000:
                self._emitted_order_ids.pop(0)

        customer_id = self._rng.choice(self._customers)
        courier_id = self._rng.choice(self._couriers)
        restaurant_id = self._rng.choice(self._restaurants)
        restaurant_type = self._rng.choice(RESTAURANT_TYPES)
        delivery_zone = self._rng.choice(DELIVERY_ZONES)
        weather = self._rng.choice(WEATHER_CONDITIONS)
        traffic = self._rng.choice(TRAFFIC_LEVELS)

        item_count = self._rng.randint(1, 6)
        base_price = {
            "Fast Food": 120.0, "Casual Dining": 220.0, "Cafe": 150.0,
            "Cloud Kitchen": 180.0, "Bakery & Desserts": 140.0, "Fine Dining": 450.0
        }[restaurant_type]
        order_amount = round(base_price * item_count * self._rng.uniform(0.9, 1.3), 2)

        # Operational physics: prep time & travel duration
        base_prep = 10.0 if restaurant_type in ["Fast Food", "Cloud Kitchen"] else 20.0
        prep_time = round(base_prep + item_count * self._rng.uniform(1.5, 3.0) + self._rng.uniform(-2, 4), 1)
        distance_km = round(self._rng.uniform(1.0, 12.0), 2)

        # Travel delay modifiers
        weather_delay = 8.0 if weather in ["Rainy", "Thunderstorm"] else 0.0
        traffic_delay = 12.0 if traffic in ["High", "Congested"] else (4.0 if traffic == "Medium" else 0.0)
        travel_duration = round(prep_time + (distance_km * 3.0) + weather_delay + traffic_delay + self._rng.uniform(-3, 6), 1)

        # SLA Benchmark calculation: Threshold = 35 min + 2.5 min/km
        expected_sla_duration = 35.0 + 2.5 * distance_km
        is_delayed = 1 if travel_duration > expected_sla_duration else 0

        # Status: ~85% Delivered, ~10% In-Transit, ~5% Cancelled
        status_prob = self._rng.random()
        if status_prob < 0.85:
            status = "Delivered"
        elif status_prob < 0.95:
            status = "In-Transit"
        else:
            status = "Cancelled"
            is_delayed = 0

        event = {
            "order_id": order_id,
            "customer_id": customer_id,
            "courier_id": courier_id,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "restaurant_id": restaurant_id,
            "restaurant_type": restaurant_type,
            "delivery_zone": delivery_zone,
            "order_amount": order_amount,
            "item_count": item_count,
            "preparation_time_min": prep_time,
            "delivery_distance_km": distance_km,
            "delivery_duration_min": travel_duration,
            "order_status": status,
            "weather_condition": weather,
            "traffic_density": traffic,
            "is_delayed": is_delayed,
            "is_duplicate_injection": is_duplicate
        }
        return event

    def generate_batch(self, count: int) -> List[Dict[str, Any]]:
        """
        Generates a deterministic batch of N order events immediately without sleeping.
        Ideal for unit testing and fast reproducible benchmarks.
        """
        return [self.generate_single_event() for _ in range(count)]

    def _producer_loop(
        self,
        max_events: Optional[int],
        duration_seconds: Optional[float]
    ):
        """Internal background producer thread worker."""
        start_time = time.time()
        emitted = 0

        try:
            while not self._stop_event.is_set():
                if max_events is not None and emitted >= max_events:
                    break
                if duration_seconds is not None and (time.time() - start_time) >= duration_seconds:
                    break

                event = self.generate_single_event()
                try:
                    self._event_queue.put(event, timeout=0.2)
                    emitted += 1
                except Exception:
                    # Queue full, continue trying
                    pass

                # Dynamic sleep interval based on target rate
                effective_rate = max(0.1, self.get_current_event_rate())
                sleep_interval = 1.0 / effective_rate
                # Small sleep with early termination check
                if self._stop_event.wait(timeout=sleep_interval):
                    break
        finally:
            self._is_running = False

    def start(
        self,
        max_events: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        blocking: bool = False
    ):
        """
        Starts the real-time event generator.
        
        Parameters:
        -----------
        max_events : Optional[int]
            Stops automatically after emitting this number of events.
        duration_seconds : Optional[float]
            Stops automatically after this duration in seconds.
        blocking : bool
            If True, runs synchronously on the current thread.
            If False, runs asynchronously in a daemon background thread.
        """
        if self._is_running:
            logger.warning("OrderStreamSimulator is already running.")
            return

        self._stop_event.clear()
        self._is_running = True

        if blocking:
            self._producer_loop(max_events=max_events, duration_seconds=duration_seconds)
        else:
            self._producer_thread = threading.Thread(
                target=self._producer_loop,
                args=(max_events, duration_seconds),
                daemon=True,
                name="StreamProducerThread"
            )
            self._producer_thread.start()

    def stop(self):
        """Signals the background producer thread to stop and waits for termination."""
        self._stop_event.set()
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=1.0)
        self._is_running = False

    def get_next_event(self, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
        """Retrieves the next event from the stream queue, or returns None if timeout occurs."""
        try:
            return self._event_queue.get(timeout=timeout)
        except Empty:
            return None

    def drain_queue(self, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """Drains and returns all currently buffered events from the queue without blocking."""
        items = []
        while not self._event_queue.empty():
            if max_items is not None and len(items) >= max_items:
                break
            try:
                items.append(self._event_queue.get_nowait())
            except Empty:
                break
        return items

    def queue_size(self) -> int:
        """Returns the number of events currently waiting in the in-memory queue."""
        return self._event_queue.qsize()


def get_default_stream_simulator(seed: int = 42) -> OrderStreamSimulator:
    """Convenience factory providing a standard deterministic simulator instance."""
    return OrderStreamSimulator(
        event_rate=15.0,
        duplicate_prob=0.06,
        customer_pool_size=120,
        courier_pool_size=35,
        seed=seed
    )
