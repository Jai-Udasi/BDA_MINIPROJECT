"""
Unit and Integration Tests for Big Data Streaming Algorithms (Syllabus Module 4).
Verifies:
1. OrderStreamSimulator schema, determinism, and bounded-batch execution.
2. BloomFilter zero false negatives, duplicate candidate detection, and empirical FP rates.
3. Flajolet-Martin count-distinct estimation accuracy and trailing zeros bit analysis.
4. DGIM bucket invariants, power-of-2 merging, window expiration, and 50% relative error bound.
"""
import pytest
from src.streaming.order_stream_simulator import OrderStreamSimulator
from src.streaming.bloom_filter import BloomFilter
from src.streaming.flajolet_martin import FlajoletMartin
from src.streaming.dgim_window import DGIMSlidingWindow


# ==============================================================================
# 1. ORDER STREAM SIMULATOR TESTS
# ==============================================================================

def test_simulator_generates_valid_schema():
    """Verify that generated streaming events contain all required schema attributes."""
    sim = OrderStreamSimulator(seed=42)
    event = sim.generate_single_event()

    required_keys = [
        "order_id", "customer_id", "courier_id", "timestamp",
        "restaurant_id", "restaurant_type", "delivery_zone",
        "order_amount", "item_count", "preparation_time_min",
        "delivery_distance_km", "delivery_duration_min", "order_status",
        "weather_condition", "traffic_density", "is_delayed", "is_duplicate_injection"
    ]
    for k in required_keys:
        assert k in event, f"Missing key '{k}' in streaming event"

    assert event["order_id"].startswith("ORD-STREAM-")
    assert event["customer_id"].startswith("CUST-")
    assert event["courier_id"].startswith("CR-")
    assert event["order_amount"] > 0
    assert event["is_delayed"] in [0, 1]


def test_simulator_deterministic_reproducibility():
    """Verify identical seeds produce exact identical event sequences."""
    sim1 = OrderStreamSimulator(seed=123)
    sim2 = OrderStreamSimulator(seed=123)

    batch1 = sim1.generate_batch(20)
    batch2 = sim2.generate_batch(20)

    assert len(batch1) == 20
    assert len(batch2) == 20
    for e1, e2 in zip(batch1, batch2):
        assert e1["order_id"] == e2["order_id"]
        assert e1["order_amount"] == e2["order_amount"]
        assert e1["is_delayed"] == e2["is_delayed"]


def test_simulator_bounded_batch_execution():
    """Verify bounded execution generates exactly the requested number of events."""
    sim = OrderStreamSimulator(seed=999)
    batch = sim.generate_batch(50)
    assert len(batch) == 50


# ==============================================================================
# 2. BLOOM FILTER TESTS
# ==============================================================================

def test_bloom_filter_zero_false_negatives():
    """
    CRITICAL THEORETICAL GUARANTEE:
    If an item is inserted, might_contain() must ALWAYS return True (zero false negatives).
    """
    bf = BloomFilter(size=1024, num_hashes=4)
    items = [f"ORD-TEST-{i}" for i in range(100)]

    for item in items:
        bf.add(item)

    # Every single inserted item must be found
    for item in items:
        assert bf.might_contain(item) is True, f"False negative detected for {item}!"


def test_bloom_filter_duplicate_detection():
    """Verify that process_order_event flags duplicate candidates."""
    bf = BloomFilter(size=2048, num_hashes=4)

    # First insertion: must be unique
    is_dup1, reason1 = bf.process_order_event("ORD-ALPHA")
    assert is_dup1 is False
    assert "UNIQUE_ORDER" in reason1

    # Second insertion with same ID: must be flagged as probable duplicate
    is_dup2, reason2 = bf.process_order_event("ORD-ALPHA")
    assert is_dup2 is True
    assert "PROBABLE_DUPLICATE_CANDIDATE" in reason2


def test_bloom_filter_empirical_vs_theoretical_fp_rate():
    """
    Verify that empirical false positive rate on uninserted items is within expected bounds.
    """
    # Create filter configured for 200 items with ~2% false positive rate
    bf = BloomFilter.create_for_capacity(expected_items=200, target_fp_rate=0.02)
    inserted = [f"INSERTED-{i}" for i in range(200)]
    for item in inserted:
        bf.add(item)

    # Test 1,000 unseen items
    unseen = [f"UNSEEN-{i}" for i in range(1000)]
    metrics = bf.evaluate_false_positive_rate(unseen)

    # Empirical rate should be close to theoretical rate (within a reasonable statistical tolerance)
    assert metrics["observed_fp_rate"] < 0.08  # Tolerant upper bound for 1,000 queries
    assert bf.occupancy_ratio > 0.0 and bf.occupancy_ratio < 1.0


def test_bloom_filter_invalid_parameters():
    """Verify ValueError is raised on invalid size or hash count."""
    with pytest.raises(ValueError):
        BloomFilter(size=0, num_hashes=3)
    with pytest.raises(ValueError):
        BloomFilter(size=100, num_hashes=0)


# ==============================================================================
# 3. FLAJOLET-MARTIN ALGORITHM TESTS
# ==============================================================================

def test_flajolet_martin_trailing_zeros_calculation():
    """Verify trailing zeros calculation across powers of 2 and edge cases."""
    assert FlajoletMartin.trailing_zeros(1) == 0      # 1 = ...0001 -> 0 zeros
    assert FlajoletMartin.trailing_zeros(2) == 1      # 2 = ...0010 -> 1 zero
    assert FlajoletMartin.trailing_zeros(4) == 2      # 4 = ...0100 -> 2 zeros
    assert FlajoletMartin.trailing_zeros(8) == 3      # 8 = ...1000 -> 3 zeros
    assert FlajoletMartin.trailing_zeros(12) == 2     # 12 = ...1100 -> 2 zeros
    assert FlajoletMartin.trailing_zeros(0) == 64     # 0 edge case -> max_bits


def test_flajolet_martin_distinct_count_estimation():
    """
    Verify Flajolet-Martin median-of-means estimate accurately tracks known distinct counts
    within acceptable theoretical error bounds.
    """
    fm = FlajoletMartin(num_groups=5, hashes_per_group=4)

    # Stream containing 80 unique couriers repeated 5 times each (400 total stream elements)
    unique_couriers = [f"CR-{200 + i}" for i in range(80)]
    for _ in range(5):
        for c in unique_couriers:
            fm.add(c)

    metrics = fm.get_metrics()
    assert metrics["exact_distinct_count"] == 80
    assert metrics["total_stream_items_processed"] == 400

    estimate = fm.estimate()
    # Flajolet-Martin median-of-means typically has relative error <= 30% on sample sizes of ~80
    relative_error = abs(estimate - 80) / 80.0
    assert relative_error <= 0.35, f"FM estimate {estimate} deviated too much from 80 (rel error: {relative_error})"


def test_flajolet_martin_empty_stream():
    """Verify FM returns 0.0 on an uninitialized/empty stream."""
    fm = FlajoletMartin()
    assert fm.estimate() == 0.0
    assert fm.get_exact_count() == 0


# ==============================================================================
# 4. DGIM SLIDING-WINDOW ALGORITHM TESTS
# ==============================================================================

def test_dgim_bucket_invariants():
    """
    CRITICAL THEORETICAL GUARANTEE:
    At no point may there be more than 2 buckets of any given size.
    """
    dgim = DGIMSlidingWindow(window_size=100)

    # Ingest a deterministic pattern of 1s and 0s
    stream = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]
    for bit in stream:
        dgim.add(bit)

    # Count occurrences of each size
    size_counts = {}
    for ts, size in dgim.get_buckets():
        # Every size must be a power of 2
        assert (size & (size - 1)) == 0, f"Bucket size {size} is not a power of 2!"
        size_counts[size] = size_counts.get(size, 0) + 1

    # Invariant check: at most 2 buckets of any size
    for size, count in size_counts.items():
        assert count <= 2, f"DGIM Invariant violated: {count} buckets of size {size}!"


def test_dgim_window_expiration():
    """Verify that buckets older than the sliding window size are purged."""
    window_size = 20
    dgim = DGIMSlidingWindow(window_size=window_size)

    # Add 10 delayed orders
    for _ in range(10):
        dgim.add(1)

    # Add 30 non-delayed orders (pushing the 1s out of the window of size 20)
    for _ in range(30):
        dgim.add(0)

    # All 1s should have expired from the window
    assert dgim.exact_count() == 0
    assert dgim.count_recent() == 0
    assert len(dgim.get_buckets()) == 0


def test_dgim_relative_error_bound():
    """
    CRITICAL THEORETICAL GUARANTEE:
    DGIM estimate for the window must have relative error <= 50%.
    """
    dgim = DGIMSlidingWindow(window_size=200)

    # Ingest 200 bits with known 30% delay rate
    import random
    rng = random.Random(42)
    for _ in range(200):
        bit = 1 if rng.random() < 0.30 else 0
        dgim.add(bit)

    exact = dgim.exact_count()
    estimated = dgim.count_recent()

    if exact > 0:
        rel_error = abs(estimated - exact) / exact
        assert rel_error <= 0.50, f"DGIM error {rel_error:.2%} exceeded theoretical 50% bound! (Est: {estimated}, Exact: {exact})"


def test_dgim_query_sub_window():
    """Verify DGIM querying smaller sub-windows k < N."""
    dgim = DGIMSlidingWindow(window_size=100)
    # Ingest 50 ones
    for _ in range(50):
        dgim.add(1)

    # Query last 10 elements
    k_10 = dgim.count_recent(k=10)
    assert k_10 > 0
    assert k_10 <= 16
