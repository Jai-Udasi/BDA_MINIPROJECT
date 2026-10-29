"""
Unit tests for Hadoop MapReduce Algorithm Demonstration (Syllabus Module 3).
Verifies Mapper, Combiner, Shuffle & Sort, and Reducer against deterministic ground-truth values.
"""
import pytest
from src.mapreduce.order_volume_mr import (
    OrderVolumeMapper,
    OrderVolumeCombiner,
    OrderVolumeShuffleSort,
    OrderVolumeReducer,
    OrderVolumeMapReduceJob
)


@pytest.fixture
def deterministic_records():
    """Deterministic records with known ground truth."""
    return [
        {"order_id": "ORD-1", "delivery_zone": "Downtown", "order_amount": 200.0, "is_delayed": 0},
        {"order_id": "ORD-2", "delivery_zone": "Downtown", "order_amount": 300.0, "is_delayed": 1},
        {"order_id": "ORD-3", "delivery_zone": "Suburbs", "order_amount": 150.0, "is_delayed": 0},
        {"order_id": "ORD-4", "delivery_zone": "Suburbs", "order_amount": 450.0, "is_delayed": 1},
        {"order_id": "ORD-5", "delivery_zone": "Downtown", "order_amount": 100.0, "is_delayed": 0},
    ]


def test_mapper_emits_expected_tuples():
    """Verify Mapper emits (zone, (amount, 1, is_delayed))."""
    mapper = OrderVolumeMapper()
    record = {"delivery_zone": "Downtown", "order_amount": 250.0, "is_delayed": 1}
    emitted = list(mapper.map(record))
    
    assert len(emitted) == 1
    key, val = emitted[0]
    assert key == "Downtown"
    assert val == (250.0, 1, 1)


def test_combiner_pre_aggregates_locally():
    """Verify Combiner pre-aggregates key-values into local subtotals."""
    combiner = OrderVolumeCombiner()
    mapped = [
        ("Downtown", (100.0, 1, 0)),
        ("Downtown", (200.0, 1, 1)),
        ("Suburbs", (50.0, 1, 0))
    ]
    combined = combiner.combine(mapped)
    combined_dict = dict(combined)

    assert "Downtown" in combined_dict
    assert combined_dict["Downtown"] == (300.0, 2, 1)
    assert combined_dict["Suburbs"] == (50.0, 1, 0)


def test_shuffle_and_sort_groups_by_key():
    """Verify Shuffle & Sort groups all combiner tuples under their respective keys."""
    shuffler = OrderVolumeShuffleSort()
    combiner_outputs = [
        ("Suburbs", (50.0, 1, 0)),
        ("Downtown", (300.0, 2, 1)),
        ("Downtown", (150.0, 1, 0)),
    ]
    grouped = shuffler.shuffle_and_sort(combiner_outputs)

    assert list(grouped.keys()) == ["Downtown", "Suburbs"]
    assert len(grouped["Downtown"]) == 2
    assert len(grouped["Suburbs"]) == 1


def test_reducer_computes_final_aggregations():
    """Verify Reducer computes accurate totals, averages, and delay percentages."""
    reducer = OrderVolumeReducer()
    values = [(300.0, 2, 1), (150.0, 1, 0)]
    result = reducer.reduce("Downtown", values)

    assert result["delivery_zone"] == "Downtown"
    assert result["total_orders"] == 3
    assert result["total_amount"] == 450.0
    assert result["avg_order_value"] == 150.0
    assert result["total_delayed"] == 1
    assert result["delay_rate_pct"] == round(1 / 3 * 100.0, 2)


def test_end_to_end_mapreduce_job(deterministic_records):
    """Verify full MapReduce pipeline on deterministic fixture."""
    job = OrderVolumeMapReduceJob()
    summary = job.run(deterministic_records)

    assert summary["total_input_records"] == 5
    assert summary["map_tuples_emitted"] == 5
    assert summary["distinct_keys_reduced"] == 2

    res_map = {r["delivery_zone"]: r for r in summary["final_reduced_results"]}
    
    # Downtown: 3 orders, 200 + 300 + 100 = 600.0, 1 delayed (33.33%)
    assert res_map["Downtown"]["total_orders"] == 3
    assert res_map["Downtown"]["total_amount"] == 600.0
    assert res_map["Downtown"]["avg_order_value"] == 200.0
    assert res_map["Downtown"]["total_delayed"] == 1

    # Suburbs: 2 orders, 150 + 450 = 600.0, 1 delayed (50.0%)
    assert res_map["Suburbs"]["total_orders"] == 2
    assert res_map["Suburbs"]["total_amount"] == 600.0
    assert res_map["Suburbs"]["avg_order_value"] == 300.0
    assert res_map["Suburbs"]["total_delayed"] == 1
