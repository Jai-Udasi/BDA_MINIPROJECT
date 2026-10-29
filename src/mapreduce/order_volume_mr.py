"""
Hadoop MapReduce Paradigm Demonstration for Food Delivery Order Analytics.
Explicitly implements Mapper, Combiner, Shuffle & Sort, and Reducer stages
to demonstrate Syllabus Module 3 (MapReduce Paradigm & Relational Algebra Aggregations).
"""
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterator
from collections import defaultdict
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import PROCESSED_ORDERS_FILE


class OrderVolumeMapper:
    """
    Map Phase: Emits key-value tuples from raw food delivery records.
    Key: delivery_zone
    Value: (order_amount, count=1, is_delayed)
    """
    def map(self, record: Dict[str, Any]) -> Iterator[Tuple[str, Tuple[float, int, int]]]:
        zone = record.get("delivery_zone", "Unknown")
        amount = float(record.get("order_amount", 0.0))
        is_delayed = int(record.get("is_delayed", 0))
        yield (zone, (amount, 1, is_delayed))


class OrderVolumeCombiner:
    """
    Combiner Phase (Mini-Reducer at Worker Node):
    Pre-aggregates intermediate key-value pairs locally before the Shuffle & Sort stage.
    Reduces network bandwidth and shuffle I/O in distributed clusters.
    """
    def combine(
        self,
        mapped_key_values: List[Tuple[str, Tuple[float, int, int]]]
    ) -> List[Tuple[str, Tuple[float, int, int]]]:
        local_groups = defaultdict(lambda: [0.0, 0, 0])
        for key, (amount, count, delayed) in mapped_key_values:
            local_groups[key][0] += amount
            local_groups[key][1] += count
            local_groups[key][2] += delayed

        return [
            (key, (round(vals[0], 2), vals[1], vals[2]))
            for key, vals in local_groups.items()
        ]


class OrderVolumeShuffleSort:
    """
    Shuffle & Sort Phase:
    Partitions, groups, and sorts intermediate outputs by key across all mappers/combiners.
    """
    def shuffle_and_sort(
        self,
        combiner_outputs: List[Tuple[str, Tuple[float, int, int]]]
    ) -> Dict[str, List[Tuple[float, int, int]]]:
        shuffled = defaultdict(list)
        for key, value_tuple in combiner_outputs:
            shuffled[key].append(value_tuple)
        return dict(sorted(shuffled.items()))


class OrderVolumeReducer:
    """
    Reduce Phase:
    Computes final aggregations for each unique delivery_zone key:
    - Total Orders
    - Total Revenue (GMV)
    - Average Order Value (AOV)
    - Total Delayed Orders
    - Delay Percentage (%)
    """
    def reduce(
        self,
        key: str,
        values_list: List[Tuple[float, int, int]]
    ) -> Dict[str, Any]:
        total_amount = sum(v[0] for v in values_list)
        total_orders = sum(v[1] for v in values_list)
        total_delayed = sum(v[2] for v in values_list)

        avg_order_value = round(total_amount / total_orders, 2) if total_orders > 0 else 0.0
        delay_rate_pct = round((total_delayed / total_orders * 100.0), 2) if total_orders > 0 else 0.0

        return {
            "delivery_zone": key,
            "total_orders": total_orders,
            "total_amount": round(total_amount, 2),
            "avg_order_value": avg_order_value,
            "total_delayed": total_delayed,
            "delay_rate_pct": delay_rate_pct
        }


class OrderVolumeMapReduceJob:
    """
    Orchestrates the complete MapReduce job execution pipeline with full observability
    over intermediate states for academic demonstration.
    """
    def __init__(self):
        self.mapper = OrderVolumeMapper()
        self.combiner = OrderVolumeCombiner()
        self.shuffler = OrderVolumeShuffleSort()
        self.reducer = OrderVolumeReducer()

    def run(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes Map -> Combine -> Shuffle/Sort -> Reduce with intermediate telemetry.
        """
        # Stage 1: Map Phase
        map_outputs: List[Tuple[str, Tuple[float, int, int]]] = []
        for record in records:
            for kv in self.mapper.map(record):
                map_outputs.append(kv)

        # Stage 2: Combiner Phase (simulating 2 local partitions)
        half = len(map_outputs) // 2
        p1 = self.combiner.combine(map_outputs[:half])
        p2 = self.combiner.combine(map_outputs[half:])
        combined_outputs = p1 + p2

        # Stage 3: Shuffle & Sort Phase
        shuffled_groups = self.shuffler.shuffle_and_sort(combined_outputs)

        # Stage 4: Reduce Phase
        reduced_results: List[Dict[str, Any]] = []
        for key, values in shuffled_groups.items():
            reduced_results.append(self.reducer.reduce(key, values))

        # Sort reduced results by total_orders descending
        reduced_results.sort(key=lambda x: x["total_orders"], reverse=True)

        return {
            "total_input_records": len(records),
            "map_tuples_emitted": len(map_outputs),
            "combiner_tuples_emitted": len(combined_outputs),
            "shuffle_reduction_ratio_pct": round(
                (1.0 - (len(combined_outputs) / max(1, len(map_outputs)))) * 100.0, 2
            ),
            "distinct_keys_reduced": len(shuffled_groups),
            "sample_map_tuples": map_outputs[:5],
            "sample_combiner_tuples": combined_outputs[:5],
            "final_reduced_results": reduced_results
        }


def run_mapreduce_on_dataset(csv_path: Path = PROCESSED_ORDERS_FILE) -> Dict[str, Any]:
    """Helper function to load CSV data and run the MapReduce job."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file for MapReduce not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    job = OrderVolumeMapReduceJob()
    return job.run(records)


if __name__ == "__main__":
    print("=" * 60)
    print("  HADOOP MAPREDUCE ALGORITHM DEMONSTRATION (Syllabus Module 3)")
    print("=" * 60)
    
    summary = run_mapreduce_on_dataset()
    print(f"Total Input Records: {summary['total_input_records']}")
    print(f"Mapper Stage: Emitted {summary['map_tuples_emitted']} (key, value) pairs")
    print(f"Sample Mapped Tuples:")
    for t in summary["sample_map_tuples"]:
        print(f"   {t[0]} -> (Amount: Rs. {t[1][0]}, Count: {t[1][1]}, Delayed: {t[1][2]})")

    print(f"\nCombiner Stage: Reduced {summary['map_tuples_emitted']} pairs down to {summary['combiner_tuples_emitted']} pairs")
    print(f"Shuffle I/O Traffic Reduction: {summary['shuffle_reduction_ratio_pct']}% saved!")
    
    print(f"\nReducer Stage: Final Aggregations by Delivery Zone:")
    for res in summary["final_reduced_results"]:
        print(f"   Zone: {res['delivery_zone']:<25} Orders: {res['total_orders']:<6} GMV: Rs. {res['total_amount']:<10.2f} Delay Rate: {res['delay_rate_pct']}%")
    print("=" * 60)
