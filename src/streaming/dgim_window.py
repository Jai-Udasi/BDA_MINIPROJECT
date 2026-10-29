"""
DGIM Sliding-Window Algorithm (Syllabus Module 4: Mining Data Streams).
Estimates the number of 1-bits (e.g. delayed orders) in the most recent N stream events
using O(log^2 N) memory, guaranteeing a relative error bounded by at most 50%.

THEORETICAL FOUNDATIONS (Datar-Gionis-Indyk-Motwani, 2002):
1. The Problem:
   Given an unbounded binary stream, count the number of 1s in the most recent k elements
   (k <= N), where raw storage of the window would take O(N) bits.
2. The Bucket Representation:
   The stream of 1s is partitioned into buckets:
   - Each bucket has (timestamp, size).
   - Every bucket size is a power of 2: 1, 2, 4, 8, 16, ...
   - Invariant: There are at least 1 and AT MOST 2 buckets of any given size.
3. Bucket Merging on Ingestion:
   When a new 1 arrives:
   - A bucket of size 1 is created at the current timestamp.
   - If there are now 3 buckets of size 1, the two oldest are merged into a bucket of size 2.
   - If that creates 3 buckets of size 2, the two oldest are merged into a bucket of size 4.
   - This cascade terminates when every size has at most 2 buckets.
4. Window Expiration:
   Any bucket whose timestamp falls older than (current_timestamp - N) is dropped.
5. Querying k Recent Events:
   To count 1s in the last k events (k <= N):
   - Sum the full sizes of all buckets whose timestamps fall strictly within the last k events.
   - For the earliest bucket that overlaps the k boundary, add half its size: floor(size / 2).
   - Relative Error Bound <= (size_last / 2) / Total <= 50%.
"""
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class DGIMBucket:
    """Represents a single DGIM bucket with a timestamp and power-of-2 size."""
    timestamp: int
    size: int


class DGIMSlidingWindow:
    """
    Implements the DGIM algorithm for counting 1-bits in a sliding window of size N.
    """
    def __init__(self, window_size: int = 1000):
        """
        Initialize the DGIM window.

        Parameters:
        -----------
        window_size : int
            The maximum number of recent events to maintain (N). Default: 1,000.
        """
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")

        self.window_size: int = window_size
        self._current_timestamp: int = 0
        
        # Buckets ordered from newest to oldest:
        # buckets[0] is the newest bucket, buckets[-1] is the oldest bucket.
        self._buckets: List[DGIMBucket] = []

        # Ground truth circular buffer (retained for verification and college demonstration)
        self._ground_truth_window: deque = deque(maxlen=window_size)
        self._total_events_processed: int = 0

    def add(self, is_delayed_bit: int):
        """
        Ingests a binary bit (1 for delayed order, 0 for on-time order).
        """
        bit = 1 if is_delayed_bit else 0
        self._current_timestamp += 1
        self._total_events_processed += 1
        self._ground_truth_window.append(bit)

        # If bit is 0, no new bucket is formed; only drop expired buckets
        if bit == 0:
            self._drop_expired_buckets()
            return

        # Bit is 1: Create a new bucket of size 1 at the current timestamp
        new_bucket = DGIMBucket(timestamp=self._current_timestamp, size=1)
        self._buckets.insert(0, new_bucket)

        # Enforce DGIM Invariant: at most 2 buckets of any given size
        self._enforce_bucket_invariants()

        # Drop buckets outside the window N
        self._drop_expired_buckets()

    def _enforce_bucket_invariants(self):
        """
        Scans buckets from newest to oldest. If three buckets of the same size exist,
        merges the two oldest into a single bucket of twice the size with the timestamp
        of the newer of the two merged buckets. Cascades up powers of two.
        """
        i = 0
        while i < len(self._buckets) - 2:
            # Check if buckets at i, i+1, i+2 have the same size
            if (self._buckets[i].size == self._buckets[i + 1].size == self._buckets[i + 2].size):
                # The two oldest are at indices i+1 and i+2
                merged_size = self._buckets[i + 1].size * 2
                merged_timestamp = self._buckets[i + 1].timestamp  # Newer timestamp of the pair
                
                # Replace the two buckets with the merged bucket
                self._buckets[i + 1] = DGIMBucket(timestamp=merged_timestamp, size=merged_size)
                self._buckets.pop(i + 2)
                
                # Re-evaluate from current position to handle potential cascade
                continue
            i += 1

    def _drop_expired_buckets(self):
        """
        Purges buckets whose timestamp is older than (current_timestamp - window_size).
        """
        cutoff_timestamp = self._current_timestamp - self.window_size
        while self._buckets and self._buckets[-1].timestamp <= cutoff_timestamp:
            self._buckets.pop()

    def count_recent(self, k: Optional[int] = None) -> int:
        """
        Estimates the number of 1-bits in the last k events (k <= window_size).
        If k is None, defaults to the full window size N.
        
        Estimation rule:
        Sum all buckets completely within k events, plus half the size of the oldest
        overlapping bucket.
        """
        if not self._buckets or self._total_events_processed == 0:
            return 0

        if k is None or k > self.window_size:
            k = min(self.window_size, self._total_events_processed)

        if k <= 0:
            return 0

        cutoff = self._current_timestamp - k + 1
        estimated_count = 0

        for i, b in enumerate(self._buckets):
            if b.timestamp >= cutoff:
                # Check if this is the last bucket in the window
                is_last_in_window = (i == len(self._buckets) - 1) or (self._buckets[i + 1].timestamp < cutoff)
                if is_last_in_window:
                    # Oldest overlapping bucket: add floor(size / 2)
                    estimated_count += (b.size // 2)
                    break
                else:
                    estimated_count += b.size
            else:
                break

        return estimated_count

    def exact_count(self, k: Optional[int] = None) -> int:
        """
        Computes the exact count of 1-bits in the last k events using the ground-truth deque.
        """
        if not self._ground_truth_window:
            return 0

        if k is None or k > len(self._ground_truth_window):
            k = len(self._ground_truth_window)

        if k <= 0:
            return 0

        # Ground truth deque has newest elements on the right
        recent_bits = list(self._ground_truth_window)[-k:]
        return sum(recent_bits)

    def get_buckets(self) -> List[Tuple[int, int]]:
        """
        Returns the active bucket list as a list of (timestamp, size) tuples.
        """
        return [(b.timestamp, b.size) for b in self._buckets]

    def get_metrics(self, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Returns full diagnostic metrics comparing DGIM estimate with exact count.
        """
        if k is None:
            k = self.window_size

        est = self.count_recent(k)
        exact = self.exact_count(k)
        abs_err = abs(est - exact)
        rel_err_pct = round((abs_err / exact * 100.0), 2) if exact > 0 else 0.0

        # Calculate theoretical memory footprint: O(log^2 N) bits
        bucket_count = len(self._buckets)
        raw_bits_needed = self.window_size
        # Each bucket stores timestamp (log N bits) and size (log N bits)
        dgim_bits_approx = bucket_count * 32  # 32 bits per bucket

        return {
            "window_size": self.window_size,
            "query_k": k,
            "total_events_processed": self._total_events_processed,
            "dgim_estimated_delayed_count": est,
            "exact_delayed_count": exact,
            "absolute_error": abs_err,
            "relative_error_pct": rel_err_pct,
            "active_bucket_count": bucket_count,
            "bucket_sizes": [b.size for b in self._buckets],
            "memory_reduction_ratio_pct": round((1.0 - (dgim_bits_approx / max(1, raw_bits_needed))) * 100.0, 1)
        }
