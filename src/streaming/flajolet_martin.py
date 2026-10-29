"""
Flajolet-Martin (FM) Algorithm for Distinct Element Counting (Syllabus Module 4: Mining Data Streams).
Estimates the number of unique identifiers (active couriers, ordering customers) in an unbounded stream
using logarithmic space O(log N) bits, without retaining the distinct elements in memory.

THEORETICAL FOUNDATIONS & ALGORITHM MECHANICS:
1. Stream Mapping:
   Each incoming identifier x is mapped deterministically to a 64-bit integer using salted SHA-256 hashes:
   y = hash_j(x).
2. Trailing Zeros Function rho(y):
   rho(y) counts the number of trailing zero bits in the binary representation of y
   (the 0-indexed position of the least significant 1-bit).
   Example:
     y = 12 (binary ...00001100) -> rho(y) = 2
     y = 7  (binary ...00000111) -> rho(y) = 0
     y = 0  -> rho(0) = 64 (safe boundary condition)
3. Mathematical Expectation:
   The probability of seeing r trailing zeros is 2^(-r).
   Over N distinct elements, the maximum trailing zeros R = max_x rho(y) satisfies:
   E[2^R] ≈ phi * N, where phi ≈ 0.77351 (the Flajolet-Martin constant).
   Single-hash estimate: N_hat = 2^R / phi ≈ 1.2928 * 2^R.
4. Variance Reduction (Median-of-Means / AMS Paradigm):
   Because 2^R only takes discrete powers of 2, a single hash has high variance.
   We deploy K = (num_groups * hashes_per_group) independent hash functions:
   - Compute N_hat for each hash function.
   - Compute the arithmetic MEAN within each group (drastically reducing variance).
   - Compute the MEDIAN across all group means (eliminating susceptibility to extreme outliers).
"""
import math
import hashlib
from typing import List, Dict, Any, Optional


class FlajoletMartin:
    """
    Implements the Flajolet-Martin distinct counting algorithm with median-of-means variance reduction.
    """
    FM_PHI = 0.77351  # Flajolet-Martin correction constant (E[2^R] / N)

    def __init__(self, num_groups: int = 5, hashes_per_group: int = 4, hash_bits: int = 64):
        """
        Initialize the Flajolet-Martin estimator.

        Parameters:
        -----------
        num_groups : int
            Number of groups across which the median is taken. Default: 5.
        hashes_per_group : int
            Number of independent hash functions per group averaged together. Default: 4.
            Total hash functions = num_groups * hashes_per_group = 20.
        hash_bits : int
            Number of bits per hash value. Default: 64.
        """
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        if hashes_per_group <= 0:
            raise ValueError(f"hashes_per_group must be positive, got {hashes_per_group}")

        self.num_groups: int = num_groups
        self.hashes_per_group: int = hashes_per_group
        self.total_hashes: int = num_groups * hashes_per_group
        self.hash_bits: int = hash_bits

        # R_registers[j] stores max trailing zeros seen by hash function j
        self._r_registers: List[int] = [0] * self.total_hashes

        # Tracking metrics for validation and benchmarking
        self._stream_items_processed: int = 0
        self._exact_ground_truth_set: set = set()  # Retained only for academic benchmarking

    @staticmethod
    def trailing_zeros(value: int, max_bits: int = 64) -> int:
        """
        Calculates the number of trailing zero bits in value.
        Safely handles edge cases like value = 0.
        """
        if value <= 0:
            return max_bits

        count = 0
        # Check bits using bitwise shift
        while (value & 1) == 0 and count < max_bits:
            count += 1
            value >>= 1

        return count

    def _hash_item(self, item: str, hash_index: int) -> int:
        """
        Computes a deterministic, uniform 64-bit integer hash for item using salted SHA-256.
        """
        item_bytes = str(item).encode("utf-8")
        salt = f"FM_SALT_{hash_index}:".encode("utf-8")
        digest = hashlib.sha256(salt + item_bytes).hexdigest()
        # Take the lowest 64 bits (16 hex chars)
        return int(digest[:16], 16)

    def add(self, item: str):
        """
        Ingests a stream element into the Flajolet-Martin registers.
        """
        item_str = str(item)
        self._stream_items_processed += 1
        self._exact_ground_truth_set.add(item_str)

        for j in range(self.total_hashes):
            h_val = self._hash_item(item_str, j)
            tz = self.trailing_zeros(h_val, self.hash_bits)
            if tz > self._r_registers[j]:
                self._r_registers[j] = tz

    def estimate(self) -> float:
        """
        Computes the median-of-means distinct count estimate across all hash groups:
        1. Calculate raw estimate N_hat_j = 2^(R_j) / phi for each hash j.
        2. Average estimates within each group of size hashes_per_group.
        3. Take the median of group averages.
        """
        if self._stream_items_processed == 0:
            return 0.0

        # If all registers are 0, estimate is 0
        if all(r == 0 for r in self._r_registers):
            return 0.0

        group_means = []
        for g in range(self.num_groups):
            group_estimates = []
            for h in range(self.hashes_per_group):
                idx = g * self.hashes_per_group + h
                r_val = self._r_registers[idx]
                raw_estimate = (2.0 ** r_val) / self.FM_PHI
                group_estimates.append(raw_estimate)

            mean_val = sum(group_estimates) / len(group_estimates)
            group_means.append(mean_val)

        # Median of group means
        group_means.sort()
        mid = len(group_means) // 2
        if len(group_means) % 2 == 1:
            median_val = group_means[mid]
        else:
            median_val = (group_means[mid - 1] + group_means[mid]) / 2.0

        return round(max(0.0, median_val), 1)

    def get_exact_count(self) -> int:
        """Returns the exact count of unique items seen (for verification)."""
        return len(self._exact_ground_truth_set)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns full diagnostic metrics comparing FM estimate with ground truth.
        """
        est = self.estimate()
        exact = self.get_exact_count()
        abs_err = abs(est - exact)
        rel_err_pct = round((abs_err / exact * 100.0), 2) if exact > 0 else 0.0

        return {
            "total_stream_items_processed": self._stream_items_processed,
            "exact_distinct_count": exact,
            "flajolet_martin_estimate": est,
            "absolute_error": round(abs_err, 1),
            "relative_error_pct": rel_err_pct,
            "num_groups": self.num_groups,
            "hashes_per_group": self.hashes_per_group,
            "max_r_observed": max(self._r_registers) if self._r_registers else 0,
            "registers_sample": self._r_registers[:8]
        }
