"""
Bloom Filter Implementation for Real-Time Event Filtering (Syllabus Module 4: Mining Data Streams).
Provides fast, space-efficient probabilistic set membership testing with zero false negatives.
Used in food delivery platforms for duplicate order suppression (network retries) and geofence filtering.

THEORETICAL FOUNDATIONS & GUARANTEES:
1. No False Negatives:
   If an element x was inserted into the Bloom filter, all k bit positions were set to 1.
   Therefore, might_contain(x) will ALWAYS return True under normal operation.
2. False Positives Can Occur:
   If an element y was NEVER inserted, but all k of its bit positions happen to have been set
   to 1 by other inserted elements (hash collision overlap), might_contain(y) returns True.
   Thus, a positive result indicates a "PROBABLE MATCH" or candidate, not a certainty.
3. False Positive Probability Formula:
   P(FP) = (1 - e^(-k * n / m))^k
   where m = bit array size, k = number of hash functions, n = number of inserted items.
"""
import math
import hashlib
from typing import List, Dict, Any, Tuple, Optional


class BloomFilter:
    """
    Space-efficient probabilistic bit-array data structure for stream filtering.
    """
    def __init__(self, size: int = 4096, num_hashes: int = 5):
        """
        Initialize the Bloom Filter.

        Parameters:
        -----------
        size : int
            Total size of the bit array in bits (m). Default is 4,096 bits.
        num_hashes : int
            Number of independent hash functions (k). Default is 5.
        """
        if size <= 0:
            raise ValueError(f"Bit array size must be positive, got {size}")
        if num_hashes <= 0:
            raise ValueError(f"Number of hash functions must be positive, got {num_hashes}")

        self.size: int = size
        self.num_hashes: int = num_hashes
        
        # Byte array holding the bit array: ceiling(size / 8) bytes
        self._byte_count: int = (size + 7) // 8
        self._bit_array: bytearray = bytearray(self._byte_count)

        # Operational metrics
        self._insertions_count: int = 0
        self._duplicate_candidates_detected: int = 0
        self._query_count: int = 0

    @classmethod
    def create_for_capacity(cls, expected_items: int, target_fp_rate: float = 0.01) -> "BloomFilter":
        """
        Factory method calculating optimal bit array size (m) and hash count (k)
        for a given capacity (n) and target false positive rate (p).

        Formulas:
            m = - (n * ln(p)) / (ln(2)^2)
            k = (m / n) * ln(2)
        """
        if expected_items <= 0:
            expected_items = 100
        if not (0.0 < target_fp_rate < 1.0):
            target_fp_rate = 0.01

        m = int(- (expected_items * math.log(target_fp_rate)) / (math.log(2) ** 2))
        k = max(1, int((m / expected_items) * math.log(2)))
        return cls(size=m, num_hashes=k)

    def _get_hash_positions(self, item: str) -> List[int]:
        """
        Generates k independent bit positions in [0, size - 1] for the given item
        using salted SHA-256 digests.
        """
        item_bytes = str(item).encode("utf-8")
        positions = []

        for i in range(self.num_hashes):
            # Salt with hash index i to guarantee independent hash distributions
            salt = str(i).encode("utf-8")
            digest = hashlib.sha256(salt + b":" + item_bytes).hexdigest()
            pos = int(digest, 16) % self.size
            positions.append(pos)

        return positions

    def add(self, item: str):
        """
        Inserts an item into the Bloom Filter by setting its k bit positions to 1.
        """
        positions = self._get_hash_positions(item)
        for pos in positions:
            byte_idx = pos // 8
            bit_idx = pos % 8
            self._bit_array[byte_idx] |= (1 << bit_idx)

        self._insertions_count += 1

    def might_contain(self, item: str) -> bool:
        """
        Queries whether the item might be in the set.
        Returns:
            False : The item is DEFINITELY NOT in the set (zero false negatives).
            True  : The item is PROBABLY in the set (subject to false positive probability).
        """
        self._query_count += 1
        positions = self._get_hash_positions(item)

        for pos in positions:
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self._bit_array[byte_idx] & (1 << bit_idx)):
                return False

        return True

    def process_order_event(self, order_id: str) -> Tuple[bool, str]:
        """
        Processes a streaming order ID:
        - If might_contain(order_id) is True: flags as a probable duplicate candidate.
        - If False: adds to the filter and marks as a unique new order.

        Returns:
        --------
        (is_probable_duplicate: bool, classification_reason: str)
        """
        if self.might_contain(order_id):
            self._duplicate_candidates_detected += 1
            return True, "PROBABLE_DUPLICATE_CANDIDATE: All hash bits set (probable network retry or re-submission)."
        else:
            self.add(order_id)
            return False, "UNIQUE_ORDER: Ingested and registered in Bloom bit array."

    @property
    def set_bits_count(self) -> int:
        """Counts the total number of bits currently set to 1 in the bit array."""
        return sum(bin(byte).count("1") for byte in self._bit_array)

    @property
    def occupancy_ratio(self) -> float:
        """Returns the fraction of bits currently set to 1 (between 0.0 and 1.0)."""
        return round(self.set_bits_count / self.size, 4)

    @property
    def theoretical_false_positive_rate(self) -> float:
        """
        Computes the theoretical false positive probability given current insertions:
        P(FP) ≈ (1 - e^(-k * n / m))^k
        """
        if self._insertions_count == 0:
            return 0.0

        exponent = - (self.num_hashes * self._insertions_count) / self.size
        p = 1.0 - math.exp(exponent)
        fp_rate = p ** self.num_hashes
        return round(min(1.0, max(0.0, fp_rate)), 5)

    def evaluate_false_positive_rate(self, unseen_items: List[str]) -> Dict[str, Any]:
        """
        Empirically tests the false positive rate by querying items known NOT to have been added.
        """
        if not unseen_items:
            return {
                "tested_unseen_count": 0,
                "observed_false_positives": 0,
                "observed_fp_rate": 0.0,
                "theoretical_fp_rate": self.theoretical_false_positive_rate
            }

        false_positives = sum(1 for item in unseen_items if self.might_contain(item))
        observed_rate = false_positives / len(unseen_items)

        return {
            "tested_unseen_count": len(unseen_items),
            "observed_false_positives": false_positives,
            "observed_fp_rate": round(observed_rate, 5),
            "theoretical_fp_rate": self.theoretical_false_positive_rate,
            "absolute_discrepancy": round(abs(observed_rate - self.theoretical_false_positive_rate), 5)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Returns full diagnostic state and metrics for dashboard rendering."""
        return {
            "bit_array_size_bits": self.size,
            "num_hashes": self.num_hashes,
            "total_insertions": self._insertions_count,
            "duplicate_candidates_detected": self._duplicate_candidates_detected,
            "set_bits": self.set_bits_count,
            "occupancy_ratio_pct": round(self.occupancy_ratio * 100.0, 2),
            "theoretical_fp_rate_pct": round(self.theoretical_false_positive_rate * 100.0, 3)
        }
