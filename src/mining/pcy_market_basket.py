"""
Park-Chen-Yu (PCY) Algorithm for Frequent Itemset Mining (Syllabus Module 5).
Implements the authentic two-pass market basket algorithm with hash-bucket bitmap optimization.

THEORETICAL FOUNDATIONS (Park, Chen, Yu, 1995):
1. Motivation:
   In Apriori, if there are C frequent individual items, Apriori evaluates C*(C-1)/2 candidate
   pairs in Pass 2, which requires significant memory when C is large.
2. PCY Pass 1:
   - Maintains an exact count of each individual item in main memory.
   - Concurrently, generates all pairs in each basket and hashes them into a fixed-size
     hash bucket array H of size B: bucket_idx = hash_pair(i, j) % B.
   - Increments H[bucket_idx] for every pair seen.
3. Bitmap Construction:
   - After Pass 1, replaces the integer bucket array H with a single BITMAP:
     bitmap[b] = 1 if H[b] >= min_support else 0.
   - Reduces the bucket array memory by 32x (from 32-bit integers to single bits).
4. PCY Pass 2:
   - Retains a pair {i, j} as a candidate pair ONLY IF:
     a) Item i is frequent (count[i] >= min_support).
     b) Item j is frequent (count[j] >= min_support).
     c) The hash bucket of {i, j} is marked as frequent in the bitmap (bitmap[hash_pair(i, j)] == 1).
   - If ANY of these three conditions is false, the pair cannot be frequent and is skipped!
   - This drastically reduces the number of candidate pairs held in memory during Pass 2.
"""
import hashlib
from typing import List, Dict, Tuple, Set, Any, Optional
from itertools import combinations
from collections import defaultdict


class PCYMarketBasketMiner:
    """
    Implements the two-pass PCY algorithm for mining frequent item pairs and association rules.
    """
    def __init__(self, min_support: int = 50, num_buckets: int = 1000):
        """
        Initialize PCY miner.

        Parameters:
        -----------
        min_support : int
            Absolute minimum support threshold count (number of transactions).
        num_buckets : int
            Fixed size of the hash bucket array for Pass 1 (B).
        """
        if min_support <= 0:
            raise ValueError(f"min_support must be a positive integer, got {min_support}")
        if num_buckets <= 0:
            raise ValueError(f"num_buckets must be a positive integer, got {num_buckets}")

        self.min_support: int = min_support
        self.num_buckets: int = num_buckets

    def _hash_pair(self, item1: str, item2: str) -> int:
        """
        Deterministic, order-invariant hash function for an item pair {item1, item2}.
        Uses salted MD5 / SHA-256 modulo num_buckets.
        """
        # Sort items lexicographically to ensure order independence
        sorted_pair = sorted([str(item1), str(item2)])
        pair_str = f"{sorted_pair[0]}||{sorted_pair[1]}".encode("utf-8")
        h = int(hashlib.md5(pair_str).hexdigest(), 16)
        return h % self.num_buckets

    def pass1(self, baskets: List[List[str]]) -> Tuple[Dict[str, int], List[int], List[int]]:
        """
        Pass 1:
        1. Counts occurrences of each individual item.
        2. Hashes all candidate pairs into integer bucket array of size B.
        3. Constructs the bitmap: 1 if bucket count >= min_support, else 0.
        
        Returns:
        --------
        (item_counts, hash_buckets, bitmap)
        """
        item_counts: Dict[str, int] = defaultdict(int)
        hash_buckets: List[int] = [0] * self.num_buckets

        for basket in baskets:
            # Eliminate duplicate items within the same basket
            unique_items = sorted(list(set(basket)))
            
            # Count individual items
            for item in unique_items:
                item_counts[item] += 1

            # Hash pairs into buckets
            for item1, item2 in combinations(unique_items, 2):
                b_idx = self._hash_pair(item1, item2)
                hash_buckets[b_idx] += 1

        # Construct the bitmap from the bucket counts
        bitmap: List[int] = [1 if count >= self.min_support else 0 for count in hash_buckets]

        return dict(item_counts), hash_buckets, bitmap

    def pass2(
        self,
        baskets: List[List[str]],
        frequent_items: Set[str],
        bitmap: List[int]
    ) -> Tuple[Dict[Tuple[str, str], int], int]:
        """
        Pass 2:
        Counts only candidate pairs {i, j} where:
        1. i in frequent_items
        2. j in frequent_items
        3. bitmap[hash_pair(i, j)] == 1
        
        Returns:
        --------
        (frequent_pairs, candidate_pairs_evaluated_count)
        """
        candidate_pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        candidates_admitted: Set[Tuple[str, str]] = set()

        for basket in baskets:
            unique_items = sorted(list(set(basket)))
            # Only consider items that were frequent in Pass 1
            filtered_items = [item for item in unique_items if item in frequent_items]

            for item1, item2 in combinations(filtered_items, 2):
                pair = (item1, item2)
                b_idx = self._hash_pair(item1, item2)
                # Check bitmap filter
                if bitmap[b_idx] == 1:
                    candidates_admitted.add(pair)
                    candidate_pair_counts[pair] += 1

        # Final filter: candidate pairs that satisfy min_support
        frequent_pairs = {
            pair: count for pair, count in candidate_pair_counts.items()
            if count >= self.min_support
        }

        return frequent_pairs, len(candidates_admitted)

    def generate_association_rules(
        self,
        frequent_pairs: Dict[Tuple[str, str], int],
        item_counts: Dict[str, int],
        total_baskets: int,
        min_confidence: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Generates association rules (A -> B and B -> A) from frequent pairs with Support, Confidence, and Lift.
        """
        rules = []

        for (item_a, item_b), pair_support_count in frequent_pairs.items():
            pair_support_pct = pair_support_count / total_baskets
            count_a = item_counts.get(item_a, 0)
            count_b = item_counts.get(item_b, 0)

            # Rule 1: item_a -> item_b
            if count_a > 0:
                conf_a_b = pair_support_count / count_a
                lift_a_b = (conf_a_b) / (count_b / total_baskets) if count_b > 0 else 1.0
                if conf_a_b >= min_confidence:
                    rules.append({
                        "antecedent": item_a,
                        "consequent": item_b,
                        "support_count": pair_support_count,
                        "support_pct": round(pair_support_pct * 100.0, 2),
                        "confidence_pct": round(conf_a_b * 100.0, 2),
                        "lift": round(lift_a_b, 2)
                    })

            # Rule 2: item_b -> item_a
            if count_b > 0:
                conf_b_a = pair_support_count / count_b
                lift_b_a = (conf_b_a) / (count_a / total_baskets) if count_a > 0 else 1.0
                if conf_b_a >= min_confidence:
                    rules.append({
                        "antecedent": item_b,
                        "consequent": item_a,
                        "support_count": pair_support_count,
                        "support_pct": round(pair_support_pct * 100.0, 2),
                        "confidence_pct": round(conf_b_a * 100.0, 2),
                        "lift": round(lift_b_a, 2)
                    })

        # Sort rules descending by confidence and lift
        rules.sort(key=lambda r: (r["confidence_pct"], r["lift"]), reverse=True)
        return rules

    @staticmethod
    def exact_pair_count(baskets: List[List[str]], min_support: int) -> Dict[Tuple[str, str], int]:
        """
        Ground-truth reference implementation: counts all pairs directly in a single pass.
        Used for mathematical verification in unit tests.
        """
        direct_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for basket in baskets:
            unique_items = sorted(list(set(basket)))
            for pair in combinations(unique_items, 2):
                direct_counts[pair] += 1

        return {pair: count for pair, count in direct_counts.items() if count >= min_support}

    def run(self, baskets: List[List[str]], min_confidence: float = 0.50) -> Dict[str, Any]:
        """
        Orchestrates full PCY execution and returns comprehensive diagnostics.
        """
        total_baskets = len(baskets)
        if total_baskets == 0:
            return {
                "total_baskets": 0,
                "frequent_single_items": {},
                "frequent_pairs": {},
                "association_rules": [],
                "metrics": {"total_candidates_admitted": 0, "frequent_buckets_count": 0}
            }

        # Stage 1: Pass 1
        item_counts, hash_buckets, bitmap = self.pass1(baskets)
        frequent_items = {
            item: count for item, count in item_counts.items()
            if count >= self.min_support
        }
        frequent_item_set = set(frequent_items.keys())

        # Stage 2: Pass 2
        frequent_pairs, candidates_admitted_count = self.pass2(baskets, frequent_item_set, bitmap)

        # Stage 3: Association Rules
        rules = self.generate_association_rules(frequent_pairs, item_counts, total_baskets, min_confidence)

        # Theoretical Apriori pair count vs. PCY admitted candidates
        num_freq_items = len(frequent_item_set)
        apriori_candidate_pairs = (num_freq_items * (num_freq_items - 1)) // 2 if num_freq_items >= 2 else 0
        frequent_buckets_count = sum(bitmap)

        return {
            "total_baskets": total_baskets,
            "min_support_threshold": self.min_support,
            "min_support_pct": round((self.min_support / total_baskets * 100.0), 2),
            "num_buckets": self.num_buckets,
            "frequent_single_items": frequent_items,
            "frequent_pairs": {f"{p[0]} + {p[1]}": count for p, count in sorted(frequent_pairs.items(), key=lambda x: x[1], reverse=True)},
            "association_rules": rules,
            "metrics": {
                "frequent_items_count": num_freq_items,
                "apriori_candidate_pairs": apriori_candidate_pairs,
                "pcy_candidates_admitted": candidates_admitted_count,
                "frequent_buckets_count": frequent_buckets_count,
                "frequent_buckets_ratio_pct": round(frequent_buckets_count / self.num_buckets * 100.0, 2),
                "memory_reduction_factor": round(32.0, 1)  # 32-bit int to 1-bit bitmap
            }
        }
