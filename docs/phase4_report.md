# Phase 4 Technical Report: Mining Big Data Streams

**Project Title:** Real-Time Food Delivery Demand Forecasting and Delay Prediction Using Big Data Analytics  
**Course:** Big Data Analytics (Course Code: BDA) — Autonomous NEP Batch 1  
**Institution:** Vivekanand Education Society's Institute of Technology (VESIT), Department of Information Technology  
**Student:** Jai Udasi (Class: D20A, Roll No: 34)  
**Syllabus Alignment:** **Module 4 — Mining Data Streams** (Stream Data Model, Filtering Streams, Counting Distinct Elements, Counting in a Window)  
**Status:** **Completed & Fully Verified (46/46 Total Tests Passing, 14 Streaming Tests Passing)**

---

## 1. Executive Summary & Problem Formulation

In high-throughput food delivery platforms (such as Swiggy, Zomato, and DoorDash), transactional events arrive as an unbounded, continuous stream at high velocity. During rush hours, tens of thousands of orders, courier GPS pings, and kitchen status updates flood the system every minute.

Traditional database architectures and batch analytics frameworks (such as Apache Spark batch or Hadoop MapReduce) cannot handle these workloads at wire speed because:
1. They require persisting records to disk or distributed file systems before running queries.
2. Indexing every real-time event creates severe I/O bottlenecks.
3. Keeping complete event histories in memory is physically impossible as stream volume grows indefinitely.

**Phase 4** implements the core algorithmic solutions from **Syllabus Module 4 (Mining Data Streams)** to process high-velocity streams in bounded memory and sub-millisecond query time:
1. **Streaming Event Simulator (`order_stream_simulator.py`):** High-throughput producer emulating Apache Kafka topic streams with dynamic rush-hour surge multipliers and synthetic duplicate injection.
2. **Bloom Filter (`bloom_filter.py`):** Space-efficient probabilistic bit array for duplicate order suppression (network retries) and geofence verification in $O(k)$ time with zero false negatives.
3. **Flajolet-Martin Algorithm (`flajolet_martin.py`):** Logarithmic space count-distinct estimation of active delivery couriers using trailing zero bit analysis and median-of-means variance reduction.
4. **DGIM Algorithm (`dgim_window.py`):** Sliding-window power-of-2 bucket packing algorithm counting delayed orders over recent windows in $O(\log^2 N)$ memory with a guaranteed relative error bound $\le 50\%$.

---

## 2. Algorithmic Deep Dive & Mathematical Formulations

### 2.1 The Stream Data Model & Ingestion Simulator
- **File:** [`src/streaming/order_stream_simulator.py`](file:///e:/FoodDeliveryBigData/src/streaming/order_stream_simulator.py)
- **Architectural Role:** Emulates real-time event streaming without requiring an external Apache Kafka cluster on the student's Windows environment. Uses Python's `queue.Queue` with a non-blocking consumer API matching standard Kafka topics.
- **Dynamic Rush-Hour Velocity:**
  $$\text{Effective Rate} = \text{Base Rate} \times \begin{cases} 
  2.5 & \text{if } 12 \le \text{Hour} \le 14 \text{ (Lunch Rush)} \\
  3.2 & \text{if } 19 \le \text{Hour} \le 22 \text{ (Dinner Rush)} \\
  0.3 & \text{if } 0 \le \text{Hour} \le 5 \text{ (Late Night Lull)} \\
  1.0 & \text{otherwise}
  \end{cases}$$
- **Duplicate Injection:** Injects deliberate duplicate order IDs (5%–10% probability) to benchmark Bloom Filter suppression against simulated mobile app network retries.

---

### 2.2 Bloom Filter (Filtering Streams)
- **File:** [`src/streaming/bloom_filter.py`](file:///e:/FoodDeliveryBigData/src/streaming/bloom_filter.py)
- **Syllabus Topic:** *Module 4.3 — Filtering Streams: The Bloom Filter*.
- **Problem Solved:** Fast duplicate order suppression and geofence boundary verification without disk lookups.
- **Mathematical Foundations:**
  - Bit array of size $m$ bits, initially all 0.
  - $k$ independent hash functions:
    $$h_i(x) = \text{int}(\text{SHA256}(\text{Salt}_i \parallel x)) \pmod m$$
  - **Insertion (`add`):** Computes $k$ bit positions and sets them to 1.
  - **Query (`might_contain`):** Computes $k$ bit positions. If any bit is 0, item is **definitely not in set** (zero false negatives). If all bits are 1, item is **probably in set** (candidate duplicate).
- **False Positive Probability Derivation:**
  - Probability a specific bit remains 0 after 1 hash insertion: $1 - \frac{1}{m}$.
  - After $n$ items inserted with $k$ hash functions:
    $$P(\text{bit is 0}) = \left(1 - \frac{1}{m}\right)^{kn} \approx e^{-kn/m}$$
  - Probability a bit is 1: $1 - e^{-kn/m}$.
  - Probability all $k$ bits for an uninserted item are 1 (False Positive):
    $$P(\text{FP}) = \left(1 - e^{-kn/m}\right)^k$$
- **Optimal Hash Count:**
  $$k_{\text{opt}} = \frac{m}{n} \ln 2 \approx 0.693 \times \frac{m}{n}$$
- **Empirical Verification:** In test benchmarks on 1,000 uninserted queries, the observed false positive rate (1.80%) closely matched the theoretical expectation (1.65%) with 0 false negatives.

---

### 2.3 Flajolet-Martin Algorithm (Counting Distinct Elements)
- **File:** [`src/streaming/flajolet_martin.py`](file:///e:/FoodDeliveryBigData/src/streaming/flajolet_martin.py)
- **Syllabus Topic:** *Module 4.4 — Counting Distinct Elements in a Stream: Flajolet-Martin Algorithm*.
- **Problem Solved:** Estimating the number of unique active couriers and ordering customers across millions of streaming events in logarithmic memory $O(\log N)$ instead of unbounded $O(N)$ sets.
- **Mathematical Foundations:**
  - Deterministically hashes each identifier $x$ to 64-bit integer $y$.
  - Computes trailing zeros $\rho(y)$ (position of the least significant 1-bit).
  - The probability that $\rho(y) \ge r$ is $2^{-r}$.
  - Over $N$ distinct elements, the maximum observed trailing zeros $R = \max_x \rho(y)$ satisfies:
    $$E[2^R] \approx \phi N, \quad \text{where } \phi \approx 0.77351 \text{ (FM Constant)}$$
  - Raw single-hash estimate:
    $$\hat{N} = \frac{2^R}{\phi} \approx 1.2928 \times 2^R$$
- **Variance Reduction (Median-of-Means / AMS Technique):**
  - Uses $K = 20$ hash functions divided into $G = 5$ groups of $H = 4$ hashes each.
  - Computes group means: $\mu_g = \frac{1}{H} \sum_{j \in g} \hat{N}_j$.
  - Takes median across group means: $\hat{N}_{\text{final}} = \text{median}(\mu_1, \dots, \mu_G)$.
  - Averaging within groups reduces variance; taking the median across groups eliminates vulnerability to extreme outlier bit patterns.
- **Verification:** On a stream with 80 unique couriers repeated 5 times (400 stream items), the FM algorithm estimated 82.6 distinct couriers (**3.25% relative error**).

---

### 2.4 DGIM Algorithm (Counting in a Sliding Window)
- **File:** [`src/streaming/dgim_window.py`](file:///e:/FoodDeliveryBigData/src/streaming/dgim_window.py)
- **Syllabus Topic:** *Module 4.5 — Counting Ones in a Window: DGIM Algorithm*.
- **Problem Solved:** Estimating delayed order breaches in the last $N$ stream events ($N=1,000$) in $O(\log^2 N)$ memory instead of storing raw event timestamps.
- **DGIM Bucket Invariants:**
  1. Each bucket stores $(\text{timestamp}, \text{size})$.
  2. Every bucket size is an exact power of 2 ($1, 2, 4, 8, 16, \dots$).
  3. Bucket timestamps are strictly decreasing (newest bucket first).
  4. There are **at least 1 and at most 2 buckets of any given size**.
- **Ingestion & Merging Rules:**
  - When a `0` arrives: timestamp advances; no new bucket is formed.
  - When a `1` arrives: create bucket of size 1 at current timestamp.
  - If 3 buckets of size 1 exist, merge the two oldest into size 2.
  - If 3 buckets of size 2 exist, merge the two oldest into size 4 (cascading).
  - Drop buckets with $\text{timestamp} \le \text{current\_timestamp} - N$.
- **Query Rule for Last $k$ Events ($k \le N$):**
  $$\text{Count}(k) = \sum_{B_i \text{ inside } k} \text{size}(B_i) + \left\lfloor \frac{\text{size}(B_{\text{oldest}})}{2} \right\rfloor$$
- **Theoretical Error Bound Guarantee:**
  $$\text{Relative Error} \le \frac{\text{size}(B_{\text{oldest}}) / 2}{\text{Actual 1s}} \le \frac{2^{j-1}}{1 + \sum_{i=0}^{j-1} 2^i} = \frac{2^{j-1}}{2^j} \le \mathbf{50\%}$$
- **Verification:** In controlled tests with 200 random bits, DGIM estimated counts exhibited an empirical error of **7.5%**, well within the theoretical 50% limit, while reducing memory footprint by **91.2%**.

---

## 3. Complexity & Memory Comparison Matrix

| Algorithm | Syllabus Topic | Space Complexity | Time Complexity per Item | Query Time | Theoretical Error Bound |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Bloom Filter** | Filtering Streams | $O(m)$ bits | $O(k)$ bit operations | $O(k)$ | Zero false negatives; $P(\text{FP}) \le 2\%$ |
| **Flajolet-Martin** | Count-Distinct | $O(K \log \log N)$ bits | $O(K)$ hash ops | $O(K)$ | Median-of-means relative error $\le 25\%$ |
| **DGIM Algorithm** | Counting in Windows | $O(\log^2 N)$ bits | $O(\log N)$ bucket merges | $O(\log N)$ | Relative error strictly $\le \mathbf{50\%}$ |
| **Naive Set / Raw Buffer** | Baseline Reference | $O(N \cdot \text{item\_size})$ bits | $O(1)$ set insert | $O(N)$ scan | Exact (0% error), but memory explodes unbounded |

---

## 4. End-to-End Event Architecture & Streamlit Integration

The complete data stream mining flow operates seamlessly without blocking:

```mermaid
sequenceDiagram
    participant S as OrderStreamSimulator (Kafka Producer)
    participant Q as Local Stream Queue (Transport)
    participant BF as Bloom Filter (Duplicate Suppression)
    participant FM as Flajolet-Martin (Active Couriers)
    participant DGIM as DGIM Window (SLA Delay Tracking)
    participant UI as Streamlit Operations Dashboard

    S->>Q: Emit real-time JSON order events (with rush multiplier)
    UI->>Q: Ingest event batch
    Q->>BF: Ingest order_id
    alt Probable Duplicate (all k bits set)
        BF-->>UI: Flag candidate duplicate (network retry alert)
    else Unique Order
        BF-->>BF: Set k bits to 1
    end
    Q->>FM: Ingest courier_id (update trailing zero registers)
    Q->>DGIM: Ingest is_delayed bit (maintain power-of-2 buckets)
    FM-->>UI: Median-of-means distinct courier estimate
    DGIM-->>UI: Estimated delayed orders in window (<= 50% error)
    UI-->>UI: Render telemetry tables, bucket hierarchy, and FP benchmarks
```

### Dashboard UI Elements Added ([`app.py`](file:///e:/FoodDeliveryBigData/app.py) & [`src/dashboard/overview.py`](file:///e:/FoodDeliveryBigData/src/dashboard/overview.py)):
- **Simulation Control Panel:** Sliders for event count (25–300), duplicate injection rate (0%–25%), courier pool size (15–60), and rush-hour surge toggle.
- **Top Scorecards with Exact vs. Approximate Labels:**
  - `Stream Events Processed` [EXACT]
  - `Bloom Filter Duplicates` [PROBABILISTIC / APPROXIMATE]
  - `FM Active Couriers` [APPROXIMATE ESTIMATE]
  - `DGIM Delayed Orders` [GUARANTEED $\le 50\%$ ERROR]
- **Four Diagnostic Subtabs:**
  1. *Bloom Filter Deduplication:* Occupancy ratio, theoretical vs. empirical false positive benchmark, sample duplicate candidate table.
  2. *Flajolet-Martin Count-Distinct:* Estimate vs. exact count, relative error %, register values across all 5 groups.
  3. *DGIM Sliding Window:* Estimate vs. exact count, active power-of-2 bucket hierarchy table `(timestamp, size)`, memory reduction ratio.
  4. *Stream Telemetry Inspector:* Tabular view of streaming order event payloads.

---

## 5. Verification Results & Test Suite Status

The automated test suite in [`tests/test_streaming_algorithms.py`](file:///e:/FoodDeliveryBigData/tests/test_streaming_algorithms.py) verifies all mathematical and operational invariants across 14 focused tests:

```text
tests/test_streaming_algorithms.py::test_simulator_generates_valid_schema PASSED
tests/test_streaming_algorithms.py::test_simulator_deterministic_reproducibility PASSED
tests/test_streaming_algorithms.py::test_simulator_bounded_batch_execution PASSED
tests/test_streaming_algorithms.py::test_bloom_filter_zero_false_negatives PASSED
tests/test_streaming_algorithms.py::test_bloom_filter_duplicate_detection PASSED
tests/test_streaming_algorithms.py::test_bloom_filter_empirical_vs_theoretical_fp_rate PASSED
tests/test_streaming_algorithms.py::test_bloom_filter_invalid_parameters PASSED
tests/test_streaming_algorithms.py::test_flajolet_martin_trailing_zeros_calculation PASSED
tests/test_streaming_algorithms.py::test_flajolet_martin_distinct_count_estimation PASSED
tests/test_streaming_algorithms.py::test_flajolet_martin_empty_stream PASSED
tests/test_streaming_algorithms.py::test_dgim_bucket_invariants PASSED
tests/test_streaming_algorithms.py::test_dgim_window_expiration PASSED
tests/test_streaming_algorithms.py::test_dgim_relative_error_bound PASSED
tests/test_streaming_algorithms.py::test_dgim_query_sub_window PASSED

============================= 14 passed in 0.12s ==============================
```

**Full Repository Test Status:**
- `test_analytics.py`: 11 passed
- `test_data_generation.py`: 5 passed
- `test_mapreduce.py`: 5 passed
- `test_preprocessing.py`: 6 passed
- `test_spark_batch.py`: 5 passed
- `test_streaming_algorithms.py`: 14 passed
- **Total: 46 passed in 13.89 seconds (100% pass rate).**

---

## 6. How to Run & Verify on Windows

1. **Run Full Test Suite:**
   ```powershell
   python -m pytest -v
   ```
2. **Run Streaming Algorithm Tests Specifically:**
   ```powershell
   python -m pytest tests/test_streaming_algorithms.py -v
   ```
3. **Launch the Interactive Streamlit Dashboard:**
   ```powershell
   python -m streamlit run app.py
   ```
   Navigate to **"4. Mining Big Data Streams"** in the sidebar navigation and click **"▶️ Run Stream Simulation"**.
