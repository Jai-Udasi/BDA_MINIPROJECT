# Big Data Analytics — Comprehensive College Viva & Defence Guide

**Project Title:** Real-Time Food Delivery Demand Forecasting and Delay Prediction Using Big Data Analytics  
**Student Name:** Jai Udasi (Class: D20A, Roll No: 34)  
**Institution:** Department of Information Technology, Vivekanand Education Society's Institute of Technology (VESIT, Autonomous NEP Batch 1)  
**Academic Course:** Big Data Analytics (Course Code: BDA)  
**Curriculum Scope:** Complete BDA Syllabus Coverage (Modules 1, 2, 3, 4, 5, and 6 Verified)  
**Test Suite Coverage:** 77 Passing Unit Tests (100% Pass Rate)

---

## Table of Contents
1. [Topic 1: Big Data Characteristics & The 5Vs (Module 1)](#topic-1-big-data-characteristics--the-5vs)
2. [Topic 2: Hadoop Architecture & MapReduce Paradigm (Module 3)](#topic-2-hadoop-architecture--mapreduce-paradigm)
3. [Topic 3: Apache Spark & PySpark Batch Engine (Module 2)](#topic-3-apache-spark--pyspark-batch-engine)
4. [Topic 4: NoSQL Databases & MongoDB Persistence (Module 2)](#topic-4-nosql-databases--mongodb-persistence)
5. [Topic 5: Mining Big Data Streams (Module 4)](#topic-5-mining-big-data-streams)
6. [Topic 6: Frequent Pattern Mining & PCY Algorithm (Module 5)](#topic-6-frequent-pattern-mining--pcy-algorithm)
7. [Topic 7: Spatial Clustering & Distance Metrics (Module 5)](#topic-7-spatial-clustering--distance-metrics)
8. [Topic 8: Supervised Delay Classification & Data Leakage (Module 5)](#topic-8-supervised-delay-classification--data-leakage)
9. [Topic 9: Time-Series Demand Forecasting (Module 5)](#topic-9-time-series-demand-forecasting)
10. [Topic 10: Link Analysis — PageRank & HITS (Module 6)](#topic-10-link-analysis--pagerank--hits)
11. [Topic 11: Recommendation Engines — Content & Collaborative (Module 6)](#topic-11-recommendation-engines)
12. [Topic 12: System Architecture, Testing & Limitations](#topic-12-system-architecture-testing--limitations)

---

## Topic 1: Big Data Characteristics & The 5Vs

### Q1: What are the 5Vs of Big Data, and how are they modeled in your project?
**Answer:**
1. **Volume:** Thousands of historical order records and continuous streaming logs. Modeled in `data/processed/cleaned_orders.csv` (2,500 records) and scalable to multi-gigabyte partitions.
2. **Velocity:** Real-time bursts during lunch (12:00–14:00) and dinner (19:00–22:00) peak surges, emulated via our high-throughput streaming generator.
3. **Variety:** Multimodal schema combining numerical amounts, temporal timestamps, spatial delivery zones, coordinates, and categorical weather/traffic tags.
4. **Veracity:** Handled by our automated cleaning pipeline (`src/preprocessing/clean_data.py`), which deduplicates records, imputes missing values via group medians, and dynamically validates SLA delay flags.
5. **Value:** Actionable predictive insights that prevent courier idle time, alleviate kitchen bottlenecks, and reduce SLA delay rates.

### Q2: Why does a traditional RDBMS fail to scale for food delivery big data?
**Answer:**
A traditional RDBMS relies on ACID transactions, rigid normalized schemas, and centralized disk-based storage. Under millions of concurrent order events:
- Write bottlenecks occur due to row-level and table-level locking.
- Heavy analytical aggregation queries lock tables and block real-time order transactions.
- Horizontal scaling across multi-node clusters requires complex sharding that degrades relational join performance.
Big Data architectures decouple real-time write ingestion (NoSQL / Streams) from distributed analytical batch processing (Spark / HDFS).

### Q3: What is the difference between Batch Processing and Stream Processing?
**Answer:**
- **Batch Processing:** Computes over large, bounded, historical datasets stored persistently on disk. Optimized for high throughput, comprehensive aggregations, and iterative model training (e.g., PySpark, Hadoop MapReduce).
- **Stream Processing:** Computes over unbounded, continuous, high-velocity data arriving at wire speed. Optimized for sub-second latency and memory-bounded approximate summaries (e.g., Bloom Filter, Flajolet-Martin, DGIM).

### Q4: How did your veracity pipeline handle anomalies in the raw dataset?
**Answer:**
In `src/preprocessing/clean_data.py`:
1. Deduplication: Identified and dropped duplicate `order_id` records.
2. Timestamp Parsing: Standardized mixed timestamp formats and dropped unparseable timestamps.
3. Imputation: Replaced missing numerical fields (`order_amount`, `preparation_time_min`) with the median of their specific `restaurant_type` or `delivery_zone`.
4. Boundary Enforcement: Clipped negative amounts and impossible distances.
5. SLA Delay Ground Truth: Enforced the deterministic SLA formula:
   $$\text{is\_delayed} = \text{delivery\_duration\_min} > (35 + 2.5 \times \text{delivery\_distance\_km})$$

---

## Topic 2: Hadoop Architecture & MapReduce Paradigm

### Q5: What are the core distributed stages of Hadoop MapReduce?
**Answer:**
1. **Map Stage:** Reads input splits and emits intermediate key-value pairs $(k_1, v_1)$.
2. **Combiner Stage:** An optional local mini-reducer running directly on mapper partitions to pre-aggregate intermediate tuples before the network transfer.
3. **Shuffle & Sort Stage:** Clusters and sorts all mapper outputs by key across cluster nodes.
4. **Reduce Stage:** Iterates over grouped values for each key to produce final aggregated outputs $(k_2, v_2)$.

### Q6: What did your custom MapReduce implementation emit, and how was it structured?
**Answer:**
In `src/mapreduce/order_volume_mr.py`:
- **Mapper:** Emits `(delivery_zone, (order_amount, 1, is_delayed))`.
- **Combiner:** Pre-aggregates tuples locally on each chunk, summing amounts and counts.
- **Shuffle & Sort:** Emulates key-based partitioning and sorting.
- **Reducer:** Calculates final total orders, GMV, Average Order Value (AOV), and delay percentage per delivery zone.

### Q7: What is the role of a Combiner, and how much traffic did it save in your project?
**Answer:**
The Combiner performs local pre-aggregation on mapper nodes to minimize the volume of data transmitted over the cluster network during the Shuffle & Sort phase.
In our execution (`src/mapreduce/order_volume_mr.py`), the Combiner condensed 2,500 raw order tuples down to 12 partition tuples across 2 chunks, achieving a **99.52% reduction in network shuffle traffic**.

### Q8: What are the two core daemons of Hadoop 2.x/3.x YARN?
**Answer:**
1. **ResourceManager (RM):** The global cluster authority that allocates CPU and memory resources across all applications.
2. **NodeManager (NM):** The per-machine agent responsible for monitoring container resource utilization (CPU, memory, disk) and reporting to the ResourceManager.

---

## Topic 3: Apache Spark & PySpark Batch Engine

### Q9: Why is Apache Spark faster than traditional Hadoop MapReduce?
**Answer:**
1. **In-Memory Computing:** Retains intermediate RDD / DataFrame partitions in RAM across iterative transformations, avoiding MapReduce's expensive disk writes between stages.
2. **DAG Execution Engine:** Constructs an optimized multi-stage Directed Acyclic Graph (DAG) with pipelined transformations rather than rigid 2-stage Map-then-Reduce barriers.
3. **Catalyst Optimizer:** Analyzes logical query plans, applies predicate pushdown and column projection, and generates optimal physical execution code.

### Q10: What is the difference between Spark Transformations and Actions?
**Answer:**
- **Transformations:** Lazy operations that define a new DataFrame from an existing one without immediate execution (e.g. `filter()`, `select()`, `groupBy()`, `withColumn()`).
- **Actions:** Eager operations that trigger computation across the DAG and return results to the driver or write to storage (e.g. `count()`, `collect()`, `show()`, `write.parquet()`).

### Q11: How did you use PySpark Window Functions in your project?
**Answer:**
In `src/analytics/spark_batch.py`, we defined window specifications using `pyspark.sql.window.Window`:
- Partitioned by `delivery_zone` and ordered by `order_timestamp` to compute rolling order velocities and rank the highest-spending orders per zone without collapsing rows.

### Q12: How was Spark SQL integrated with PySpark DataFrames?
**Answer:**
We registered the DataFrame as a temporary view:
```python
df.createOrReplaceTempView("orders")
spark.sql("""
    SELECT delivery_zone, COUNT(*) as volume, AVG(order_amount) as aov
    FROM orders
    GROUP BY delivery_zone
    ORDER BY volume DESC
""")
```
This enables standard ANSI SQL analytics over distributed partitions.

---

## Topic 4: NoSQL Databases & MongoDB Persistence

### Q13: What are the four primary NoSQL database categories?
**Answer:**
1. **Document Stores:** MongoDB, CouchDB (stores semi-structured JSON/BSON documents).
2. **Key-Value Stores:** Redis, DynamoDB (associative key-value pairs).
3. **Column-Family Stores:** Apache Cassandra, HBase (wide-column tables for high-write time series).
4. **Graph Databases:** Neo4j (nodes, edges, and relationship properties).

### Q14: Explain the design of your three MongoDB collections.
**Answer:**
In `src/nosql/mongo_manager.py`:
1. `active_orders`: Holds in-flight orders with fields `order_id`, `delivery_zone`, `status`, `order_amount`, and `predicted_delay_risk`.
2. `courier_telemetry`: Stores courier GPS coordinates (`latitude`, `longitude`), `active_order_id`, and `battery_pct` with GeoJSON `Point` geometry.
3. `zone_risk_scores`: Maintains calculated zone risk scores (0.0 to 1.0) and alert levels (`Normal`, `Critical`).

### Q15: Why are secondary indexes and GeoJSON 2dsphere indexes important in MongoDB?
**Answer:**
Without indexes, MongoDB must execute an expensive $O(N)$ full collection scan (COLLSCAN) for every query.
- A unique index on `order_id` ensures idempotent upserts and $O(\log N)$ point lookups.
- A `2dsphere` index on `courier_telemetry.location` enables fast spherical geospatial queries (`$near`, `$geoWithin`) to locate the closest available couriers to a restaurant.

### Q16: How did you handle environment configuration and local fallback for MongoDB?
**Answer:**
- Connection URIs and database names are loaded dynamically from environment variables (`MONGODB_URI`, `MONGODB_DB_NAME`) without hardcoding credentials.
- If MongoDB is unreachable, `MongoManager` automatically activates a local embedded SQLite fallback store with **identical public method signatures**, ensuring zero disruption during demonstrations.

---

## Topic 5: Mining Big Data Streams

### Q17: What is a Bloom Filter, and can it produce false negatives?
**Answer:**
A Bloom Filter is a space-efficient probabilistic data structure testing set membership in $O(k)$ time using $m$ bits and $k$ independent hash functions.
- **Zero False Negatives:** If an element was inserted, all $k$ corresponding bits were set to 1 and are never cleared. Thus, a query will strictly return `True`.
- **False Positives:** Random hash collisions from other inserted elements can cause all $k$ bits of an uninserted item to be 1, resulting in a false positive.

### Q18: What is the optimal number of hash functions $k$ in a Bloom Filter?
**Answer:**
Given bit array size $m$ and expected items $n$:
$$k = \frac{m}{n} \ln 2 \approx 0.693 \frac{m}{n}$$
This balances bit dispersion and minimizes the theoretical false positive probability $P(\text{FP}) \approx (1 - e^{-kn/m})^k$.

### Q19: How does the Flajolet-Martin (FM) Algorithm estimate distinct counts in logarithmic space?
**Answer:**
Each element identifier is hashed to a uniform binary string. The function $\rho(y)$ measures the number of trailing zeros.
- The probability of observing $r$ trailing zeros is $2^{-(r+1)}$.
- If the maximum trailing zeros observed is $R$, the raw distinct count estimate is $2^R / \phi$ where $\phi \approx 0.77351$.
- To reduce variance from powers-of-two granularity, we use multiple hash functions grouped together, taking the mean within groups and the median across groups (Median-of-Means).

### Q20: Explain the DGIM Algorithm and how it counts 1s in a sliding window with bounded error.
**Answer:**
The DGIM algorithm tracks 1-bits in the last $N$ stream events using $O(\log^2 N)$ bits of memory:
- **Bucket Invariant:** Each bucket has a timestamp and a size that is a power of 2. There are at least 1 and at most 2 buckets of any given size.
- **Cascading Merges:** When a 3rd bucket of size $2^j$ is formed, the two oldest merge into size $2^{j+1}$.
- **Bounded Error:** When querying the last $k$ events, the oldest bucket may overlap the window boundary. Adding half its size ensures the relative error is strictly bounded:
  $$\text{Relative Error} \le \frac{2^{j-1}}{2^j} = 50\%$$

---

## Topic 6: Frequent Pattern Mining & PCY Algorithm

### Q21: What is the combinatorial bottleneck of the classic Apriori algorithm?
**Answer:**
In Pass 1, Apriori finds frequent single items $L_1$. In Pass 2, it forms the candidate pair set $C_2 = L_1 \times L_1$ and maintains an integer counter for every pair. If $|L_1| = 10^5$, $|C_2| \approx 5 \times 10^9$ candidate pairs, requiring $\approx 20\text{ GB}$ of RAM and exhausting system memory.

### Q22: How does the Park-Chen-Yu (PCY) algorithm optimize memory during Pass 1 and Pass 2?
**Answer:**
- **Pass 1:** PCY utilizes idle RAM to maintain a hash table of $B$ integer buckets. For every basket, all item pairs are hashed to a bucket using MD5, incrementing the bucket counter.
- **Bitmap Construction:** Compresses the integer bucket array into a bit array (bitmap) where bit $b = 1$ if bucket count $\ge s$ (support threshold), achieving a $32\times$ memory reduction.
- **Pass 2 Pruning:** A pair $\{i, j\}$ is counted in Pass 2 if and only if $i \in L_1$, $j \in L_1$, and $\text{bitmap}[h(\{i, j\})] == 1$.

### Q23: Define Support, Confidence, and Lift in Association Rule Mining.
**Answer:**
For rule $A \implies B$:
1. **Support:** Fraction of total transactions containing both $A$ and $B$: $\frac{\text{Count}(A \cup B)}{N}$.
2. **Confidence:** Conditional probability of buying $B$ given $A$: $\frac{\text{Support}(A \cup B)}{\text{Support}(A)}$.
3. **Lift:** Ratio of observed co-occurrence to expected co-occurrence if independent:
   $$\text{Lift} = \frac{\text{Confidence}(A \implies B)}{\text{Support}(B)}$$
   - $\text{Lift} > 1$: Strong positive complementarity (e.g. Burger + Fries).
   - $\text{Lift} = 1$: Independent items.
   - $\text{Lift} < 1$: Substitute / negatively correlated items.

---

## Topic 7: Spatial Clustering & Distance Metrics

### Q24: Why is Euclidean distance invalid for spatial clustering of delivery locations?
**Answer:**
Euclidean distance assumes a flat 2D plane ($\sqrt{\Delta x^2 + \Delta y^2}$). Because Earth is spherical, 1 degree of latitude is roughly constant ($\approx 111\text{ km}$), but 1 degree of longitude shrinks towards the poles ($\approx 111 \cdot \cos(\text{latitude})\text{ km}$). Euclidean distance introduces severe geometric distortion.

### Q25: State the Haversine Great-Circle Distance formula.
**Answer:**
$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
$$d = 2 R \cdot \arcsin\left(\sqrt{a}\right)$$
where $\phi_1, \phi_2$ are latitudes in radians, $\Delta \lambda$ is longitude difference, and $R = 6371.0\text{ km}$ is Earth's mean radius.

### Q26: How do Inertia and Silhouette Score help select the optimal number of clusters $k$?
**Answer:**
- **Inertia (WCSS):** Measures internal cluster compactness. The Elbow Method looks for the point where the marginal drop in inertia flattens out.
- **Silhouette Score ($S \in [-1, 1]$):** Compares mean intra-cluster distance $a(i)$ to nearest-cluster distance $b(i)$:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  A score near $+1$ indicates dense, well-separated clusters. In our project, $k = 6$ produced the highest silhouette score (0.841).

---

## Topic 8: Supervised Delay Classification & Data Leakage

### Q27: What is Data Leakage in Machine Learning, and how did you prevent it?
**Answer:**
Data leakage occurs when information from the target variable or information unavailable at prediction time is included in training features, causing inflated test accuracy that fails in production.
- **Excluded Features:** We strictly excluded `delivery_duration_min` (which directly determines delay) and `order_status`.
- **Train-Only Preprocessing:** `StandardScaler` and `OneHotEncoder` were fitted strictly on $X_{\text{train}}$ and then used to transform $X_{\text{test}}$, preventing test distribution parameters from leaking into training.

### Q28: What pre-delivery features were used for delay risk classification?
**Answer:**
Only attributes known at order placement:
`delivery_distance_km`, `preparation_time_min`, `item_count`, `order_hour`, `weather_condition`, `traffic_density`, and `delivery_zone`.

### Q29: Compare your Logistic Regression and Random Forest classifier results.
**Answer:**
- **Logistic Regression (Linear Baseline):** Accuracy 76.4%, ROC-AUC 0.824.
- **Random Forest Classifier (Ensemble):** Accuracy **83.6%**, Precision **80.5%**, Recall **76.9%**, F1-Score **78.7%**, ROC-AUC **0.891**.
Random Forest captured non-linear interaction terms between severe weather and high traffic density.

### Q30: What were the top three features driving delivery delays?
**Answer:**
1. `traffic_density_High` (24.2% importance): Congestion creates transit bottlenecks.
2. `delivery_distance_km` (21.8% importance): Longer delivery routes multiply delay risk.
3. `preparation_time_min` (18.5% importance): Kitchen backlogs leave couriers with insufficient transit buffers.

---

## Topic 9: Time-Series Demand Forecasting

### Q31: Why is standard random K-Fold cross-validation invalid for time-series forecasting?
**Answer:**
Random splitting assumes independent and identically distributed (i.i.d.) observations. Time series data has strong temporal autocorrelation. Randomly shuffling data leaks future rush periods (e.g., 8:00 PM dinner peak) into training to predict past periods (e.g., 2:00 PM lunch), creating catastrophic lookahead bias. We enforced a **strict chronological 80/20 train/test split**.

### Q32: Why must zero-order hours be explicitly filled in food delivery time series?
**Answer:**
Transactional order logs only record rows when an order occurs. If no orders occur between 3:00 AM and 4:00 AM, that hour is absent from raw data.
Without zero-order re-indexing:
- Time intervals become irregular.
- Lag features (`lag_1h`) mistakenly reference earlier non-consecutive hours.
- Off-peak demand is severely overestimated.
We reconstructed an unbroken hourly grid with `order_count = 0` for silent hours.

### Q33: What lag features did you engineer, and what does each capture?
**Answer:**
- `lag_1h`: Immediate short-term persistence.
- `lag_2h`, `lag_3h`: Momentum of lunch/dinner rush build-up.
- `lag_24h`: Daily seasonal cycle at the exact same hour yesterday.
- `rolling_mean_3h`: Noise-damped moving average.
- `hour_of_day`: Diurnal bimodal cycle.
- `is_weekend`: Higher weekend evening order volumes.

### Q34: How did your Ridge forecasting model compare against the Naive Baseline?
**Answer:**
- **Naive Persistence Baseline ($\hat{y}_{t+1} = y_t$):** MAE 4.82, RMSE 6.45 orders/hour.
- **Ridge Regression with Lag Features:** MAE **2.61**, RMSE **3.48** orders/hour.
The Ridge model achieved a **45.8% reduction in forecasting error** over the persistence baseline.

---

## Topic 10: Link Analysis — PageRank & HITS

### Q35: How is PageRank computed on a bipartite customer-restaurant graph?
**Answer:**
The bipartite graph treats customers and restaurants as nodes, with directed edges representing order transactions.
PageRank power iteration updates each node $p$:
$$PR_{t+1}(p) = \frac{1 - d}{N} + d \sum_{q \in M(p)} \frac{PR_t(q)}{L(q)}$$
where $d = 0.85$ is the damping factor, $N$ is total nodes, and $L(q)$ is the out-degree of node $q$. Total network probability strictly sums to 1.0.

### Q36: What is Kleinberg's HITS algorithm, and what do Hubs and Authorities represent?
**Answer:**
HITS models mutual reinforcement between two node classes:
- **Authorities ($a$):** Prominent restaurants that attract repeated orders from influential customers.
- **Hubs ($h$):** Active customers who order from diverse, high-rating restaurants.
Power iteration on adjacency matrix $A$:
$$a_{t+1} = A^T h_t, \quad h_{t+1} = A a_{t+1}$$
Both vectors are $L_2$ normalized at each iteration until convergence.

### Q37: How does PageRank differ fundamentally from HITS?
**Answer:**
1. **Scope:** PageRank computes a single global centrality score using a random walk with teleportation. HITS computes two distinct dual scores (Hub and Authority).
2. **Matrix Formulation:** PageRank iterates on a row-stochastic transition probability matrix $M$. HITS iterates directly on the adjacency matrix $A$ and its transpose $A^T$.
3. **Normalization:** PageRank uses $L_1$ normalization (probabilities sum to 1). HITS uses $L_2$ vector normalization.

---

## Topic 11: Recommendation Engines

### Q38: What is Content-Based Filtering, and how is cosine similarity computed?
**Answer:**
Content-Based Filtering recommends items based on the similarity between an item's feature profile and a user's past taste profile.
For user profile vector $\vec{u}$ and restaurant feature vector $\vec{r}$:
$$\text{Cosine Similarity} = \frac{\vec{u} \cdot \vec{r}}{\|\vec{u}\| \|\vec{r}\|}$$
Features include cuisine tags, average ticket size, prep time, and delivery zone.

### Q39: What is Collaborative Filtering, and how does it predict unvisited items?
**Answer:**
Collaborative Filtering recommends items based on the collective behavior of similar users.
1. Constructs a customer $\times$ restaurant interaction frequency matrix.
2. Computes pairwise customer similarity using normalized cosine similarity.
3. For target user $u$ and unvisited restaurant $r$, predicts affinity as:
   $$\hat{R}(u, r) = \frac{\sum_{v \in \text{Peers}} \text{Sim}(u, v) \cdot R(v, r)}{\sum_{v \in \text{Peers}} |\text{Sim}(u, v)|}$$

### Q40: How does your system handle the Cold-Start problem for new customers?
**Answer:**
When an unseen customer ID enters the system:
- User interaction history is empty, making collaborative filtering undefined.
- The engine detects the cold-start state and gracefully falls back to top-k popular restaurants by regional volume and highest PageRank authority scores until order history is established.

---

## Topic 12: System Architecture, Testing & Limitations

### Q41: How many automated unit tests exist in the project, and what is the pass rate?
**Answer:**
The project contains **77 automated unit tests** across 12 test modules covering data generation, cleaning, analytics, MapReduce, PySpark batch, streaming algorithms, PCY mining, spatial clustering, delay ML, demand forecasting, NoSQL persistence, and link analysis. The pass rate is **100% (77 passed, 0 failed in ~33s)**.

### Q42: What honest technical boundaries exist in your project?
**Answer:**
1. **In-Memory Streaming:** Emulates Kafka topic semantics locally rather than connecting to a physical multi-broker Kafka cluster.
2. **Synthetic Demonstration Data:** Menu items and geographic coordinates were generated synthetically to preserve the clean 14-column orders table.
3. **Local Spark:** PySpark executes on a local multi-core master (`local[*]`) rather than an external distributed Hadoop cluster.
4. **NoSQL Fallback:** The persistence tier runs on embedded SQLite when an external MongoDB daemon is absent.

### Q43: How do the six phases map to your college BDA syllabus?
**Answer:**
- **Phase 1 (Module 1):** Big Data 5Vs & Veracity Cleaning.
- **Phase 2 (Module 2):** Exploratory Data Analytics & Spark Environment Diagnostics.
- **Phase 3 (Module 2 & 3):** PySpark DataFrame Batch Engine & Hadoop MapReduce.
- **Phase 4 (Module 4):** Mining Big Data Streams (Bloom Filter, Flajolet-Martin, DGIM).
- **Phase 5 (Module 5):** PCY Frequent Itemsets, Haversine K-Means, Delay ML, Demand Forecasting.
- **Phase 6 (Module 2 & 6):** MongoDB Operational Store, PageRank, HITS, Dual Recommenders & Operations Command Center.

### Q44: What is the difference between a False Positive and a False Negative in delay prediction?
**Answer:**
- **False Positive (Type I Error):** Model predicts an order will be delayed, but it arrives on time. Leads to unnecessary courier re-routing or premature kitchen alerts.
- **False Negative (Type II Error):** Model predicts an order will arrive on time, but it breaches the SLA. Leads to customer complaints, refunds, and driver shortages.
In our model, we balanced both by tuning classification thresholds, achieving a Precision of 80.5% and Recall of 76.9% (F1 = 78.7%).

### Q45: How does the Unified Operations Command Center synthesize these phases?
**Answer:**
The Command Center (Tab 6 in Streamlit) acts as the operational nerve center:
- Displays wire-speed event counts and Bloom/FM/DGIM streaming telemetry.
- Highlights zones exceeding real-time delay risk thresholds (default: 75%).
- Overlays spatial staging hubs and multi-horizon demand projections.
- Surfaces PCY combo recommendations and personalized cuisine matches.
- Monitors system health across the Spark engine, MapReduce, ML models, and MongoDB persistence tier.

---
*Comprehensive College Viva Guide prepared for University Examination & Defence.*
