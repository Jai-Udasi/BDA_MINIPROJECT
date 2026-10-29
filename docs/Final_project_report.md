# Big Data Analytics — Final Capstone Project Report

**Project Title:** Real-Time Food Delivery Demand Forecasting and Delay Prediction Using Big Data Analytics  
**Student Name:** Jai Udasi (Class: D20A, Roll No: 34)  
**Institution:** Department of Information Technology, Vivekanand Education Society's Institute of Technology (VESIT, Autonomous NEP Batch 1)  
**Academic Course:** Big Data Analytics (BDA)  
**Primary Case Study Reference:** `BDA_34_EXP1.pdf` (Experiment 1)  
**Syllabus Alignment:** Modules 1, 2, 3, 4, 5, and 6  
**Final Status:** COMPLETED, INTEGRATED & 100% VERIFIED (77/77 Unit Tests Passing)

---

## Executive Abstract

Online food delivery platforms process millions of orders across heterogeneous spatial-temporal environments characterized by sharp meal rush surges, road congestion, kitchen bottlenecks, and severe weather disruptions. Traditional centralized Relational Database Management Systems (RDBMS) suffer from write-lock contention and cannot scale to meet concurrent high-volume batch analytics and unbounded real-time event processing.

This capstone project delivers an end-to-end Big Data Analytics platform addressing the full university curriculum:
1. **Veracity Data Pipeline (Module 1):** Ingestion, schema standardization, and automated data cleaning across 2,500 multimodal order records.
2. **Distributed Batch Engine (Module 2):** PySpark DataFrames with strict `StructType` schemas, windowed velocity ranking, and Spark SQL analytics on OpenJDK 21 LTS.
3. **Hadoop MapReduce Engine (Module 3):** Standalone MapReduce job demonstrating Mapper emissions, Combiner local pre-aggregation saving **99.52%** network shuffle traffic, and Reducer zone aggregations.
4. **Stream Mining Algorithms (Module 4):** Sub-linear memory stream algorithms including a Bloom Filter for duplicate retry suppression (0 false negatives), Flajolet-Martin distinct active courier estimation in $O(\log N)$ space, and the DGIM sliding-window algorithm bounding delayed order counts with $\le 50\%$ relative error.
5. **Data Mining & Predictive ML (Module 5):** The two-pass Park-Chen-Yu (PCY) algorithm for frequent meal itemsets using a hash bucket bitmap; Haversine great-circle K-Means spatial clustering for courier staging hubs; a zero-data-leakage Random Forest delay risk classifier (ROC-AUC 0.891); and chronological hourly demand forecasting using Ridge regression with lag features (45.8% error reduction over baseline).
6. **Link Analysis, Recommenders & NoSQL Command Center (Modules 2 & 6):** PageRank and Kleinberg's HITS bipartite graph centrality; Content-Based and Collaborative Filtering recommendation engines; a 3-collection MongoDB operational persistence tier with transparent SQLite fallback; and a Unified Real-Time Operations Command Center.

The complete system is backed by **77 automated unit tests** achieving a 100% pass rate in ~33 seconds on Windows.

---

## 1. Problem Formulation & Operational Motivation

### 1.1 Case Study Context (`BDA_34_EXP1.pdf`)
Modern food delivery platforms operate in a highly volatile, real-time environment. Platform profitability and customer retention depend on delivery punctuality. In urban centers (such as Mumbai), delivery delays cascade rapidly due to:
- **Diurnal Demand Surges:** Extreme volume spikes during lunch (12:00 PM – 2:30 PM) and dinner (7:00 PM – 10:30 PM).
- **Spatial Clusters:** Commercial tech corridors (e.g. East Tech Park) experience concentrated surges that exhaust localized courier fleets.
- **Exogenous Disruptions:** Monsoonal rains and traffic bottlenecks increase travel times non-linearly.
- **Kitchen Preparation Bottlenecks:** Orders with high item counts cause preparation delays that exceed standard delivery buffers.

### 1.2 The Big Data Bottleneck
Traditional single-node relational databases fail to support this dual requirement:
- Analytical aggregations over millions of historical transactions lock disk tables, delaying transactional writes.
- Continuous streaming event streams exhaust connection pools and exceed single-node memory capacities.
- A decoupled, distributed Big Data architecture is required to isolate historical batch computation from wire-speed stream ingestion and low-latency operational persistence.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Data Sources
        S1["Historical Batch Orders<br/>(cleaned_orders.csv: 2,500 records)"]
        S2["Live Event Simulator<br/>(High-Throughput Kafka-style Stream)"]
        S3["Synthetic Coordinates & Baskets<br/>(500 Locations, 1,000 Transactions)"]
    end

    subgraph Core Distributed & Streaming Engines
        E1["Apache Spark Batch Engine<br/>(PySpark DataFrames, Windowing & Spark SQL)"]
        E2["Hadoop MapReduce Engine<br/>(Mapper, Combiner 99.52% Reduction, Reducer)"]
        E3["Streaming Mining Suite<br/>(Bloom Filter, Flajolet-Martin, DGIM Window)"]
    end

    subgraph Mining, ML & Graph Analytics
        M1["PCY Algorithm<br/>(Two-Pass Hash Bucket Bitmap Mining)"]
        M2["Spatial Clustering<br/>(Haversine Metric, Multi-K Silhouette)"]
        M3["Zero-Leakage Delay Classifier<br/>(Random Forest vs Logistic Regression)"]
        M4["Time-Series Demand Forecaster<br/>(Chronological Ridge + Lag Features)"]
        M5["Bipartite Graph Link Analysis<br/>(PageRank Centrality & HITS Hubs/Authorities)"]
        M6["Dual Recommendation Engines<br/>(Content-Based Cosine & Collaborative Filtering)"]
    end

    subgraph Persistence & Command Interface
        P1["MongoDB NoSQL Store<br/>(active_orders, courier_telemetry, zone_risk_scores)"]
        P2["Local Persistent Fallback<br/>(Embedded SQLite Store)"]
        UI["Unified Operations Command Center<br/>(Streamlit Interactive 6-Phase Dashboard)"]
    end

    S1 --> E1
    S1 --> E2
    S2 --> E3
    S3 --> M1
    S3 --> M2
    S1 --> M3
    S1 --> M4
    S1 --> M5
    S1 --> M6

    E1 & E2 & E3 & M1 & M2 & M3 & M4 & M5 & M6 --> P1
    P1 -.->|Auto Fallback| P2
    P1 & P2 --> UI
```

---

## 3. Dataset Characteristics, Preprocessing & Veracity Pipeline

### 3.1 Schema & Field Specifications
The primary historical dataset (`data/processed/cleaned_orders.csv`) consists of 2,500 records across 14 validated columns:

| Column Name | Data Type | Operational Role | Physical Domain Range |
| :--- | :--- | :--- | :--- |
| `order_id` | String | Unique Transaction Primary Key | `ORD-000001` to `ORD-002500` |
| `customer_id` | String | Customer Identifier | 500 distinct customers |
| `restaurant_id` | String | Restaurant Identifier | 50 distinct restaurants |
| `delivery_zone` | Categorical | Geographic delivery sector | 6 Mumbai urban zones |
| `order_timestamp` | Datetime | Order placement ISO timestamp | 2026-09-01 to 2026-10-01 |
| `order_amount` | Float | Gross order value in INR | ₹50.0 to ₹10,000.0 |
| `item_count` | Integer | Total food items ordered | 1 to 50 items |
| `delivery_fee` | Float | Base dispatch fee in INR | Derived from distance |
| `delivery_duration_min` | Float | Total delivery elapsed time | 5.0 to 180.0 minutes |
| `preparation_time_min` | Float | Kitchen cooking & packing duration | 5.0 to 120.0 minutes |
| `weather_condition` | Categorical | Ambient weather at order placement | Clear, Rainy, Windy, Heavy Rain |
| `traffic_density` | Categorical | Road traffic congestion level | Low, Medium, High, Jam |
| `order_status` | Categorical | Final order delivery state | Delivered, In Transit, Cancelled |
| `is_delayed` | Boolean | Ground truth SLA breach indicator | Deterministic SLA rule |

### 3.2 Automated Data Cleaning Pipeline (`clean_data.py`)
The raw ingestion pipeline audits and cleans incoming data:
- **Deduplication:** Dropped duplicate order records preserving the earliest timestamp.
- **Missing Value Imputation:** Group median imputation conditioned on `restaurant_type` and `delivery_zone`.
- **Dynamic SLA Ground Truth:** Enforces the deterministic service level threshold:
  $$\text{is\_delayed} = \text{delivery\_duration\_min} > (35 + 2.5 \times \text{delivery\_distance\_km})$$
- **Synthetic Separation:** To preserve the clean 14-column dataset, separate deterministic synthetic datasets were generated for market baskets (`synthetic_market_baskets.json`) and spatial coordinates (`synthetic_delivery_coordinates.csv`).

---

## 4. Distributed Batch Analytics: Apache Spark & MapReduce

### 4.1 Apache Spark (PySpark) Batch Engine (`spark_batch.py`)
- **Runtime:** OpenJDK 21 LTS, PySpark 4.2.0 initialized with local multi-core master `local[*]`.
- **DataFrame Transformations:** Single-pass distributed aggregations grouping by `delivery_zone` and `weather_condition`.
- **Window Analytics:** Rolling order velocities and spend ranks computed using `pyspark.sql.window.Window.partitionBy("delivery_zone").orderBy("order_timestamp")`.
- **Spark SQL Integration:** Temporary view `orders` created to execute ANSI SQL queries.

### 4.2 Hadoop MapReduce Paradigm (`order_volume_mr.py`)
- **Mapper:** Emits `(delivery_zone, (order_amount, 1, is_delayed))`.
- **Combiner:** Mini-reducer executing directly on worker partitions. Pre-aggregates intermediate records, shrinking network shuffle traffic from 2,500 records down to 12 partition records (**99.52% traffic reduction**).
- **Shuffle & Sort:** Emulates key sorting and hash partitioning.
- **Reducer:** Produces final aggregated totals, GMV, and delay percentages.
- **Cross-Engine Consistency:** Proved 100% numerical consistency between Spark SQL, MapReduce, and Pandas baselines.

---

## 5. Sub-Linear Streaming Algorithms (Module 4)

### 5.1 Bloom Filter (`bloom_filter.py`)
- **Purpose:** Suppresses duplicate mobile app retry requests in wire-speed memory.
- **Formula:** Bit array of size $m = 10,000$ bits with $k = 7$ independent MD5/SHA256 salted hash functions.
- **Guarantee:** **Zero false negatives** verified experimentally. Observed empirical false positive rate of 0.8% matches the theoretical bound $P(\text{FP}) \approx (1 - e^{-kn/m})^k$.

### 5.2 Flajolet-Martin Algorithm (`flajolet_martin.py`)
- **Purpose:** Estimates the count of distinct active couriers in $O(\log N)$ memory bits.
- **Mechanism:** Hashes stream identifiers to 64-bit binary strings and tracks maximum trailing zeros $R = \max(\rho(y))$.
- **Variance Reduction:** Grouped 32 hash functions into 8 buckets, computing the mean within buckets and median across buckets (Median-of-Means).

### 5.3 DGIM Sliding-Window Algorithm (`dgim_window.py`)
- **Purpose:** Tracks delayed orders over the most recent $N = 100$ stream events using $O(\log^2 N)$ bits.
- **Bucket Invariants:** Power-of-2 bucket sizes ($1, 2, 4, 8, \dots$). At most 2 buckets of any given size. When a 3rd bucket arrives, the two oldest merge into the next power of 2.
- **Error Bound:** The relative error is strictly bounded $\le 50\%$.

---

## 6. Advanced Data Mining & Machine Learning (Module 5)

### 6.1 Park-Chen-Yu (PCY) Frequent-Itemset Mining (`pcy_market_basket.py`)
- **Pass 1:** Counts individual items; hashes all candidate pairs into an integer bucket array $H$ of size $B = 5,000$.
- **Bitmap Reduction:** Converts the 32-bit integer array $H$ into a 1-bit bitmap where bit $b = 1$ if bucket count $\ge s$, reducing intermediate memory by $32\times$.
- **Pass 2 Pruning:** Only allocates pair counters for candidate pairs where $i \in L_1$, $j \in L_1$, and $\text{bitmap}[h(\{i, j\})] == 1$.
- **Discovered Rule:** `Chicken Biryani` $\implies$ `Gulab Jamun` (Support: 13.9%, Confidence: 68.8%, Lift: 3.41).

### 6.2 Spatial Delivery-Hub Clustering (`spatial_clustering.py`)
- **Metric:** Haversine great-circle spherical distance (km) accounting for Earth's curvature.
- **Optimization:** Hyperparameter search across $k \in [3, 8]$ using Inertia (WCSS) and Silhouette scores.
- **Result:** Discovered optimal $k = 6$ staging hubs (Silhouette = 0.841) matching Mumbai delivery nodes (Chembur, Kurla, Ghatkopar, Sion, Powai, Wadala).

### 6.3 Supervised Delay Classification (`delay_prediction.py`)
- **Zero Leakage:** Strictly excluded post-delivery features (`delivery_duration_min`, `order_status`). Preprocessors fitted exclusively on $X_{\text{train}}$.
- **Performance:** Random Forest Classifier achieved **Accuracy 83.6%**, Precision 80.5%, Recall 76.9%, F1 78.7%, and **ROC-AUC 0.891** (compared to Logistic Regression baseline of 76.4%).
- **Top Predictors:** Traffic density (24.2%), delivery distance (21.8%), kitchen prep time (18.5%).
- **Production Artifact:** Saved pipeline to `data/processed/models/delay_classifier.joblib` (1.7 MB).

### 6.4 Chronological Time-Series Demand Forecasting (`demand_forecasting.py`)
- **Continuous Grid:** Reconstructed continuous hourly index filling zero-order hours with 0 counts.
- **Temporal Split:** Strict chronological 80/20 forward-in-time train/test split.
- **Performance:** Ridge regression with lag features (`lag_1h` to `lag_24h`, rolling means) achieved **MAE 2.61 orders/hr** (a **45.8% error reduction** over the Naive Persistence Baseline MAE of 4.82).

---

## 7. Link Analysis, Recommenders & NoSQL Persistence (Modules 2 & 6)

### 7.1 Bipartite Graph Centrality: PageRank & HITS (`link_analysis_recommender.py`)
- **PageRank:** Power iteration ($d = 0.85$) over the customer-restaurant ordering graph identifying anchor restaurants capturing systemic demand. Conserves total probability of 1.0.
- **Kleinberg's HITS:** Mutual reinforcement computing Authority scores for popular dining destinations and Hub scores for active customer taste-makers.

### 7.2 Dual Recommendation Engines (`link_analysis_recommender.py`)
- **Content-Based Filtering:** Vector cosine similarity on restaurant profiles (cuisine, price tier, delivery speed, prep time) matching customer past preferences.
- **Collaborative Filtering:** Customer-restaurant interaction matrix affinity recommending unvisited spots based on peer ordering habits.
- **Cold-Start Handling:** Graceful fallback to regional volume baselines for unseen customers.

### 7.3 MongoDB Operational Store (`mongo_manager.py`)
- **Collections:** `active_orders` (in-flight state), `courier_telemetry` (GeoJSON coordinates with 2dsphere indexing), `zone_risk_scores` (alert thresholds).
- **Security:** URIs and credentials configured via environment variables (`MONGODB_URI`, `MONGODB_DB_NAME`).
- **Zero-Failure Fallback:** Embedded SQLite fallback backend ensuring seamless demonstration even without an active MongoDB server.

---

## 8. Unified Operations Command Center

The interactive Streamlit dashboard (`app.py` and `overview.py`) unites all six phases into an operational command cockpit:
- **Section A (Live Event Stream):** Wire-speed simulated Kafka events with live Bloom Filter, Flajolet-Martin, and DGIM metrics.
- **Section B (Demand & Zone Overview):** Zone order volumes, multi-horizon forecasts (1h, 3h, 24h), and spatial courier staging map.
- **Section C (Proactive Alerting):** Configurable threshold slider (default: 75%) with instant red/amber SLA breach warnings.
- **Section D (Recommendation Showcase):** PCY frequent itemset rules and personalized cuisine matches.
- **Section E (System Health):** Real-time diagnostic status across Spark, MapReduce, streaming, ML, and MongoDB tiers.

---

## 9. Comprehensive Test Verification Summary

The project test suite contains **77 automated unit tests** across 12 test modules:

| Test Module | Tests | Verified Scope | Result |
| :--- | :---: | :--- | :---: |
| `tests/test_data_generation.py` | 5 | Reproducibility, schema conformity, physical bounds | **PASSED** |
| `tests/test_preprocessing.py` | 6 | Deduplication, group median imputation, dynamic SLA | **PASSED** |
| `tests/test_analytics.py` | 11 | KPI formulas, outlier IQR detection, dashboard filters | **PASSED** |
| `tests/test_mapreduce.py` | 5 | Mapper, Combiner traffic reduction, Reducer | **PASSED** |
| `tests/test_spark_batch.py` | 5 | PySpark DataFrames, window ranking, Spark SQL | **PASSED** |
| `tests/test_streaming_algorithms.py` | 14 | Bloom Filter zero FN, FM trailing zeros, DGIM bound | **PASSED** |
| `tests/test_pcy_market_basket.py` | 6 | Two-pass hash bitmap, association rules, exact counter | **PASSED** |
| `tests/test_spatial_clustering.py` | 4 | Haversine distance, K-Means multi-k silhouette | **PASSED** |
| `tests/test_delay_prediction.py` | 4 | Zero data leakage, Random Forest classification | **PASSED** |
| `tests/test_demand_forecasting.py` | 4 | Continuous grid, chronological splitting, lag features | **PASSED** |
| `tests/test_mongo_manager.py` | 7 | MongoDB upsert, indexes, SQLite fallback | **PASSED** |
| `tests/test_link_analysis_recommender.py` | 6 | PageRank convergence, HITS, dual recommenders | **PASSED** |
| **TOTAL** | **77** | **Full Project Test Suite (Executed in ~33s)** | **100% PASS** |

---

## 10. Honest Technical Limitations & Future Work

1. **Streaming Broker:** Order events are simulated locally with Kafka semantics; a distributed multi-broker cluster was not physically deployed.
2. **Item-Level Data:** Co-ordered food items were generated deterministically to maintain the historical orders table schema.
3. **Cluster Scale:** Apache Spark executed in local multi-threaded mode (`local[*]`) rather than on a multi-node Hadoop YARN cluster.
4. **Future Enhancements:** Deploying Apache Kafka and Apache Flink for distributed CEP, training Neural Collaborative Filtering (NCF) recommenders, and integrating live OpenStreetMap GPS telemetry.

---

## 11. Conclusion

This capstone project successfully demonstrates a fully functional, mathematically sound Big Data Analytics architecture. By combining historical batch processing (Apache Spark, MapReduce) with sub-linear streaming algorithms (Bloom Filter, Flajolet-Martin, DGIM), predictive machine learning, link analysis, and operational NoSQL persistence, the system provides a scalable, enterprise-grade solution for modern food delivery operations.

*Report submitted by Jai Udasi (Roll No: 34, Class: D20A, Department of Information Technology, VESIT).*
