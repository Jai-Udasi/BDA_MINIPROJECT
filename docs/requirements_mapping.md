# Requirements & Syllabus Mapping Document

## Project Metadata
- **Project Title:** Real-Time Food Delivery Demand Forecasting and Delay Prediction Using Big Data Analytics
- **Student Name & Roll:** Jai Udasi (Class: D20A, Roll No: 34)
- **Institution:** Vivekanand Education Society's Institute of Technology (VESIT) — Department of Information Technology (NEP Batch 1)
- **Course:** Big Data Analytics (BDA)
- **Current Milestone:** Phase 6 Complete (Full Project Verified: MongoDB Operational Store, Link Analysis, PageRank, HITS, Dual Recommenders & Real-Time Operations Command Center)

---

## 1. Executive Summary & Problem Formulation

### 1.1 Problem Statement (from `BDA_34_EXP1.pdf`)
Modern online food delivery platforms (such as Zomato, Swiggy, and DoorDash) process millions of food orders every day. Order demand fluctuates rapidly and non-linearly across:
- **Temporal Rush Windows:** Bimodal lunch (12:00 PM – 2:30 PM) and dinner rushes (7:00 PM – 10:30 PM).
- **Spatial Locations:** Urban clusters, tech parks, waterfronts, and university campuses.
- **Environmental Disruptions:** Thunderstorms, heavy rainfall, road traffic congestion.
- **Kitchen Operational Bottlenecks:** Food preparation backlogs and item counts.

Simultaneously, delivery delays degrade customer retention. Traditional centralized relational database management systems (RDBMS) fail to scale under high-volume batch analytics combined with continuous, high-velocity streaming events. This system establishes a scalable Big Data architecture combining historical batch analytics with real-time stream ingestion, time-series demand forecasting, machine learning delay risk classification, and interactive operational dashboards.

---

## 2. Comprehensive Syllabus Mapping Matrix (`bda syllabus.pdf`)

Below is the verified matrix mapping every curriculum module to its concrete project implementation and current phase status:

| Syllabus Module & Topics | Core Syllabus Concepts | Practical Project Implementation | Current Status |
| :--- | :--- | :--- | :--- |
| **Module 1: Introduction to Big Data** | • Characteristics (5Vs: Volume, Velocity, Variety, Veracity, Value)<br>• Traditional vs. Big Data approach<br>• Real-world challenges | • **5V Modeling:** Modeled in `docs/phase1_report.md` and dataset synthesis.<br>• **Veracity Pipeline:** Implemented in `src/preprocessing/clean_data.py`. Handles schema validation, deduplication, missing value imputation, and anomaly correction.<br>• **Data Contracts:** Defined in `src/analytics/contracts.py`. | **VERIFIED & OPERATIONAL (Phase 1 & 2)** |
| **Module 2: Big Data Frameworks** | • Hadoop Architecture (HDFS, YARN)<br>• Apache Spark ecosystem & PySpark DataFrames<br>• NoSQL architecture patterns (MongoDB) | • **Spark Compatibility:** System verified for OpenJDK 21 LTS in `src/analytics/spark_compat.py`.<br>• **PySpark Execution:** Implemented in `src/analytics/spark_batch.py` using native DataFrames, Window functions, and Spark SQL.<br>• **NoSQL Storage:** Implemented in `src/nosql/mongo_manager.py` with 3 logical collections (`active_orders`, `courier_telemetry` with GeoJSON 2dsphere indexing, `zone_risk_scores`), environment config, and embedded SQLite fallback. | **VERIFIED & OPERATIONAL (Phase 3 & 6)** |
| **Module 3: MapReduce Paradigm** | • Map tasks, Reduce tasks, Grouping by Key, Combiners<br>• Relational Algebra via MapReduce (Selection, Projection, Join, Aggregation)<br>• Matrix operations | • **MapReduce Algorithm Demo:** Implemented in `src/mapreduce/order_volume_mr.py` showing explicit Mapper, Combiner (99.52% traffic reduction), Shuffle & Sort, and Reducer key-value pairs.<br>• Relational Algebra aggregations and selections verified. | **VERIFIED & OPERATIONAL (Phase 3)** |
| **Module 4: Mining Big Data Streams** | • Stream Data Model & DSMS<br>• Stream sampling techniques<br>• Filtering streams: Bloom Filter<br>• Count-Distinct: Flajolet-Martin Algorithm<br>• Sliding-window counting: DGIM Algorithm | • **Order Stream Simulator:** `src/streaming/order_stream_simulator.py` emulating continuous Kafka order events.<br>• **Bloom Filter:** `src/streaming/bloom_filter.py` with 0 false negatives, duplicate suppression, empirical FP benchmark.<br>• **Flajolet-Martin Algorithm:** `src/streaming/flajolet_martin.py` median-of-means distinct courier estimation.<br>• **DGIM Algorithm:** `src/streaming/dgim_window.py` power-of-2 bucket packing, window expiration, relative error <= 50%. | **VERIFIED & OPERATIONAL (Phase 4)** |
| **Module 5: Big Data Mining Algorithms** | • Frequent Pattern Mining: PCY Algorithm, Multistage PCY, SON with MapReduce<br>• Clustering: Canopy Clustering, CURE, K-Means with spatial metrics<br>• Classification: Logistic Regression, Random Forest, Spark MLlib | • **Frequent Itemset Mining (PCY):** Authentic two-pass PCY algorithm in `src/mining/pcy_market_basket.py` using hash buckets, bitmap reduction, candidate filtering, and association rules (support, confidence, lift) on meal combos.<br>• **Spatial Hub Clustering:** K-Means clustering in `src/mining/spatial_clustering.py` using Haversine great-circle distance (km) with inertia and silhouette optimization over $k \in [3, 8]$ to determine courier staging centers.<br>• **Delivery Delay Prediction:** Supervised classification in `src/ml/delay_prediction.py` predicting binary `is_delayed` with strict zero-leakage guarantee (excluding `delivery_duration_min`, `order_status`), comparing Logistic Regression vs. Random Forest with ROC-AUC, confusion matrix, and feature importances.<br>• **Time-Series Demand Forecasting:** Chronological hourly regression in `src/ml/demand_forecasting.py` using Ridge regression with lag features (`lag_1h` to `lag_24h`, rolling means) benchmarked against Naive Persistence Baseline with zero-order filling and recursive multi-horizon projections. | **VERIFIED & OPERATIONAL (Phase 5)** |
| **Module 6: Link Analysis & Recommendation Engines** | • PageRank, HITS Algorithm<br>• Social network graph clustering<br>• Recommendation Systems: Content-Based, Collaborative Filtering | • **Link Analysis:** Implemented in `src/mining/link_analysis_recommender.py` running PageRank ($d = 0.85$, probability sum = 1.0) and Kleinberg's HITS (Hubs & Authorities) on customer-restaurant bipartite ordering graph.<br>• **Dual Recommendation Engines:** Content-based cosine profile matching and User-Item Collaborative Filtering.<br>• **Unified Operations Command Center:** Integrated 6-phase cockpit in `src/dashboard/overview.py`. | **VERIFIED & OPERATIONAL (Phase 6)** |

---

## 3. Course Outcomes (CO) Fulfillment Audit

- **CO1 (Explain Big Data Motivation & Sources):** Fulfilled by modeling the 5Vs, multimodal data, and contrasting RDBMS scaling limits against distributed architectures.
- **CO2 (Demonstrate Hadoop & NoSQL Frameworks):** Fully fulfilled: PySpark DataFrames, OpenJDK 21 LTS, and MongoDB 3-collection persistence with transparent SQLite fallback operational.
- **CO3 (Implement Tasks using MapReduce Paradigm):** Fulfilled: Explicit MapReduce job with intermediate key-value transparency and 99.52% Combiner shuffle traffic reduction.
- **CO4 (Apply Data Mining Algorithms):** Fully fulfilled: PCY frequent itemset mining on meal transactions, spatial K-Means clustering with Haversine distance, supervised delay classification with zero leakage, and chronological hourly demand forecasting.
- **CO5 (Design Stream Mining Algorithms):** Fully fulfilled: Bloom Filter, Flajolet-Martin, and DGIM algorithms implemented and verified with 14 unit tests in Phase 4.
- **CO6 (Design Recommendation Engines):** Fully fulfilled: PageRank and HITS bipartite graph centrality, Content-Based cosine similarity matching, and Collaborative Filtering unvisited recommendation engines verified with 6 unit tests in Phase 6.

---

## 4. Technology Stack Evolution Plan

| Project Phase | Primary Big Data Technology | Scope & Responsibilities | Implementation Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Python 3.14, Pandas, NumPy | Reproducible dataset pipeline, schema validation, veracity cleaning, unit tests. | **Completed & Verified** |
| **Phase 2** | Pandas, Streamlit, Plotly, Pytest | Executive KPIs, temporal demand analysis, delivery performance, zone rankings, data quality, and PySpark architectural contracts. | **Completed & Verified** |
| **Phase 3** | **Apache Spark (PySpark DataFrames), MapReduce** | Local SparkSession (`local[*]`), DataFrame transformations, windowed aggregations, Spark SQL, standalone MapReduce demonstration, and cross-validation tests. | **Completed & Verified** |
| **Phase 4** | **Streaming Mining Algorithms** | Custom Bloom Filter, Flajolet-Martin (FM), DGIM Algorithm, and streaming order event simulation. | **Completed & Verified** |
| **Phase 5** | **Machine Learning & Big Data Mining** | Park-Chen-Yu (PCY) market basket algorithm, spatial K-Means clustering (Haversine km), zero-leakage delay classifier (Random Forest vs Logistic Regression), and chronological lag-feature time-series demand forecasting (Ridge). | **Completed & Verified** |
| **Phase 6** | **Link Analysis, Recommenders & NoSQL Command Center** | Bipartite customer-restaurant graphs, PageRank, HITS, Content-Based & Collaborative Recommenders, MongoDB operational store, Unified Operations Command Center. | **Completed & Verified (Final Milestone)** |
