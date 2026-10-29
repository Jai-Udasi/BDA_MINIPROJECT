# Real-Time Food Delivery Demand Forecasting and Delay Prediction Using Big Data Analytics

**Academic Capstone Project — Final Implementation & Submission (77/77 Automated Tests Passing)**  
- **Student Name:** Jai Udasi (Class: D20A, Roll No: 34)  
- **Course:** Big Data Analytics (Autonomous NEP Batch 1)  
- **Department:** Information Technology, Vivekanand Education Society's Institute of Technology (VESIT)  
- **Current Milestone:** Phase 6 Final Capstone Complete (77/77 Automated Tests Passing in ~33s)

---

## 📌 Project Overview
This project delivers the complete, working capstone implementation for **Experiment 1 (Case Study Selection & Big Data Roadmap)** from the Big Data Analytics syllabus. The system models the high-volume, spatial-temporal operational challenges of modern food delivery platforms (such as Swiggy, Zomato, and DoorDash), combining historical batch analytics with real-time stream processing, demand forecasting, predictive delay modeling, graph link analysis, recommendation engines, and operational NoSQL persistence.

**Comprehensive Syllabus Coverage (Modules 1 to 6):**
- **Module 1 (Big Data 5Vs & Veracity):** 2,500 validated order records, schema enforcement, group median imputation, dynamic SLA ground truth in `src/preprocessing/clean_data.py`.
- **Module 2 (Big Data Frameworks & NoSQL):** Apache Spark (PySpark 4.2.0) DataFrame batch transformations, window functions, and Spark SQL; MongoDB operational document persistence (`src/nosql/mongo_manager.py`) with 3 logical collections and embedded SQLite fallback.
- **Module 3 (MapReduce Paradigm):** Hadoop MapReduce job (`src/mapreduce/order_volume_mr.py`) with Mapper, local Combiner pre-aggregation (**99.52% shuffle traffic reduction**), Shuffle & Sort, and Reducer aggregations.
- **Module 4 (Mining Data Streams):** Sub-linear space stream algorithms (`src/streaming/`): Bloom Filter (0 false negatives), Flajolet-Martin distinct active courier estimator in $O(\log N)$ bits, DGIM sliding-window delayed order tracking with $\le 50\%$ bounded error.
- **Module 5 (Data Mining & Machine Learning):** Park-Chen-Yu (PCY) two-pass frequent itemset mining with hash bucket bitmap; Haversine great-circle K-Means spatial hub clustering; Zero-Data-Leakage Random Forest delay classifier (ROC-AUC 0.891); and chronological hourly demand forecasting (Ridge with lag features, 45.8% error reduction).
- **Module 6 (Link Analysis, Recommenders & Command Center):** PageRank ($d = 0.85$) and Kleinberg's HITS bipartite customer-restaurant graph centrality; Content-Based cosine profile matching and User-Item Collaborative Filtering; Unified Operations Command Center integrated into Streamlit (`app.py` & `overview.py`).
- **77 Automated Tests:** 100% test pass rate with `pytest` across all 12 test modules in ~33 seconds.

---

## 📁 Project Directory Structure

```text
FoodDeliveryBigData/
│
├── app.py                          # Streamlit interactive web dashboard entry point (6 Syllabus Phases)
├── requirements.txt                # Core dependencies: pandas, streamlit, plotly, pyspark, scikit-learn, scipy, pymongo, pytest
├── README.md                       # Setup, verification & execution guide
├── .gitignore                      # Git configuration
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Dynamic project paths, column schemas, thresholds
│
├── data/
│   ├── raw/
│   │   └── raw_orders.csv          # Raw generated data with veracity anomalies (2,537 rows)
│   ├── processed/
│   │   ├── cleaned_orders.csv      # Cleaned and validated dataset (2,500 rows)
│   │   ├── data_quality_report.json# Transformation audit log
│   │   ├── nosql_fallback.db       # Embedded SQLite persistent store for offline NoSQL fallback
│   │   ├── models/
│   │   │   └── delay_classifier.joblib # Fitted preprocessor & Random Forest pipeline (1.7 MB)
│   │   └── spark_output/           # Curated Apache Spark batch outputs & JSON summaries
│   └── synthetic/
│       ├── synthetic_orders.csv    # Source synthetic dataset archive
│       ├── synthetic_market_baskets.json # 1,000 order transactions for PCY mining
│       └── synthetic_delivery_coordinates.csv # 500 coordinates clustered in Mumbai zones
│
├── src/
│   ├── __init__.py
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── generate_orders.py      # Reproducible synthetic data generation engine (Seed: 42)
│   │   └── generate_phase5_data.py # Synthetic market basket & spatial coordinates generator
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── clean_data.py           # Deduplication, datetime parsing, group imputation
│   ├── mapreduce/
│   │   ├── __init__.py
│   │   └── order_volume_mr.py      # Hadoop MapReduce Engine (Mapper, Combiner, Shuffler, Reducer)
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── contracts.py            # Standard analytical data contracts
│   │   ├── spark_compat.py         # PySpark & Java environment readiness diagnostics
│   │   ├── spark_batch.py          # PySpark DataFrame, Window & Spark SQL batch engine
│   │   └── exploratory_analysis.py # Pure-function EDA, KPIs, distributions, filters
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── order_stream_simulator.py # High-throughput streaming event generator & queue transport
│   │   ├── bloom_filter.py         # Bloom Filter bit array with O(k) duplicate suppression
│   │   ├── flajolet_martin.py      # Flajolet-Martin distinct active courier estimator
│   │   └── dgim_window.py          # DGIM sliding-window power-of-2 delay counting
│   ├── mining/
│   │   ├── __init__.py
│   │   ├── pcy_market_basket.py    # Authentic two-pass PCY algorithm with hash bitmap
│   │   ├── spatial_clustering.py   # Spatial K-Means clustering with Haversine distance
│   │   └── link_analysis_recommender.py # PageRank, HITS, Content-Based & Collaborative Recommenders
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── delay_prediction.py     # Supervised delay classifier with zero data leakage
│   │   └── demand_forecasting.py   # Chronological lag-feature time-series demand forecasting
│   ├── nosql/
│   │   ├── __init__.py
│   │   └── mongo_manager.py        # MongoDB Operational Store with embedded SQLite Fallback
│   └── dashboard/
│       ├── __init__.py
│       └── overview.py             # Streamlit visual components across all 6 syllabus modules
│
├── tests/
│   ├── __init__.py
│   ├── test_data_generation.py     # Unit tests for data generator and reproducibility (5 tests)
│   ├── test_preprocessing.py       # Unit tests for cleaning, nulls, anomalies (6 tests)
│   ├── test_analytics.py           # Unit tests for KPIs, formulas, filters, outliers (11 tests)
│   ├── test_mapreduce.py           # Unit tests for Mapper, Combiner, Reducer stages (5 tests)
│   ├── test_spark_batch.py         # Unit tests for Spark DataFrames, aggregations, SQL (5 tests)
│   ├── test_streaming_algorithms.py# Unit tests for Bloom Filter, Flajolet-Martin, DGIM (14 tests)
│   ├── test_pcy_market_basket.py   # Unit tests for PCY two-pass hash bitmap mining (6 tests)
│   ├── test_spatial_clustering.py  # Unit tests for spatial Haversine K-Means clustering (4 tests)
│   ├── test_delay_prediction.py    # Unit tests for zero data leakage delay classifier (4 tests)
│   ├── test_demand_forecasting.py  # Unit tests for chronological time-series forecasting (4 tests)
│   ├── test_mongo_manager.py       # Unit tests for MongoDB upserts, indexes & SQLite fallback (7 tests)
│   └── test_link_analysis_recommender.py # Unit tests for PageRank, HITS & dual recommenders (6 tests)
│
├── presentation/
│   └── bda_project_slides.html     # Interactive 14-slide self-contained HTML presentation deck
│
└── docs/
    ├── BDA_34_EXP1.pdf             # Original Case Study document
    ├── bda syllabus.pdf            # College BDA syllabus document
    ├── requirements_mapping.md     # Full mapping of case study & syllabus modules 1-6
    ├── phase_plan.md               # 6-phase engineering roadmap
    ├── data_dictionary.md          # Complete schema definition, KPIs, and formulas
    ├── phase1_report.md            # Phase 1 engineering report
    ├── phase2_report.md            # Phase 2 engineering report
    ├── phase3_report.md            # Phase 3 Spark & MapReduce report
    ├── phase4_report.md            # Comprehensive Phase 4 Stream Mining report
    ├── phase5_report.md            # Comprehensive Phase 5 Machine Learning & Mining report
    ├── Final_project_report.md     # Final Capstone Project Report
    └── viva_questions_and_answers.md # 45+ comprehensive viva questions & answers across Modules 1-6
```

---

## 🚀 Step-by-Step Windows Setup & Execution Guide

Follow these exact steps in your VS Code terminal or PowerShell window on Windows:

### Step 1: Open Terminal in Project Directory
```powershell
cd e:\FoodDeliveryBigData
```

### Step 2: Verify Big Data Prerequisites (Java JDK & PySpark)
```powershell
python src/analytics/spark_compat.py
```
- **Output:** Verifies that Java OpenJDK 21 LTS and PySpark 4.2.0 are detected and configured.

### Step 3: Run the Hadoop MapReduce Paradigm Demonstration (Module 3)
```powershell
python src/mapreduce/order_volume_mr.py
```
- **Output:** Demonstrates Mapper key-value emissions, Combiner local pre-aggregation (saving 99.52% shuffle traffic), Shuffle & Sort, and final Reducer zone aggregations.

### Step 4: Run the Apache Spark (PySpark) Batch Analytics Engine (Module 2)
```powershell
python src/analytics/spark_batch.py
```
- **Output:** Initializes local master `SparkSession`, loads 2,500 orders with `StructType` schema, executes single-pass distributed aggregations, PySpark Window ranking, and Spark SQL queries.

### Step 5: Run Machine Learning, Mining & Link Analysis Scripts
```powershell
# 1. Frequent Itemset Mining using Park-Chen-Yu (PCY)
python src/mining/pcy_market_basket.py

# 2. Spatial Delivery-Hub Clustering (Haversine K-Means)
python src/mining/spatial_clustering.py

# 3. Supervised Delivery Delay Risk Classifier (Zero Leakage)
python src/ml/delay_prediction.py

# 4. Chronological Hourly Demand Forecasting (Lag Features & Ridge)
python src/ml/demand_forecasting.py
```

### Step 6: Run the Complete Automated Test Suite (77 Tests)
```powershell
python -m pytest -v
```
- **Expected Result:** `77 passed in ~33s` (100% pass rate across all 12 test modules).

### Step 7: Launch the Interactive Streamlit Operations Command Center
```powershell
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### Step 8: View the Presentation Slide Deck
Open [`presentation/bda_project_slides.html`](file:///e:/FoodDeliveryBigData/presentation/bda_project_slides.html) directly in any web browser. Use arrow keys (`←` / `→`) to navigate through all 14 slides.

---

## 🎯 What to Demonstrate to Your College Teacher & Examiner

When demonstrating this project to your professor or viva examiner:

1. **Syllabus Module Relevance:**
   - Open [`docs/requirements_mapping.md`](file:///e:/FoodDeliveryBigData/docs/requirements_mapping.md) and [`docs/Final_project_report.md`](file:///e:/FoodDeliveryBigData/docs/Final_project_report.md) showing verified coverage of **Modules 1, 2, 3, 4, 5, and 6**.
2. **Terminal Demonstrations:**
   - Run `python -m pytest -v`: Show all **77 unit tests passing** with zero failures.
   - Run `python src/mapreduce/order_volume_mr.py`: Show the Combiner reducing network shuffle traffic by 99.52%.
   - Run `python src/analytics/spark_batch.py`: Show PySpark DataFrame windowed aggregations on local multi-core engine.
3. **Streamlit "6. Link Analysis, Mining Social-Network Graphs and Recommendation Engines" Tab:**
   - **Subtab 1 (Unified Command Center):** Show the real-time event stream, multi-horizon demand forecast, zone delay alerts exceeding 75% threshold, and system health status.
   - **Subtab 2 (Bipartite Graph Centrality):** Show PageRank authority restaurants and HITS Hubs & Authorities scores.
   - **Subtab 3 (Dual Recommenders):** Select a customer from the dropdown to compare Content-Based cuisine similarity vs. Collaborative Filtering peer recommendations.
   - **Subtab 4 (MongoDB NoSQL Store):** Show active collections (`active_orders`, `courier_telemetry`, `zone_risk_scores`), click **"Ingest Live In-Flight Order"** to demonstrate live upsert and instant document increment.
4. **Viva & Theory Preparation:**
   - Review [`docs/viva_questions_and_answers.md`](file:///e:/FoodDeliveryBigData/docs/viva_questions_and_answers.md) containing 45+ in-depth viva questions covering every module.
5. **Presentation Slides:**
   - Present [`presentation/bda_project_slides.html`](file:///e:/FoodDeliveryBigData/presentation/bda_project_slides.html) during the capstone viva.

---

## 🎓 Capstone Completion Certification
This project has achieved **100% completion across all 6 phases** specified in the curriculum and walkthrough. All code, tests, documentation, and presentation artifacts are verified, persistent, and runnable on Windows.
