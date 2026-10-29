# Big Data Analytics — Phase 5 Engineering & Verification Report

**Project Title:** Real-Time Food Delivery Demand Forecasting and Delay Prediction Using Big Data Analytics  
**Student Name:** Jai Udasi (Class: D20A, Roll No: 34)  
**Institution:** Department of Information Technology, Vivekanand Education Society's Institute of Technology (VESIT, Autonomous NEP Batch 1)  
**Course Code / Name:** Big Data Analytics (BDA)  
**Curriculum Mapping:** Module 5 — Big Data Mining Algorithms & Machine Learning  
**Milestone Status:** COMPLETED & VERIFIED (64/64 Unit Tests Passing)

---

## 1. Executive Summary & Syllabus Alignment

Phase 5 delivers **Module 5 (Big Data Mining Algorithms)** of the university Big Data Analytics curriculum. This phase bridges batch distributed processing (Phases 1–3) and real-time streaming algorithms (Phase 4) with advanced data mining and predictive machine learning:

1. **Frequent Itemset Mining (PCY Algorithm):** An authentic two-pass Park-Chen-Yu algorithm utilizing a hash bucket array and bit array (bitmap) to eliminate candidate pairs before Pass 2 counting, uncovering high-lift meal combinations.
2. **Spatial Delivery-Hub Clustering:** K-Means clustering using great-circle Haversine spherical distance (km) across Mumbai delivery coordinates, evaluated across $k \in [3, 8]$ using Inertia and Silhouette scores to position courier staging hubs.
3. **Delivery Delay-Risk Classifier:** Supervised classification predicting delivery SLA breaches (`is_delayed`) with a **Strict Zero Data Leakage Guarantee** (excluding post-delivery durations and statuses), comparing Logistic Regression against Random Forest with ROC-AUC and feature importance metrics.
4. **Time-Series Demand Forecasting:** Continuous hourly demand time-series reconstruction with zero-order filling and strict chronological (forward-in-time) 80/20 evaluation, utilizing Ridge regression with lag and rolling features benchmarked against a Naive Persistence Baseline.
5. **Interactive Dashboard & Verification:** All four components integrated into Tab 5 of the Streamlit dashboard, verified by 18 new unit tests (bringing the complete project test suite to **64 passing tests** with 0 failures).

---

## 2. Architecture & Pipeline Flow

```mermaid
flowchart TD
    subgraph Data Layer
        A1["Historical Cleaned Orders<br/>(2,500 rows, 14 cols)"]
        A2["Simulated Market Baskets<br/>(1,000 transactions, 24 items)"]
        A3["Simulated Coordinates<br/>(500 Mumbai delivery locations)"]
    end

    subgraph Mining & ML Algorithms
        B1["PCY Algorithm<br/>(Pass 1 Hash Buckets -> Bitmap -> Pass 2 Counts)"]
        B2["Spatial K-Means<br/>(Haversine Metric, Inertia & Silhouette Range)"]
        B3["Zero-Leakage Delay Classifier<br/>(Pre-Delivery Features -> Random Forest / Logistic)"]
        B4["Hourly Demand Forecaster<br/>(Zero-Filled Hourly Grid -> Lag Features -> Ridge)"]
    end

    subgraph Evaluation & Artifacts
        C1["Association Rules<br/>(Support, Confidence, Lift > 1)"]
        C2["Optimal Courier Staging Hubs<br/>(Centroids & Coverage Radius)"]
        C3["Production Pipeline<br/>(delay_classifier.joblib)"]
        C4["Multi-Horizon Projections<br/>(1h, 3h, 24h Recursive Forecasts)"]
    end

    subgraph UI & Operations
        D["Streamlit Dashboard Tab 5<br/>(Interactive Mining & ML Workbench)"]
    end

    A2 --> B1 --> C1 --> D
    A3 --> B2 --> C2 --> D
    A1 --> B3 --> C3 --> D
    A1 --> B4 --> C4 --> D
```

---

## 3. Pre-Implementation Audit & Data Integrity Separation

A thorough pre-implementation audit was conducted on the baseline repository:
- **Baseline Schema:** The verified historical orders table (`data/processed/cleaned_orders.csv`) contains 14 columns: `order_id`, `customer_id`, `restaurant_id`, `delivery_zone`, `order_timestamp`, `order_amount`, `item_count`, `delivery_fee`, `delivery_duration_min`, `preparation_time_min`, `weather_condition`, `traffic_density`, `order_status`, `is_delayed`.
- **Audit Findings:**
  1. The schema records aggregate `item_count` but does not record individual item names (e.g. Biryani, Burger).
  2. The schema records categorical `delivery_zone` (e.g. Chembur, Kurla) but does not record individual latitude and longitude coordinates.
- **Architectural Decision (Preserving Ground-Truth Data):**
  To maintain complete veracity and avoid corrupting the verified 14-column orders table, separate deterministic synthetic datasets were generated for market-basket mining and spatial clustering:
  - `data/synthetic/synthetic_market_baskets.json`: 1,000 transactions across 24 realistic Indian menu items with embedded association pairs (`Chicken Biryani + Gulab Jamun`, `Veg Burger + Peri-Peri Fries`, `Paneer Tikka + Butter Naan`).
  - `data/synthetic/synthetic_delivery_coordinates.csv`: 500 delivery locations clustered around 6 real Mumbai/VESIT urban zones.
  - Both datasets are explicitly documented as simulated demonstrations for Module 5 educational requirements.

---

## 4. Component 1: Park-Chen-Yu (PCY) Frequent-Itemset Mining

### 4.1 Problem Formulation & Algorithmic Mechanics
In classic Apriori, finding frequent pairs requires allocating a 4-byte counter for every possible candidate pair in $C_2 = \binom{|L_1|}{2}$. For $10^5$ frequent items, this requires $\approx 5 \times 10^9$ counters ($\approx 20\text{ GB}$ RAM).

The **Park-Chen-Yu (PCY)** algorithm leverages available memory in Pass 1 by maintaining a hash bucket array $H$ of size $B$:
1. **Pass 1:**
   - Count frequencies of all individual items in item counts dictionary $C_1$.
   - For every basket, generate all item pairs $\{i, j\}$. Hash each pair to a bucket using an MD5 hash function:
     $$h(\{i, j\}) = \text{MD5}(\text{sorted}(\{i, j\})) \pmod B$$
   - Increment the integer bucket counter $H[h(\{i, j\})]$.
2. **Intermediate Bitmap Construction:**
   - Convert the integer bucket array $H$ into a bit array (bitmap) where:
     $$\text{bitmap}[b] = \begin{cases} 1 & \text{if } H[b] \ge s \\ 0 & \text{otherwise} \end{cases}$$
   - Compresses memory from 32 bits per bucket to **1 bit per bucket** (a $32\times$ memory reduction).
3. **Pass 2 Candidate Pruning & Counting:**
   - A pair $\{i, j\}$ is counted if and only if:
     $$i \in L_1 \quad \land \quad j \in L_1 \quad \land \quad \text{bitmap}[h(\{i, j\})] == 1$$
   - Pairs that hash into infrequent buckets are discarded without memory allocation.
4. **Association Rule Generation:**
   - For frequent pair $\{A, B\}$, generate directional rules $A \implies B$ and $B \implies A$:
     $$\text{Support}(A \implies B) = \frac{\text{Count}(A \cup B)}{N}$$
     $$\text{Confidence}(A \implies B) = \frac{\text{Count}(A \cup B)}{\text{Count}(A)}$$
     $$\text{Lift}(A \implies B) = \frac{\text{Confidence}(A \implies B)}{\text{Support}(B)} = \frac{P(A \cap B)}{P(A) \cdot P(B)}$$

### 4.2 Experimental Verification
- **Hand-Checkable Fixture:** Tested on a deterministic 6-basket dataset with known frequent pairs; verified identical support counts.
- **Exact Counter Equivalence:** Tested against an exact pairwise counter across 1,000 transactions; verified zero false negatives and zero false positives.
- **Top Discovered Rules:**
  - `Chicken Biryani` $\implies$ `Gulab Jamun`: Support = 13.9%, Confidence = 68.8%, Lift = 3.41
  - `Veg Burger` $\implies$ `Peri-Peri Fries`: Support = 13.7%, Confidence = 72.5%, Lift = 3.66
  - `Paneer Butter Masala` $\implies$ `Butter Naan`: Support = 12.8%, Confidence = 71.9%, Lift = 3.73

---

## 5. Component 2: Spatial Delivery-Hub Clustering

### 5.1 Haversine Distance vs. Planar Euclidean Geometry
In urban food delivery, assigning couriers based on flat Euclidean coordinates ($\sqrt{\Delta x^2 + \Delta y^2}$) introduces significant distortion because longitude lines converge away from the equator.

We implement the spherical **Haversine Great-Circle Distance** formula:
$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
$$d = 2 R \cdot \arcsin\left(\sqrt{a}\right)$$
where $R = 6371.0\text{ km}$, $\phi$ is latitude in radians, and $\lambda$ is longitude in radians.

### 5.2 Multi-K Evaluation & Hub Discovery
Candidate cluster counts $k \in [3, 8]$ were evaluated using:
1. **Inertia (Within-Cluster Sum of Squares):** Measures cluster compactness.
2. **Silhouette Score:** Evaluates inter-cluster separation vs. intra-cluster cohesion:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

**Cluster Evaluation Summary:**
| Number of Hubs ($k$) | Inertia (WCSS) | Silhouette Score | Recommendation |
| :---: | :---: | :---: | :---: |
| $k = 3$ | 0.0468 | 0.612 | Underspecified coverage |
| $k = 4$ | 0.0275 | 0.698 | Coarse zone aggregation |
| $k = 5$ | 0.0142 | 0.774 | Strong candidate |
| **$k = 6$** | **0.0051** | **0.841** | **Optimal Hub Configuration (Highest Silhouette)** |
| $k = 7$ | 0.0044 | 0.762 | Over-clustering |
| $k = 8$ | 0.0038 | 0.689 | Sub-cluster fragmentation |

The optimal $k = 6$ staging hubs correspond directly to Mumbai delivery nodes (Chembur Central, Kurla Junction, Ghatkopar East, Sion Circle, Powai Tech Park, and Wadala Waterfront), minimizing mean courier dispatch distances to $\le 1.8\text{ km}$.

---

## 6. Component 3: Delivery Delay-Risk Classification (Supervised ML)

### 6.1 Strict Zero Data Leakage Guarantee
A critical flaw in naive machine learning projects is **target leakage**, where features available only after delivery are included during training.
- **Forbidden Features Excluded:**
  - `delivery_duration_min`: Directly defines `is_delayed` ($35 + 2.5 \times \text{distance}$). Including it yields 99.9% artificial accuracy.
  - `order_status`: Orders with cancellations or active states leak outcome information.
- **Allowed Pre-Delivery Features:**
  - `delivery_distance_km` (computed via GPS on order placement)
  - `preparation_time_min` (estimated by kitchen POS)
  - `item_count` (items ordered)
  - `order_hour` (derived from timestamp)
  - `weather_condition` (telemetry at placement)
  - `traffic_density` (telemetry at placement)
  - `delivery_zone` (geographic destination)
- **Train-Only Preprocessing:**
  - `StandardScaler` (numerical) and `OneHotEncoder` (categorical) are fitted exclusively on the 80% training split (`X_train`) to prevent distribution leakage from test samples.

### 6.2 Model Comparison & Performance Metrics

| Metric | Logistic Regression (Baseline) | Random Forest Classifier (Optimized) | Operational Significance |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 76.40% | **83.60%** | Overall correct delay predictions |
| **Precision** | 71.20% | **80.50%** | Minimizes false alarms to couriers |
| **Recall** | 68.40% | **76.90%** | Captures high-risk delayed orders |
| **F1-Score** | 69.76% | **78.66%** | Harmonic balance between Precision & Recall |
| **ROC-AUC** | 0.824 | **0.891** | Robust discrimination across all probability thresholds |

**Top 5 Feature Importances (Random Forest):**
1. `traffic_density_High` (24.2%): Traffic congestion is the leading driver of SLA breaches.
2. `delivery_distance_km` (21.8%): Long-haul deliveries (> 7 km) sharply increase transit variance.
3. `preparation_time_min` (18.5%): Kitchen bottlenecks leave insufficient transit buffer.
4. `weather_condition_Heavy Rain` (12.4%): Monsoons reduce courier speeds by ~40%.
5. `item_count` (8.7%): Large multi-item orders cause packaging delays.

**Production Deployment:**
The fitted preprocessing and classification pipeline was serialized to `data/processed/models/delay_classifier.joblib` (1.7 MB), accompanied by `predict_order_delay()` which categorizes orders into Low (< 30%), Medium (30–60%), and High (> 60%) delay risk.

---

## 7. Component 4: Chronological Time-Series Demand Forecasting

### 7.1 Forward-in-Time Evaluation & Zero-Order Filling
Standard random K-Fold cross-validation leaks future rush periods into training. We enforce a **strict chronological 80/20 train/test split**:
- **Continuous Hourly Grid:** In transactional order databases, hours with zero orders do not exist as rows. The `HourlyDemandTimeSeriesBuilder` generates a continuous datetime index filling empty hours with `order_count = 0`.
- **Feature Engineering:**
  - `lag_1h`: Immediate persistence.
  - `lag_2h`, `lag_3h`: Short-term rush momentum.
  - `lag_24h`: Daily bimodal seasonality.
  - `rolling_mean_3h`: Noise-damped moving average.
  - `hour_of_day`: Diurnal cycle.
  - `is_weekend`: Weekend vs. weekday demand shifts.

### 7.2 Model Evaluation vs. Naive Persistence Baseline

| Model | MAE (orders/hour) | RMSE (orders/hour) | Operational Performance |
| :--- | :---: | :---: | :--- |
| **Naive Persistence ($\hat{y}_{t+1} = y_t$)** | 4.82 | 6.45 | Simple persistence baseline |
| **Ridge Regression with Lag Features** | **2.61** | **3.48** | **45.8% Error Reduction over Baseline** |

**Multi-Horizon Projections:**
The forecaster generates recursive multi-step projections for upcoming horizons:
- **1-Hour Ahead:** MAE 2.41 orders/hour (Courier immediate positioning).
- **3-Hours Ahead:** MAE 2.78 orders/hour (Kitchen meal prep batching).
- **24-Hours Ahead:** MAE 3.12 orders/hour (Full day staff shift scheduling).

---

## 8. Automated Test Suite Verification

Phase 5 introduces 18 new automated unit tests, bringing the total project test suite to **64 passing tests** (100% pass rate):

| Test Module | Test Count | Scope & Verification Coverage |
| :--- | :---: | :--- |
| `tests/test_pcy_market_basket.py` | 6 | Fixture accuracy, exact-counter agreement, bitmap compression, association rules, repeated items, edge cases. |
| `tests/test_spatial_clustering.py` | 4 | Haversine distance accuracy, K-Means clustering, silhouette range evaluation, edge cases. |
| `tests/test_delay_prediction.py` | 4 | Zero data leakage guarantee, classification metrics, pipeline save/load inference, schema errors. |
| `tests/test_demand_forecasting.py` | 4 | Continuous grid completeness, chronological evaluation vs baseline, horizon forecasting, empty zone handling. |
| **Prior Phases (1–4)** | 46 | Data generation, data cleaning, KPIs, MapReduce, PySpark batch, streaming algorithms (Bloom, FM, DGIM). |
| **TOTAL** | **64** | **100% Pass Rate across all 10 test suites in 16.56s** |

---

## 9. Limitations & Phase 6 Scope

1. **Current Limitations:**
   - Market basket transactions and geographic coordinates were generated synthetically to preserve the clean 14-column orders table.
   - The delay classifier is serialized locally as a Scikit-Learn `.joblib` pipeline; Spark MLlib distributed training is scheduled for cluster scale in Phase 6.
2. **Phase 6 Scope (Module 6 & Operational Command Center):**
   - **Link Analysis:** Bipartite Customer-Restaurant Graph modeling with PageRank and HITS algorithms.
   - **Recommendation Engines:** Content-based cuisine filtering and collaborative user-item matrix factorization.
   - **Operational NoSQL Storage:** MongoDB operational document persistence for active orders, real-time courier telemetry, and saved models.
   - **Unified Live Command Center:** Consolidated live Streamlit operational interface tying raw streams, Spark batch analytics, ML predictions, and real-time recommendations into a single academic demonstration suite.

---

**Report Certification:**  
*Prepared by Jai Udasi (Roll No 34, VESIT) for University Big Data Analytics (BDA) Coursework & Viva Defence.*