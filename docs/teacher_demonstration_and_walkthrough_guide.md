# College Teacher Demonstration & Live Walkthrough Guide

**Project Title:** Real-Time Food Delivery Demand Forecasting and Delay Prediction Using Big Data Analytics  
**Student Name:** Jai Udasi (Class: D20A, Roll No: 34)  
**Institution:** Department of Information Technology, Vivekanand Education Society's Institute of Technology (VESIT, Autonomous NEP Batch 1)  
**Course Code / Name:** Big Data Analytics (BDA)  
**Case Study Reference:** `BDA_34_EXP1.pdf` (Experiment 1: Case Study Selection & Big Data Roadmap)  
**Test Suite Verification:** **77 / 77 Automated Unit Tests Passing (100% Pass Rate)**

---

## 🎯 Purpose of This Document
This document is your **complete, step-by-step presentation script**. It tells you exactly:
1. What commands to run in the terminal before your teacher arrives.
2. In what sequence to demonstrate each feature on the Streamlit dashboard.
3. Exactly what words and technical phrases to say to impress the examiner.
4. How to answer the top 10 toughest questions the teacher might throw at you.

---

## ⚡ Quick 1-Minute Pre-Demo Terminal Setup

Open a PowerShell terminal in `e:\FoodDeliveryBigData` and prepare your two terminal windows:

### Terminal Window 1: Automated Verification & Background Tests
```powershell
cd e:\FoodDeliveryBigData
python -m pytest tests/ -q
```
*Expected Output:* `77 passed, 13 warnings in ~15s` (Keep this terminal visible so the teacher can see 100% test coverage).

### Terminal Window 2: Launch Streamlit Dashboard
```powershell
cd e:\FoodDeliveryBigData
python -m streamlit run app.py
```
Open your browser at **`http://localhost:8501`**. Maximize the browser window (press `F11` if full screen is preferred).

---

## 🎬 10-Minute Step-by-Step Teacher Walkthrough Script

---

### Step 1: The Elevator Pitch (1.5 Minutes)
**What to say to your teacher:**
> *"Good morning, Professor. My project is 'Real-Time Food Delivery Demand Forecasting and Delay Prediction Using Big Data Analytics' for BDA Experiment 1.
>
> In modern food delivery aggregators like Swiggy and Zomato, order demand surges non-linearly during lunch and dinner rushes, while road congestion, rain, and kitchen bottlenecks cause delivery delays. Traditional RDBMS databases fail because transactional table locks collapse under concurrent analytical queries and high-velocity event streams.
>
> To solve this, I built an end-to-end Big Data system addressing all 6 modules of our college BDA syllabus:
> 1. **Module 1:** 5V modeling and automated veracity data cleaning across 2,500 records.
> 2. **Module 2:** Distributed Apache Spark (PySpark) batch engine with window analytics and Spark SQL.
> 3. **Module 3:** Hadoop MapReduce demonstration with a Combiner that slashes shuffle traffic by 99.52%.
> 4. **Module 4:** Sub-linear streaming algorithms: Bloom Filter (0 false negatives), Flajolet-Martin ($O(\log N)$ distinct couriers), and DGIM sliding-window delay counting.
> 5. **Module 5:** Park-Chen-Yu (PCY) frequent itemset mining, Haversine K-Means spatial clustering, a zero-leakage Random Forest delay classifier, and chronological demand forecasting.
> 6. **Module 6 & NoSQL:** Bipartite graph link analysis (PageRank & HITS), dual recommendation engines (Content-based & Collaborative), a 3-collection MongoDB operational store with embedded SQLite fallback, and a real-time Operations Command Center.
>
> The entire project is backed by **77 automated unit tests** passing with 100% reliability."*

---

### Step 2: Show the Terminal Proofs (2 Minutes)
Switch to **Terminal 1** to show raw engine execution before the UI:

#### 1. Show the Hadoop MapReduce Demonstration (Module 3)
```powershell
python src/mapreduce/order_volume_mr.py
```
* **What to point out on screen:**
  - Show the **Mapper** emitting `(zone, (order_amount, 1, is_delayed))`.
  - **Crucial Metric:** Point to the **Combiner saving 99.52% network shuffle traffic** (reducing 2,500 raw order tuples to only 12 partition tuples across worker chunks).
  - Show the **Reducer** computing total GMV, AOV, and delay percentage per zone.

#### 2. Show the Apache Spark (PySpark) Batch Engine (Module 2)
```powershell
python src/analytics/spark_batch.py
```
* **What to point out on screen:**
  - Emphasize that it initializes a real local `SparkSession` using OpenJDK 21 LTS (`master("local[*]")`).
  - Show the PySpark DataFrame transformations, windowed rolling velocity ranking, and Spark SQL table queries executing on Spark's Catalyst engine.

---

### Step 3: Interactive Dashboard Walkthrough (6 Minutes)
Switch to the browser running **`http://localhost:8501`**.

In the sidebar under **"🧭 Syllabus Phase Navigation"**, guide the teacher through the 6 modules:

---

#### 📍 Phase 1: Introduction to Big Data (Module 1)
* Select **"1. Introduction to Big Data"** in the sidebar.
* **What to show:**
  - Show the **5Vs Breakdown Cards** (Volume, Velocity, Variety, Veracity, Value).
  - Expand **"Data Veracity & Quality Audit"**: Show how the automated cleaning pipeline in `src/preprocessing/clean_data.py` resolved raw anomalies (null values, duplicates, negative amounts) into 2,500 pristine, schema-validated records.
  - Show the **IQR Outlier Box Plot** detecting delivery duration anomalies via Tukey's method ($1.5 \times \text{IQR}$).

---

#### 📍 Phase 2: Big Data Frameworks (Module 2)
* Select **"2. Big Data Frameworks"** in the sidebar.
* **What to show:**
  - Show the **Executive KPI cards** (Total Orders: 2,500, GMV: ₹1.46M, AOV: ₹583.56, Observed Delay Rate: 26.6%).
  - Show the **Diurnal 24-Hour Bimodal Demand Curve** displaying the sharp lunch peak (12:00–14:00) and dinner rush (19:00–22:00).
  - Show the **Environment Readiness Diagnostic Card** verifying OpenJDK 21 LTS and PySpark 4.2.0.

---

#### 📍 Phase 3: MapReduce Paradigm (Module 3)
* Select **"3. MapReduce Paradigm"** in the sidebar.
* **What to show:**
  - Show the interactive visual architecture stepper: `Input Splits → Mapper → Combiner → Shuffle & Sort → Reducer`.
  - Point to the **Combiner Network Efficiency Card**: Explain that without Combiners, massive multi-terabyte cluster shuffles cause network congestion. Our Combiner eliminated 99.52% of shuffle data before the network phase.
  - Show the cross-engine mathematical consistency table verifying that Spark, MapReduce, and Pandas produce identical results to 4 decimal places.

---

#### 📍 Phase 4: Mining Big Data Streams (Module 4)
* Select **"4. Mining Big Data Streams"** in the sidebar.
* **What to do:**
  - Click the **"▶️ Run Stream Simulation"** button.
* **What to explain:**
  1. **Bloom Filter:** Point to the duplicate detection card. Explain: *"When mobile users retry on slow networks, the Bloom Filter checks order IDs in $O(k)$ time using $k=7$ hash functions. It has a **strict mathematical guarantee of 0 false negatives**."*
  2. **Flajolet-Martin (FM):** Point to the distinct courier estimator. Explain: *"It uses trailing zero bit analysis $\rho(y)$ to estimate distinct couriers in $O(\log N)$ space rather than storing every ID in memory, with median-of-means variance reduction."*
  3. **DGIM Sliding Window:** Point to the sliding window table. Explain: *"It packs recent delay occurrences into power-of-2 buckets ($1, 2, 4, 8, \dots$) in $O(\log^2 N)$ memory, with relative estimation error strictly bounded $\le 50\%$."*

---

#### 📍 Phase 5: Big Data Mining Algorithms & ML (Module 5)
* Select **"5. Big Data Mining Algorithms"** in the sidebar. Guide the teacher through its 4 subtabs:
  1. **Subtab 1 (PCY Market Basket):** Show the Park-Chen-Yu two-pass algorithm. Explain: *"Unlike Apriori which crashes memory by creating $O(N^2)$ candidate pairs, PCY hashes pairs into a bucket array in Pass 1 and compresses it into a 1-bit bitmap ($32\times$ memory reduction), pruning infrequent pairs before Pass 2."* Point to the top association rule: `Chicken Biryani → Gulab Jamun` (Lift: 3.41).
  2. **Subtab 2 (Spatial Clustering):** Show the K-Means courier staging hubs. Explain: *"Euclidean flat distance causes distortion on Earth's curvature. We implemented the spherical **Haversine Great-Circle formula**. Our multi-$k$ search proved that $k=6$ staging hubs gives the highest Silhouette score (0.841) across Mumbai clusters."*
  3. **Subtab 3 (Delay Risk Predictor):** Drag the traffic slider to `High` and weather to `Heavy Rain`. Show the live probability update and badge changing to `High Delay Risk`. Explain: *"This Random Forest classifier (ROC-AUC 0.891) has a **strict Zero Data Leakage guarantee**: post-delivery features (`delivery_duration_min` and `order_status`) are strictly excluded, and scalers were fit only on training data."*
  4. **Subtab 4 (Demand Forecasting):** Show the chronological demand curve. Explain: *"We enforced a strict forward-in-time 80/20 train/test split to avoid lookahead bias and filled zero-order hours with 0. Our Ridge model with lag features achieved an MAE of 2.61 orders/hr, beating the persistence baseline by 45.8%."*

---

#### 📍 Phase 6: Link Analysis, Recommenders & Command Center (Module 6 & NoSQL)
* Select **"6. Link Analysis, Mining Social-Network Graphs and Recommendation Engines"** in the sidebar.
* **Subtab 1: Unified Operations Command Center:**
  - Show the live cockpit synthesizing all phases: wire-speed event rate, multi-horizon demand forecasts (1h, 3h, 24h), proactive SLA breach alert (highlighting zones exceeding the 75% threshold), and the 6-tier system health diagnostic.
* **Subtab 2: Bipartite Graph Centrality (Link Analysis):**
  - Show **PageRank** ($d=0.85$, probability strictly summing to 1.0) identifying anchor restaurants (`REST-128`).
  - Show **Kleinberg's HITS** computing mutual reinforcement between authority restaurants and hub customers.
* **Subtab 3: Dual Recommendation Engines:**
  - Select a customer from the dropdown (`CUST-100`, `CUST-101`, etc.).
  - Show side-by-side:
    - **Content-Based:** Cosine similarity on restaurant cuisine, prep time, and price profile vectors.
    - **Collaborative Filtering:** Customer-restaurant interaction matrix affinity matching peer ordering habits.
* **Subtab 4: MongoDB NoSQL Operational Store:**
  - Show the health diagnostic card: displays the 3 collections (`active_orders`, `courier_telemetry`, `zone_risk_scores`) and shows active status (or local SQLite fallback mode).
  - Click the **"📥 Ingest Sample Live In-Flight Order & Courier GPS"** button: Watch the page instantly update and increment document counts live!

---

### Step 4: The Presentation Slides (1 Minute, Optional)
If the teacher asks for your presentation slides:
* Open [`presentation/bda_project_slides.html`](file:///e:/FoodDeliveryBigData/presentation/bda_project_slides.html) in your browser.
* Use `ArrowRight` to show **Slide 11 (Command Center)** and **Slide 12 (77 Passing Tests Table)**.

---

## 🧠 Top 10 Teacher Viva Questions & Winning Answers

Have these 10 concise, punchy answers ready:

### 1. "Why did you use PCY instead of standard Apriori?"
> *"Standard Apriori suffers from a combinatorial candidate pair explosion in Pass 2 ($C_2 = \binom{|L_1|}{2}$), requiring gigabytes of RAM for pair counters. PCY utilizes unused RAM in Pass 1 to hash pairs into buckets and compresses them into a 1-bit bitmap ($32\times$ memory reduction), pruning candidate pairs before Pass 2 counting without ever creating unnecessary counters."*

### 2. "Can your Bloom Filter produce false negatives?"
> *"No, mathematically never. When an item is added, all $k$ bit positions are set to 1. Since bits in a standard Bloom filter are never cleared or decremented, querying an inserted item will always find all $k$ bits set to 1. False positives can occur due to random collisions, but false negatives are strictly impossible."*

### 3. "Why is Euclidean distance wrong for delivery clustering?"
> *"Euclidean distance assumes a flat 2D plane ($\sqrt{\Delta x^2 + \Delta y^2}$). Because Earth is an oblate spheroid, 1 degree of latitude is roughly constant (~111 km), but longitude converges towards the poles ($111 \cdot \cos(\text{lat})$). We implemented the spherical **Haversine Great-Circle formula** to calculate exact physical surface distances in kilometers across Mumbai clusters."*

### 4. "How did you ensure Zero Data Leakage in delay prediction?"
> *"We strictly excluded post-delivery attributes: `delivery_duration_min` (which directly defines SLA delay via $35 + 2.5 \times \text{distance}$) and `order_status`. Including them produces artificial 99.9% accuracy. Furthermore, all feature scalers (`StandardScaler`) and encoders were fitted strictly on the 80% training split, preventing test sample distribution parameters from leaking into training."*

### 5. "Why not use random train/test split or K-Fold for demand forecasting?"
> *"Random splitting assumes independent and identically distributed (i.i.d.) observations. Time series data has strong temporal autocorrelation. Shuffling data leaks future peak rush hours into the training set to predict past hours (lookahead bias). We enforced a strict chronological 80/20 forward-in-time split and reconstructed an unbroken hourly grid with zero-order filling."*

### 6. "What does Lift > 1 indicate in PCY association rules?"
> *"Lift measures how much more frequently two items co-occur compared to random chance: $\text{Lift} = \frac{P(A \cap B)}{P(A) \cdot P(B)}$. A Lift > 1 indicates positive complementarity (e.g. Biryani + Gulab Jamun at Lift 3.41), meaning buying item A significantly increases the probability of buying item B."*

### 7. "What is the exact role of the Combiner in MapReduce?"
> *"The Combiner is a local mini-reducer running directly on mapper worker partitions before data is transmitted across the network. In our project, it pre-aggregated 2,500 intermediate order tuples into 12 partition tuples, slashing cluster network shuffle bandwidth by **99.52%**."*

### 8. "Why is the DGIM sliding-window error bounded by 50%?"
> *"DGIM represents counts using power-of-2 buckets ($1, 2, 4, 8, \dots$). When answering a query for the last $k$ events, the oldest bucket may cross the window boundary. We add half its size. The maximum possible error is half that bucket's size ($2^{j-1}$), while the true count is at least the sum of earlier buckets ($2^j$). Thus, relative error is strictly $\le \frac{2^{j-1}}{2^j} = 50\%$."*

### 9. "What happens if MongoDB is not running on the demonstration PC?"
> *"The application has a zero-failure persistence architecture. `MongoManager` attempts a fast connection check with a 1500ms timeout. If MongoDB is absent, it seamlessly activates an embedded SQLite persistent store (`nosql_fallback.db`) with identical public method signatures. All inserts, upserts, and queries work 100% offline."*

### 10. "How does PageRank differ from Kleinberg's HITS on your bipartite graph?"
> *"PageRank computes a single global structural centrality score using a random walk with teleportation ($d=0.85$) that sums to 1.0 across all nodes. HITS computes two distinct dual scores: **Authority scores** (popular restaurants attracting high order volume) and **Hub scores** (customers ordering from diverse, high-rating restaurants) via mutually reinforcing matrix power iterations."*

---

## 🏆 Summary Checklist for Submission

| Item | Location | Status |
| :--- | :--- | :---: |
| Full 77-Test Pytest Suite | `python -m pytest tests/ -q` | **100% Passing** |
| MapReduce 99.52% Combiner Demo | `python src/mapreduce/order_volume_mr.py` | **Verified** |
| Spark Batch Analytics Engine | `python src/analytics/spark_batch.py` | **Verified** |
| Streamlit Operations Command Center | `python -m streamlit run app.py` | **Operational** |
| HTML Presentation Slide Deck | [`presentation/bda_project_slides.html`](file:///e:/FoodDeliveryBigData/presentation/bda_project_slides.html) | **14 Slides Ready** |
| Final Project Academic Report | [`docs/Final_project_report.md`](file:///e:/FoodDeliveryBigData/docs/Final_project_report.md) | **11 Sections** |
| 45+ Viva Questions & Answers | [`docs/viva_questions_and_answers.md`](file:///e:/FoodDeliveryBigData/docs/viva_questions_and_answers.md) | **Complete** |
| Syllabus Requirements Matrix | [`docs/requirements_mapping.md`](file:///e:/FoodDeliveryBigData/docs/requirements_mapping.md) | **Modules 1-6 Verified** |

*All code, documentation, and presentation materials are locked, verified, and ready for your college viva defence.*
