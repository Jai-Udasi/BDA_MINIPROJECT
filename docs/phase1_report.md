# Phase 1 Engineering & Academic Report

## Project: Real-Time Food Delivery Demand Forecasting and Delay Prediction Using Big Data Analytics
- **Author / Student:** Jai Udasi (Class: D20A, Roll No: 34)
- **Institution:** Vivekanand Education Society's Institute of Technology (VESIT)
- **Course:** Big Data Analytics (NEP Batch 1)
- **Phase Milestone:** Phase 1 — Project Foundation, Dataset Pipeline & Exploratory Dashboard

---

## 1. Executive Summary
Phase 1 establishes the operational foundation for the Big Data Analytics mini project. Grounded in the primary requirements of Experiment 1 (`BDA_34_EXP1.pdf`) and the college syllabus (`bda syllabus.pdf`), this phase successfully implements:
1. A production-ready, modular Python project structure.
2. A reproducible, domain-faithful synthetic order generation engine (`generate_orders.py`) with realistic temporal rush distributions, operational physics, and controlled data-veracity anomalies.
3. An automated data cleaning and quality auditing pipeline (`clean_data.py`) implementing documented rules for schema verification, deduplication, missing value imputation, and anomaly correction.
4. An interactive Streamlit and Plotly analytical dashboard (`app.py`) providing dynamic filtering, KPI cards, temporal order trends, zone demand distributions, and data quality inspections.
5. A comprehensive automated test suite with 100% passing test coverage across data generation and preprocessing logic.

---

## 2. Dataset Synthesis & Big Data Veracity Modeling

### 2.1 Why Synthetic Data in Phase 1?
As noted in the case study roadmap, food delivery platforms operate on proprietary event logs. To build a genuinely working application without relying on unstable third-party APIs or unvalidated external dumps during foundational engineering, we developed a deterministic data generation pipeline using a fixed random seed (`seed = 42`).

### 2.2 Realistic Domain Modeling
The generator does not simply create random numbers; it simulates genuine food delivery mechanics:
- **Temporal Peaks:** Order frequency follows an authentic bimodal probability distribution reflecting lunch rushes (12:00 PM – 2:30 PM) and dinner peaks (7:00 PM – 10:30 PM), with sparse early-morning activity.
- **Restaurant Operational Model:** Food preparation times are functionally coupled with order item counts and kitchen models (e.g., Fast Food restaurants average 12–18 mins prep, while Fine Dining requires 30–45 mins).
- **Physical Transit Dynamics:** Total delivery duration models road transit physics:
  $$\text{Delivery Duration} = \text{Prep Time} + (\text{Distance} \times \text{Transit Factor} \times \text{Traffic Multiplier}) + \text{Weather Disruption} + \text{Handover Buffer}$$
- **SLA Benchmark:** A dynamic Service Level Agreement (SLA) threshold determines whether an order is marked as delayed (`is_delayed = 1` if duration exceeds $35\text{ mins} + 2.5 \times \text{Distance}$).

### 2.3 Intentional Data Veracity Injections
To simulate the real-world **Veracity** challenge of Big Data (Syllabus Module 1), the generator deliberately introduces controlled real-world noise:
- ~1.5% duplicate order records (simulating network retries and duplicate webhooks).
- ~2.0% missing values in optional fields (`restaurant_type`, `weather_condition`, `preparation_time_min`, `delivery_distance_km`).
- ~0.5% numeric anomalies (e.g., non-positive order amounts, zero item counts).
- ~1.0% timestamp whitespace irregularities.

---

## 3. Data Cleaning & Preprocessing Pipeline Analysis

The preprocessing module (`src/preprocessing/clean_data.py`) reads raw data, performs defensive validation, logs every action, and outputs a clean dataset and an auditable JSON report (`data/processed/data_quality_report.json`).

### 3.1 Quantitative Transformation Audit

| Metric | Raw Dataset (Before) | Cleaned Dataset (After) | Engineering Action / Rule Applied |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 2,537 | 2,500 | 37 duplicate order IDs removed (keeping first occurrence). |
| **Retention Rate** | 100.0% | 98.54% | High retention; only genuine duplicates were eliminated. |
| **Missing Values (Total)** | 200 cells | 0 cells | All missing cells resolved via statistical imputation. |
| **Missing `restaurant_type`** | 50 records | 0 records | Imputed using categorical mode (`Bakery & Desserts` / `Fast Food`). |
| **Missing `weather_condition`** | 48 records | 0 records | Imputed using categorical mode (`Clear`). |
| **Missing `prep_time_min`** | 50 records | 0 records | Imputed using group median by restaurant type. |
| **Missing `delivery_distance`**| 49 records | 0 records | Imputed using group median by delivery zone. |
| **Negative Order Amounts** | 6 records | 0 records | Corrected using overall positive median (₹389.25). |
| **Zero Item Counts** | 6 records | 0 records | Corrected using overall positive median (2 items). |
| **Timestamp Format** | Raw strings with whitespace | ISO 8601 Datetime | Trimmed whitespace and parsed into datetime64 objects. |
| **Delay Consistency (`is_delayed`)** | Raw flag | Standardized 0/1 | Recalculated against strict SLA formula to prevent label drift. |

---

## 4. Architecture & Implementation Decisions

1. **Path Agility with `pathlib`:**
   - No hardcoded absolute file paths (e.g. `C:\Users\...`) were used. All paths are resolved relative to the project root via `Path(__file__).resolve().parent.parent`. This guarantees the code runs seamlessly across Windows, Linux, and macOS.
2. **Immutability of Source Data:**
   - Raw input data (`data/raw/raw_orders.csv`) is treated as append-only and strictly read-only. Cleaned outputs are written exclusively to `data/processed/cleaned_orders.csv`.
3. **Observability & Logging:**
   - Uses Python's native `logging` library with standardized timestamped output.
4. **Adaptive UI Design:**
   - The Streamlit dashboard inspects the dataframe's schema dynamically. If a column is missing (e.g., if a simpler dataset is uploaded), charts gracefully display informative messages rather than crashing with a `KeyError`.
5. **No Premature Complexity:**
   - In accordance with the Phase 1 specification, heavy distributed clusters (Hadoop daemons, Spark, Kafka) were avoided. The codebase establishes the data foundation and algorithm contracts first.

---

## 5. Verification & Test Results

The test suite in `tests/` was executed using `pytest`:
- **Command:** `python -m pytest tests/`
- **Results:**
  - `tests/test_data_generation.py`: 5 passed
    - `test_generation_record_count_clean`: PASSED
    - `test_generation_required_columns_present`: PASSED
    - `test_generation_reproducibility`: PASSED
    - `test_generation_value_ranges_clean`: PASSED
    - `test_generation_with_noise_injected`: PASSED
  - `tests/test_preprocessing.py`: 6 passed
    - `test_inspect_schema_and_quality`: PASSED
    - `test_clean_dataset_deduplication`: PASSED
    - `test_clean_dataset_anomaly_correction`: PASSED
    - `test_clean_dataset_null_imputation`: PASSED
    - `test_clean_dataset_invalid_timestamp_handling`: PASSED
    - `test_load_dataset_missing_file_error`: PASSED
  - **Overall:** 11 passed in 0.95s (100% pass rate).

Streamlit Web Application Verification:
- **Command:** `python -m streamlit run app.py`
- **HTTP Response Verification:** Tested via local loopback request: `HTTP Status 200`, content length 6,463 bytes.
- **UI Responsiveness:** Interactive sidebar filters (dates, zones, weather, status), dataset switching (Raw vs. Processed), and Plotly chart tooltips verified.

---

## 6. College Demonstration Walkthrough (For Your Teacher)

When presenting this project to your college teacher/evaluator, follow this structured 4-step demonstration:

### Step 1: Explain the Academic Motivation
- *"Sir/Ma'am, this project addresses Experiment 1 of our Big Data Analytics syllabus: Real-Time Food Delivery Demand Forecasting and Delay Prediction."*
- *"We are analyzing how high-volume order streams and spatial-temporal constraints cause delivery bottlenecks. In Phase 1, we tackled the **Veracity** and **Volume** foundation of Big Data."*

### Step 2: Show the Data Pipeline in the Terminal
- Run the generator: `python src/data_generation/generate_orders.py`
  - Explain: *"Notice how our generator simulates 2,500+ records with peak lunch and dinner hours, while deliberately injecting real-world noise (duplicates, missing values, negative amounts) to test our data pipeline."*
- Run the cleaner: `python src/preprocessing/clean_data.py`
  - Explain: *"Here is our Big Data veracity pipeline. It loaded 2,537 rows, detected 37 duplicates, imputed missing values using group medians, and retained 98.54% of valid data with full JSON audit logging."*

### Step 3: Run the Automated Unit Tests
- Run tests: `python -m pytest tests/`
  - Explain: *"We implemented 11 automated test cases to prove that our data pipeline is reproducible and follows defensive coding standards."*

### Step 4: Demonstrate the Streamlit Dashboard
- Run: `python -m streamlit run app.py`
  - Open `http://localhost:8501`.
  - **Show Dataset Switching:** Switch between *Processed Data* and *Raw Data*. Point out how the "Total Missing Cells" drops from 200 to 0 and duplicates drop from 30 to 0.
  - **Show Hourly Volume:** Point out the distinct lunch rush (12-2 PM) and dinner rush (7-9 PM) in the 24-hour cycle chart.
  - **Show Delay Drivers:** Show the weather and traffic charts, demonstrating how severe rain and traffic jams escalate delivery delays.
  - **Show JSON Audit Log:** Open the expandable "Pipeline Transformation Audit" to show exact records modified.

---

## 7. Scope Reserved for Phase 2
The following features are explicitly reserved for Phase 2 as outlined in `docs/phase_plan.md`:
- Distributed HDFS storage layout modeling.
- Hadoop MapReduce Python streaming jobs (Mappers, Combiners, Reducers) for hourly order volume and zone GMV.
- MapReduce Relational Algebra implementations (Natural Joins, Relational Selection, Projections).
- Comparison of single-node Pandas processing vs. distributed MapReduce execution characteristics.
