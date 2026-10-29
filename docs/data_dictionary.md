# Data Dictionary & Metrics Specification

## Project: Real-Time Food Delivery Demand Forecasting and Delay Prediction
**Document Version:** 2.0 (Phase 2 Exploratory Data Analytics Update)  
**Dataset Type:** Synthetic Baseline Dataset (Reproducible via Seed 42)

---

## 1. Raw & Processed Dataset Fields

| Field Name | Storage Type | Origin Classification | Description / Meaning | Constraints & Valid Range | Nullable / Handling Rule | Role in Big Data Analytics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `order_id` | String | Synthetic Identifier | Unique tracking identifier for each transaction. | Format: `ORD-XXXXX` (e.g., `ORD-10001`). Non-empty and unique. | Not nullable. Duplicates dropped keeping first. | Primary Key; used for tracking transactions and distinct counting via Flajolet-Martin. |
| `order_timestamp` | String / Datetime | Synthetic (Simulated) | The exact date and time when the customer placed the order. | Format: `YYYY-MM-DD HH:MM:SS`. Span: Oct 01, 2026 to Oct 11, 2026. | Not nullable. Whitespace trimmed, parsed to ISO datetime. | Temporal aggregation: 24-hour cycle demand, peak periods, time-series forecasting. |
| `restaurant_id` | String | Synthetic Identifier | Unique identifier for the food preparation establishment. | Format: `REST-XXX` (e.g., `REST-100` to `REST-129`). | Not nullable. Categorized or imputed with `REST-UNKNOWN`. | Grouping key for restaurant workload analytics and MapReduce aggregation. |
| `restaurant_type` | String (Categorical) | Synthetic Attribute | Cuisine tier and operational model of the food establishment. | Allowed: `Fast Food`, `Casual Dining`, `Cafe`, `Cloud Kitchen`, `Bakery & Desserts`, `Fine Dining`. | Nullable in raw. Imputed with statistical mode. | Feature for kitchen preparation bottleneck analysis and delay modeling. |
| `delivery_zone` | String (Categorical) | Synthetic Attribute | Urban sector where delivery is fulfilled. | Allowed: `Downtown`, `North Suburbs`, `South Suburbs`, `East Tech Park`, `West University Campus`, `Central Waterfront`. | Nullable in raw. Imputed with mode (`Downtown`). | Primary spatial dimension for demand forecasting and fleet staging. |
| `order_amount` | Float64 | Synthetic Metric | Total gross transactional value of the order in INR (₹). | Numeric: ₹50.0 to ₹10,000.0. Values ≤ 0 are deemed invalid anomalies. | Not nullable. Non-positive anomalies imputed with median amount (₹389.25). | Financial analytics, Gross Merchandise Value (GMV), customer spending tiers. |
| `item_count` | Int64 | Synthetic Metric | Total count of food items in the order. | Integer: 1 to 50 items. Values ≤ 0 are invalid anomalies. | Not nullable. Invalid values replaced with median (2 items). | Feature directly impacting kitchen prep time and packing duration. |
| `preparation_time_min` | Float64 | Synthetic Metric | Time from order placement until food is ready for courier pickup. | Continuous: 5.0 to 120.0 minutes. | Nullable in raw. Imputed with restaurant-type median prep time. | Operational delay bottleneck detection; primary component of total cycle time. |
| `delivery_distance_km` | Float64 | Synthetic Metric | Straight-line road travel distance from restaurant to customer drop-off. | Continuous: 0.1 to 40.0 kilometers. | Nullable in raw. Imputed with zone median distance. | Transit time predictor; fleet vehicle range and dispatch optimization. |
| `delivery_duration_min`| Float64 | Synthetic Metric | Total elapsed time from customer order placement to successful doorstep delivery. | Continuous: 5.0 to 180.0 minutes. | Nullable in raw. Imputed using `prep_time + (distance * 2.8) + 5.0`. | Ground truth target variable for delivery delay regression analysis. |
| `order_status` | String (Categorical) | Synthetic Attribute | Final terminal lifecycle state of the order transaction. | Allowed: `Delivered`, `Cancelled`, `In Transit`. | Nullable in raw. Imputed with `Delivered` if duration exists. | Filtering target for successful fulfillment analytics and cancellation tracking. |
| `weather_condition` | String (Categorical) | Synthetic Attribute | Environmental weather state during transit. | Allowed: `Clear`, `Rainy`, `Windy`, `Foggy`, `Heavy Rain`. | Nullable in raw. Imputed with `Clear`. | Environmental disruption feature impacting travel speed and transit delays. |
| `traffic_density` | String (Categorical) | Synthetic Attribute | Road congestion density index measured across transit corridor. | Allowed: `Low`, `Medium`, `High`, `Jam`. | Nullable in raw. Imputed with `Medium`. | Major external delay factor used in machine learning delay classification. |
| `is_delayed` | Int64 (Binary) | Derived Metric | Indicator flag indicating whether delivery exceeded the expected SLA threshold. | `1` (Delayed: duration > SLA benchmark), `0` (On-Time: duration ≤ SLA). | Derived deterministically from `delivery_duration_min` and distance SLA. | Primary binary classification target label for Phase 5 Machine Learning models. |

---

## 2. Derived Dashboard Metrics & KPI Mathematical Formulations

Every operational metric rendered in the Phase 2 executive overview dashboard follows explicit mathematical formulas and documented denominators:

### 2.1 Total Unique Orders ($N_{\text{orders}}$)
- **Definition:** Distinct count of unique non-null order IDs present in the active dataset view.
- **Formula:** $N_{\text{orders}} = |\text{Unique}(\text{order\_id})|$.

### 2.2 Completion Rate ($\%$)
- **Numerator:** Orders where $\text{order\_status} = \text{'Delivered'}$ (case-insensitive).
- **Denominator:** Total Unique Orders ($N_{\text{orders}}$).
- **Formula:** $\text{Completion Rate} = \frac{\text{Completed Orders}}{N_{\text{orders}}} \times 100$.

### 2.3 Cancellation Rate ($\%$)
- **Numerator:** Orders where $\text{order\_status} = \text{'Cancelled'}$ (case-insensitive).
- **Denominator:** Total Unique Orders ($N_{\text{orders}}$).
- **Formula:** $\text{Cancellation Rate} = \frac{\text{Cancelled Orders}}{N_{\text{orders}}} \times 100$.

### 2.4 Total Revenue (Gross Merchandise Value - GMV)
- **Definition:** Total financial value generated by successful and in-transit orders (excludes cancelled orders).
- **Formula:** $\text{GMV} = \sum_{i \in \text{Non-Cancelled}} \text{order\_amount}_i$.

### 2.5 Average & Median Delivery Duration
- **Strict Rule:** Computed **exclusively on completed orders** with valid non-null delivery durations. Never counts missing values or cancelled orders as zero.
- **Formula (Average):** $\bar{D} = \frac{\sum_{j \in \text{Delivered}} \text{delivery\_duration\_min}_j}{N_{\text{Delivered with Duration}}}$.
- **Formula (Median):** $D_{\text{med}} = \text{Median}(\{\text{delivery\_duration\_min}_j \mid j \in \text{Delivered}\})$.

### 2.6 Delayed-Order Percentage ($\%$)
- **Numerator:** Count of eligible completed orders where $\text{is\_delayed} = 1$ (i.e., delivery duration exceeded the SLA benchmark).
- **Denominator:** Total count of completed orders that possess a valid delivery duration and distance.
- **Formula:** $\text{Delay Rate} = \frac{\sum \text{is\_delayed}}{\text{Eligible Delivered Orders}} \times 100$.

### 2.7 Service Level Agreement (SLA) Threshold
- **Benchmark:** Represents the maximum acceptable delivery duration before an order is deemed delayed.
- **Formula:** $\text{SLA Benchmark} = 35.0\text{ min} + (2.5\text{ min/km} \times \text{delivery\_distance\_km})$.
- **Note:** Clearly labeled as a threshold-based operational estimate in compliance with case study standards.

### 2.8 Peak Demand Period Flag (`is_peak`)
- **Condition:** An hour is classified as `is_peak = True` if it falls into lunch rush (12:00–14:00), dinner rush (19:00–21:00), or has an order count in the top quartile (75th percentile) of 24-hour volume.
