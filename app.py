"""
Streamlit Web Application Entry Point.
Real-Time Food Delivery Demand Forecasting and Delay Prediction Using Big Data Analytics
Phase 2: Exploratory Data Analytics and Interactive Operations Dashboard
"""
import sys
import json
import logging
from pathlib import Path
import streamlit as st
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import (
    RAW_ORDERS_FILE,
    PROCESSED_ORDERS_FILE,
    SYNTHETIC_ORDERS_FILE,
    DATA_QUALITY_REPORT_FILE
)
from src.data_generation.generate_orders import generate_synthetic_orders, save_dataset
from src.preprocessing.clean_data import run_pipeline
from src.analytics.exploratory_analysis import (
    compute_executive_kpis,
    apply_dashboard_filters
)
from src.dashboard.overview import (
    render_header,
    render_executive_kpi_cards,
    render_demand_tab,
    render_performance_tab,
    render_comparisons_tab,
    render_data_quality_tab,
    render_spark_architecture_tab,
    render_pipeline_connectivity_banner,
    render_phase1_view,
    render_phase2_view,
    render_phase3_view,
    render_phase4_view,
    render_phase5_view,
    render_phase6_view
)

# Page configuration
st.set_page_config(
    page_title="Food Delivery Big Data Analytics",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Loads CSV dataset safely into memory."""
    path = Path(file_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    # 1. Top Header & Academic Problem Statement
    render_header(is_synthetic=True)

    # 2. Sidebar Controls: Navigation & Dataset Management
    st.sidebar.markdown("## 🧭 Syllabus Phase Navigation")
    
    phases = [
        "1. Introduction to Big Data",
        "2. Big Data Frameworks",
        "3. MapReduce Paradigm",
        "4. Mining Big Data Streams",
        "5. Big Data Mining Algorithms",
        "6. Link Analysis, Mining Social-Network Graphs and Recommendation Engines"
    ]

    if "current_phase_idx" not in st.session_state:
        st.session_state["current_phase_idx"] = 0

    selected_phase = st.sidebar.radio(
        "Choose Phase / Syllabus Module:",
        phases,
        index=st.session_state["current_phase_idx"],
        key="phase_nav_radio"
    )
    # Sync selected index
    phase_idx = phases.index(selected_phase)
    st.session_state["current_phase_idx"] = phase_idx

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Dataset & Pipeline Controls")
    
    dataset_choice = st.sidebar.selectbox(
        "Active Dataset View:",
        [
            "Processed Data (Cleaned & Validated)",
            "Raw Data (With Injected Veracity Anomalies)",
            "Synthetic Source Archive"
        ],
        index=0
    )

    path_map = {
        "Processed Data (Cleaned & Validated)": PROCESSED_ORDERS_FILE,
        "Raw Data (With Injected Veracity Anomalies)": RAW_ORDERS_FILE,
        "Synthetic Source Archive": SYNTHETIC_ORDERS_FILE
    }
    target_path = path_map[dataset_choice]

    # Pipeline Operation Triggers
    with st.sidebar.expander("🔄 Pipeline Maintenance", expanded=False):
        if st.button("Regenerate Data (Seed 42)", use_container_width=True):
            with st.spinner("Generating reproducible dataset..."):
                new_df = generate_synthetic_orders(include_noise=True)
                save_dataset(new_df, SYNTHETIC_ORDERS_FILE)
                save_dataset(new_df, RAW_ORDERS_FILE)
                st.cache_data.clear()
                st.success("Dataset regenerated!")
                st.rerun()

        if st.button("Run Preprocessing Pipeline", use_container_width=True):
            with st.spinner("Executing veracity cleaning..."):
                run_pipeline()
                st.cache_data.clear()
                st.success("Cleaned dataset updated!")
                st.rerun()

    # Automatic initial bootstrap if dataset is absent
    if not target_path.exists():
        st.warning(f"Dataset file not found at `{target_path}`. Bootstrapping initial data...")
        init_df = generate_synthetic_orders(include_noise=True)
        save_dataset(init_df, SYNTHETIC_ORDERS_FILE)
        save_dataset(init_df, RAW_ORDERS_FILE)
        run_pipeline()
        st.cache_data.clear()
        st.rerun()

    # Load active data
    full_df = load_data(str(target_path))
    if full_df.empty:
        st.error(f"Selected dataset at `{target_path}` is empty.")
        return

    # Baseline KPIs
    full_kpis = compute_executive_kpis(full_df)

    # 3. Interactive Multi-Dimensional Filters
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Multi-Dimensional Filters")

    # Clear Filters Button
    if st.sidebar.button("Reset All Filters", use_container_width=True):
        st.rerun()

    # Filter State Variables
    start_d, end_d = None, None
    day_type = "All"
    hour_range = None
    selected_zones = None
    selected_rest_types = None
    selected_rest_ids = None
    selected_statuses = None
    selected_weather = None
    selected_traffic = None

    # Temporal Filters
    if "order_timestamp" in full_df.columns:
        valid_ts = pd.to_datetime(full_df["order_timestamp"], errors="coerce").dropna()
        if not valid_ts.empty:
            min_date = valid_ts.min().date()
            max_date = valid_ts.max().date()
            if min_date < max_date:
                date_input = st.sidebar.date_input(
                    "Date Range:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
                    start_d, end_d = date_input

        day_type = st.sidebar.radio(
            "Day of Week Segment:",
            ["All", "Weekdays (Mon-Fri)", "Weekends (Sat-Sun)"],
            horizontal=True
        )

        hour_range = st.sidebar.slider(
            "Hour of Day (0–23):",
            min_value=0,
            max_value=23,
            value=(0, 23)
        )

    # Spatial & Entity Filters
    if "delivery_zone" in full_df.columns:
        zone_opts = sorted([str(z) for z in full_df["delivery_zone"].dropna().unique()])
        selected_zones = st.sidebar.multiselect("Delivery Zone:", options=zone_opts, default=[])

    if "restaurant_type" in full_df.columns:
        rtype_opts = sorted([str(r) for r in full_df["restaurant_type"].dropna().unique()])
        selected_rest_types = st.sidebar.multiselect("Restaurant Type:", options=rtype_opts, default=[])

    if "restaurant_id" in full_df.columns:
        rid_opts = sorted([str(r) for r in full_df["restaurant_id"].dropna().unique()])
        selected_rest_ids = st.sidebar.multiselect("Restaurant ID:", options=rid_opts, default=[])

    # Operational Filters
    if "order_status" in full_df.columns:
        status_opts = sorted([str(s) for s in full_df["order_status"].dropna().unique()])
        selected_statuses = st.sidebar.multiselect("Order Status:", options=status_opts, default=[])

    if "weather_condition" in full_df.columns:
        weather_opts = sorted([str(w) for w in full_df["weather_condition"].dropna().unique()])
        selected_weather = st.sidebar.multiselect("Weather Condition:", options=weather_opts, default=[])

    if "traffic_density" in full_df.columns:
        traffic_opts = sorted([str(t) for t in full_df["traffic_density"].dropna().unique()])
        selected_traffic = st.sidebar.multiselect("Traffic Density:", options=traffic_opts, default=[])

    # Apply Filters
    filtered_df = apply_dashboard_filters(
        full_df,
        start_date=start_d,
        end_date=end_d,
        selected_zones=selected_zones,
        selected_restaurant_types=selected_rest_types,
        selected_restaurant_ids=selected_rest_ids,
        selected_statuses=selected_statuses,
        selected_weather=selected_weather,
        selected_traffic=selected_traffic,
        day_type_filter=day_type,
        hour_range=hour_range
    )

    # Record Counter Display
    n_filtered = len(filtered_df)
    n_total = len(full_df)
    pct_retained = round((n_filtered / n_total * 100.0), 1) if n_total > 0 else 0.0
    st.sidebar.markdown(f"**Matching Records:** `{n_filtered:,} / {n_total:,}` ({pct_retained}%)")

    # 4. Handle Empty Filter Results Gracefully
    if filtered_df.empty:
        st.warning("⚠️ **No records match the selected filter criteria.** Please adjust or reset the filters in the sidebar.")
        return

    # 5. Render Interconnected Pipeline Banner
    render_pipeline_connectivity_banner(phase_idx)

    # 6. Render Active Syllabus Phase View
    if phase_idx == 0:
        render_phase1_view(filtered_df, full_dataset_kpis=full_kpis)
    elif phase_idx == 1:
        render_phase2_view(filtered_df)
    elif phase_idx == 2:
        render_phase3_view(filtered_df)
    elif phase_idx == 3:
        render_phase4_view(filtered_df)
    elif phase_idx == 4:
        render_phase5_view(filtered_df)
    elif phase_idx == 5:
        render_phase6_view(filtered_df)

    # 7. Sequential Phase Progression (Interconnecting all 6 Phases)
    st.markdown("---")
    st.markdown("##### 🧭 Sequential Phase Progression")
    nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
    with nav_c1:
        if phase_idx > 0:
            prev_label = phases[phase_idx - 1].split(".")[1].strip()
            if st.button(f"⬅️ Previous: {prev_label}", use_container_width=True, key="prev_phase_btn"):
                st.session_state["current_phase_idx"] = phase_idx - 1
                st.rerun()
    with nav_c3:
        if phase_idx < 5:
            next_label = phases[phase_idx + 1].split(".")[1].strip()
            if st.button(f"Next: {next_label} ➡️", use_container_width=True, key="next_phase_btn"):
                st.session_state["current_phase_idx"] = phase_idx + 1
                st.rerun()
        else:
            if st.button("🔄 Back to Phase 1", use_container_width=True, key="restart_phase_btn"):
                st.session_state["current_phase_idx"] = 0
                st.rerun()

    # Footer
    st.markdown("---")
    st.caption(
        "Big Data Analytics Mini Project — Phase 2 Implementation | "
        "VESIT Department of Information Technology (NEP Batch 1) | Student: Jai Udasi (D20A_34)"
    )


if __name__ == "__main__":
    main()
