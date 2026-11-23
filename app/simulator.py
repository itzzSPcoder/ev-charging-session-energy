"""
Dehradun Green Charge Network - EV Charging Session Simulator
Interactive Streamlit Dashboard for Demonstration and Viva
Group G171 | Problem MP298 | SCSE3040 MLOps
"""

import time
import os
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="EV Charging Simulator | Group G171",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .error-card {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 10px;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)


# Load Champion Model (Direct fallback if FastAPI server is not running)
@st.cache_resource
def load_champion_model():
    model_path = "models/model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


model = load_champion_model()

# Header
st.markdown('<div class="main-title">⚡ Dehradun Green Charge Network --- Session Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title"><b>Group G171 | Problem MP298:</b> EV Charging Session Energy Prediction & Grid Simulation</div>', unsafe_allow_html=True)

# Tabs
tab_single, tab_traffic, tab_pipeline = st.tabs(["🚗 Single Session Simulator", "📊 24-Hour Network Load Simulator", "ℹ️ Pipeline & Model Architecture"])

# -------------------------------------------------------------
# TAB 1: Single Session Simulator
# -------------------------------------------------------------
with tab_single:
    col_input, col_result = st.columns([1.1, 1], gap="large")

    with col_input:
        st.subheader("1. Vehicle & Session Parameters")

        # Quick preset buttons
        preset = st.selectbox(
            "Select Vehicle Preset (or configure manually below):",
            [
                "Custom Configuration",
                "Tata Nexon EV (Car, Mall 30kW Charger)",
                "Ola S1 Pro (2-Wheeler, Residential 3.3kW)",
                "Mahindra Treo (3-Wheeler E-Rickshaw, Highway 15kW)",
                "MG ZS EV (Car, Highway 50kW Fast Charger)",
            ]
        )

        # Default values based on preset
        if preset == "Tata Nexon EV (Car, Mall 30kW Charger)":
            def_v, def_st, def_kw, def_soc, def_mins, def_temp, def_age, def_q = "car", "Mall", 30.0, 20, 60, 28.0, 1.5, 1
        elif preset == "Ola S1 Pro (2-Wheeler, Residential 3.3kW)":
            def_v, def_st, def_kw, def_soc, def_mins, def_temp, def_age, def_q = "two_wheeler", "Residential", 3.3, 30, 90, 25.0, 1.0, 0
        elif preset == "Mahindra Treo (3-Wheeler E-Rickshaw, Highway 15kW)":
            def_v, def_st, def_kw, def_soc, def_mins, def_temp, def_age, def_q = "three_wheeler", "Highway", 15.0, 15, 45, 32.0, 3.0, 3
        elif preset == "MG ZS EV (Car, Highway 50kW Fast Charger)":
            def_v, def_st, def_kw, def_soc, def_mins, def_temp, def_age, def_q = "car", "Highway", 50.0, 10, 40, 35.0, 0.5, 2
        else:
            def_v, def_st, def_kw, def_soc, def_mins, def_temp, def_age, def_q = "car", "Mall", 25.0, 25, 60, 28.0, 2.0, 1

        v_type = st.selectbox(
            "Vehicle Type:",
            ["car", "two_wheeler", "three_wheeler"],
            index=["car", "two_wheeler", "three_wheeler"].index(def_v),
            format_func=lambda x: {"car": "🚗 Electric Car", "two_wheeler": "🛵 Electric 2-Wheeler", "three_wheeler": "🛺 Electric 3-Wheeler"}[x]
        )

        st_type = st.selectbox(
            "Station Location:",
            ["Mall", "Highway", "Residential"],
            index=["Mall", "Highway", "Residential"].index(def_st),
            format_func=lambda x: {"Mall": "🏬 Shopping Mall Station", "Highway": "🛣️ Highway Fast-Charging Hub", "Residential": "🏡 Residential Community"}[x]
        )

        c1, c2 = st.columns(2)
        with c1:
            charger_kw = st.slider("Charger Output (kW):", min_value=3.0, max_value=60.0, value=float(def_kw), step=0.5)
            soc_start = st.slider("Battery SOC at Plug-In (%):", min_value=5, max_value=80, value=int(def_soc), step=1)
            age_yr = st.slider("Vehicle Age (years):", min_value=0.0, max_value=8.0, value=float(def_age), step=0.5)

        with c2:
            mins = st.number_input("Session Minutes (Duration):", min_value=1, max_value=300, value=int(def_mins), help="Client rule: Must be >= 5 minutes")
            temp = st.number_input("Ambient Air Temperature (°C):", min_value=0.0, max_value=100.0, value=float(def_temp), step=0.5, help="Client rule: Must be <= 75°C")
            queue = st.slider("Queue Ahead (waiting vehicles):", min_value=0, max_value=10, value=int(def_q), step=1)

        api_mode = st.toggle("Connect via Live FastAPI Service (http://localhost:8000/predict)", value=False)

    with col_result:
        st.subheader("2. Real-Time Energy & Cost Prediction")

        # Client Rule Validations (Live UI check mirroring FastAPI rules)
        rule_violation = False
        if temp > 75.0:
            st.error("❌ **Request Rejected (HTTP 422)**: Temperature above 75°C is physically impossible for the station sensors. Rejecting input!")
            rule_violation = True
        if mins < 5:
            st.error("❌ **Request Rejected (HTTP 422)**: Planned session below 5 minutes is considered a typing mistake. Rejecting input!")
            rule_violation = True

        if not rule_violation:
            payload = {
                "battery_start_pct": float(soc_start),
                "session_minutes": float(mins),
                "charger_kw": float(charger_kw),
                "temp_c": float(temp),
                "vehicle_age_yr": float(age_yr),
                "queue_before": int(queue),
                "vehicle": v_type,
                "station_type": st_type,
            }

            pred_val = None
            latency_ms = 0

            if api_mode:
                # Call live FastAPI endpoint
                t0 = time.time()
                try:
                    res = requests.post("http://localhost:8000/predict", json=payload, timeout=2.0)
                    latency_ms = int((time.time() - t0) * 1000)
                    if res.status_code == 200:
                        pred_val = res.json().get("prediction")
                        st.success(f"Connected to FastAPI Service (Response latency: {latency_ms} ms)")
                    else:
                        st.warning(f"FastAPI returned error {res.status_code}: {res.text}")
                except Exception as e:
                    st.info(f"FastAPI server not detected on localhost:8000. Falling back to local Champion model.")
                    if model:
                        t0 = time.time()
                        df_in = pd.DataFrame([payload])
                        pred_val = float(model.predict(df_in)[0])
                        latency_ms = int((time.time() - t0) * 1000)
            else:
                if model:
                    t0 = time.time()
                    df_in = pd.DataFrame([payload])
                    pred_val = float(model.predict(df_in)[0])
                    latency_ms = int((time.time() - t0) * 1000)
                else:
                    st.error("Champion model not found at `models/model.joblib`. Please train model first.")

            if pred_val is not None:
                # Key Metrics Cards
                m1, m2, m3 = st.columns(3)
                cost_estimate = pred_val * 9.50  # Average ₹9.50 per kWh in Uttarakhand EV tariff

                with m1:
                    st.metric("⚡ Predicted Energy", f"{pred_val:.2f} kWh")
                with m2:
                    st.metric("💰 Estimated Cost", f"₹ {cost_estimate:.1f}")
                with m3:
                    avg_power = (pred_val / (mins / 60.0)) if mins > 0 else 0
                    st.metric("🔌 Avg Power Drawn", f"{avg_power:.1f} kW")

                # Progress & Charging Curve Visualizer
                st.markdown("---")
                st.write("**Simulated State of Charge (SOC) Progression:**")
                time_steps = np.linspace(0, mins, 20)
                # Realistic charging curve with taper above 80%
                soc_progress = [min(100.0, soc_start + (100 - soc_start) * (1 - np.exp(-1.8 * (t / mins)))) for t in time_steps]

                chart_df = pd.DataFrame({
                    "Minutes": np.round(time_steps, 1),
                    "Battery %": np.round(soc_progress, 1),
                }).set_index("Minutes")

                st.line_chart(chart_df, color="#10B981")

                st.caption(f"ℹ️ Model version: **@champion (v2)** | Input inference time: **{latency_ms} ms** | Model artifact: `models/model.joblib`")

# -------------------------------------------------------------
# TAB 2: 24-Hour Network Load Simulator
# -------------------------------------------------------------
with tab_traffic:
    st.subheader("24-Hour Station Traffic & Grid Load Simulator")
    st.write("Simulate an entire day of EV charging across Dehradun Green Charge Network to visualize peak demand.")

    col_sim_btn, col_sim_stat = st.columns([1, 2])
    with col_sim_btn:
        n_vehicles = st.slider("Total Vehicles Arriving in 24h:", 20, 200, 50, step=10)
        sim_btn = st.button("🚀 Run 24-Hour Simulation", type="primary")

    if sim_btn and model:
        np.random.seed(42)
        hours = np.random.uniform(6, 23, n_vehicles)
        vehicles = np.random.choice(["car", "two_wheeler", "three_wheeler"], p=[0.3, 0.4, 0.3], size=n_vehicles)
        stations = np.random.choice(["Mall", "Highway", "Residential"], p=[0.3, 0.4, 0.3], size=n_vehicles)
        chargers = [50.0 if s == "Highway" and v == "car" else (22.0 if s == "Mall" else 7.4) for s, v in zip(stations, vehicles)]
        socs = np.random.randint(10, 60, n_vehicles)
        durations = np.random.randint(20, 120, n_vehicles)
        temps = [25.0 + 15.0 * np.sin(np.pi * (h - 6) / 16) for h in hours]
        ages = np.random.uniform(0.5, 6.0, n_vehicles)
        queues = np.random.randint(0, 6, n_vehicles)

        sim_df = pd.DataFrame({
            "hour": np.round(hours, 1),
            "battery_start_pct": socs,
            "session_minutes": durations,
            "charger_kw": chargers,
            "temp_c": np.round(temps, 1),
            "vehicle_age_yr": np.round(ages, 1),
            "queue_before": queues,
            "vehicle": vehicles,
            "station_type": stations,
        })

        features = ["battery_start_pct", "session_minutes", "charger_kw", "temp_c", "vehicle_age_yr", "queue_before", "vehicle", "station_type"]
        sim_df["predicted_kwh"] = np.round(model.predict(sim_df[features]), 2)
        sim_df["revenue_inr"] = np.round(sim_df["predicted_kwh"] * 9.5, 1)

        total_energy = sim_df["predicted_kwh"].sum()
        total_rev = sim_df["revenue_inr"].sum()

        r1, r2, r3 = st.columns(3)
        r1.metric("Total Daily Energy Delivered", f"{total_energy:.1f} kWh")
        r2.metric("Total Estimated Revenue", f"₹ {total_rev:,.0f}")
        r3.metric("Peak Concurrent Sessions", f"{len(sim_df[sim_df['hour'].between(17, 20)])} EVs (Evening Peak)")

        st.write("##### Hourly Energy Demand Curve (kWh):")
        hourly_summary = sim_df.groupby(sim_df["hour"].astype(int))["predicted_kwh"].sum().reset_index()
        hourly_summary.columns = ["Hour of Day", "Total Energy (kWh)"]
        st.bar_chart(hourly_summary.set_index("Hour of Day"), color="#3B82F6")

        st.write("##### Station Type Breakdown:")
        st.dataframe(sim_df.groupby("station_type")[["predicted_kwh", "revenue_inr"]].sum().reset_index(), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: Pipeline & Architecture Info
# -------------------------------------------------------------
with tab_pipeline:
    st.subheader("System Architecture & Implementation")
    st.markdown("""
    - **Course**: SCSE3040 MLOps
    - **Group**: G171 | **Problem**: MP298 (EV Charging Session Energy)
    - **Client**: Dehradun Green Charge Network
    - **Members**:
        - **Sparsh Bansal (S24CSEU0560)** --- Ops & Service Lead (FastAPI, Docker, K8s, CI)
        - **Siddharth Rawat (S24CSEU0542)** --- Data Lead (DVC, 7 Data Issues, Cleaning)
        - **Tanisha Rathore (S24CSEU0591)** --- Model Lead (MLflow Tracking, Drift PSI, Retraining)
    - **Champion Model**: Gradient Boosting Regressor (v2) promoted after achieving **57.28% error reduction** over v1.
    - **GitHub Repo**: [https://github.com/itzzSPcoder/ev-charging-session-energy](https://github.com/itzzSPcoder/ev-charging-session-energy)
    """)
