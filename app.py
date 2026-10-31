"""
Universal Multimodal & Signal Intelligence Engine - Web Dashboard
Interactive Streamlit application implementing the 5-Stage Universal Architecture.
"""

from __future__ import annotations
import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from universal_engine.models import ModalityType, EngineOutput
from universal_engine.dsp import UniversalSignalProcessor
from universal_engine.memory import UniversalMemoryStore
from universal_engine.generators import SignalDataGenerator
from universal_engine.llm import (
    create_llm_client,
    DeterministicLocalEngine,
    GeminiClient,
    OllamaClient,
)
from universal_engine.engine import UniversalIntelligenceEngine

# -------------------------------------------------------------
# 1. Page Configuration & Theme
# -------------------------------------------------------------
st.set_page_config(
    page_title="Universal Signal AI Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .takeaway-card {
        background: #ffffff;
        border-left: 4px solid #1E88E5;
        border-radius: 4px 8px 8px 4px;
        padding: 12px 18px;
        margin-bottom: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        font-size: 0.98rem;
    }
    .qa-card {
        background: #fdfefe;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .qa-q {
        font-weight: 600;
        color: #1e293b;
        font-size: 1.0rem;
        margin-bottom: 6px;
    }
    .qa-a {
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
        background: #f1f5f9;
        padding: 8px 12px;
        border-radius: 6px;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 12px;
        background: #e0f2fe;
        color: #0369a1;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 2. Session State Initialization
# -------------------------------------------------------------
if "memory_store" not in st.session_state:
    st.session_state.memory_store = UniversalMemoryStore(db_path=":memory:")

if "last_output" not in st.session_state:
    st.session_state.last_output = None

if "last_features" not in st.session_state:
    st.session_state.last_features = None


# -------------------------------------------------------------
# 3. Sidebar: Configuration & Ingestion Controls
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    tier_choice = st.selectbox(
        "Deployment Tier",
        options=[
            "Deterministic Local (Edge / Air-Gapped)",
            "Google Gemini Cloud (API)",
            "Local Ollama (DeepSeek-R1 / vLLM)",
        ],
        index=0,
    )

    api_key_input = ""
    ollama_url = "http://localhost:11434"
    if "Gemini" in tier_choice:
        api_key_input = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Defaults to GEMINI_API_KEY environment variable if set",
        )
    elif "Ollama" in tier_choice:
        ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434")

    st.markdown("---")
    st.markdown("### 📡 Signal & Modality")
    modality_label = st.radio(
        "Select Signal Type",
        options=[
            "IoT & Industrial Vibration",
            "Biomedical (ECG / HRV)",
            "RF & Electromagnetic (I/Q)",
            "Tabular Telemetry / Event Logs",
            "Technical Documents & Manuals",
        ],
        index=0,
    )

    source_type = st.radio("Data Source", ["Synthetic Production Scenario", "Custom Upload"], index=0)

    st.markdown("---")
    st.markdown("### 🗄️ DuckDB Memory Store")
    total_runs = st.session_state.memory_store.con.execute("SELECT count(*) FROM signal_metrics").fetchone()[0]
    st.caption(f"Persistent runs in DuckDB: **{total_runs}**")
    if st.button("Reset Memory Store", use_container_width=True):
        st.session_state.memory_store = UniversalMemoryStore(db_path=":memory:")
        st.session_state.last_output = None
        st.rerun()


# -------------------------------------------------------------
# 4. Data Preparation / Ingestion Sandbox
# -------------------------------------------------------------
raw_signal_array = None
time_axis = None
q_signal_array = None
df_input = None
doc_text = ""
sampling_rate = 1000.0
modality_enum = ModalityType.IOT_SENSOR

if modality_label == "IoT & Industrial Vibration":
    modality_enum = ModalityType.IOT_SENSOR
    sampling_rate = 2000.0
    if source_type == "Synthetic Production Scenario":
        vibration_type = st.selectbox(
            "Fault Scenario",
            ["Bearing Outer Race Defect (BPFO @ 14.2 Hz)", "Baseline Normal Operation", "Critical Multi-Harmonic Unbalance"],
        )
        has_anom = "Bearing" in vibration_type or "Critical" in vibration_type
        time_axis, raw_signal_array = SignalDataGenerator.generate_iot_vibration(
            duration_sec=2.0,
            sampling_rate_hz=sampling_rate,
            bearing_fault_freq_hz=14.2 if "Bearing" in vibration_type else 28.4,
            add_anomalies=has_anom,
        )
        doc_text = SignalDataGenerator.sample_technical_document()
    else:
        uploaded = st.file_uploader("Upload CSV / NPY vibration file", type=["csv", "npy", "txt"])
        if uploaded:
            if uploaded.name.endswith(".csv"):
                df_u = pd.read_csv(uploaded)
                raw_signal_array = df_u.iloc[:, -1].values
            elif uploaded.name.endswith(".npy"):
                raw_signal_array = np.load(uploaded)
            time_axis = np.linspace(0, len(raw_signal_array) / sampling_rate, len(raw_signal_array))

elif modality_label == "Biomedical (ECG / HRV)":
    modality_enum = ModalityType.BIOMEDICAL
    sampling_rate = 250.0
    if source_type == "Synthetic Production Scenario":
        ecg_scenario = st.selectbox("Clinical Condition", ["Premature Ventricular Contractions (PVC Ectopic)", "Normal Sinus Rhythm (75 BPM)"])
        time_axis, raw_signal_array = SignalDataGenerator.generate_biomedical_ecg(
            duration_sec=6.0,
            sampling_rate_hz=sampling_rate,
            inject_arrhythmia="PVC" in ecg_scenario,
        )
        doc_text = """# CLINICAL TELEMETRY MONITORING PROTOCOL
- Target Baseline: Lead II Electrocardiography at 250 Hz
- Normal Range: Heart Rate 60–100 BPM, QRS duration < 120 ms
- Alert Criteria: Ventricular ectopy (> 3 PVCs/min), severe bradycardia (< 50 BPM) or tachycardia (> 110 BPM).
"""
    else:
        uploaded = st.file_uploader("Upload ECG signal file (.csv, .npy)", type=["csv", "npy"])
        if uploaded:
            if uploaded.name.endswith(".csv"):
                raw_signal_array = pd.read_csv(uploaded).iloc[:, -1].values
            else:
                raw_signal_array = np.load(uploaded)
            time_axis = np.linspace(0, len(raw_signal_array) / sampling_rate, len(raw_signal_array))

elif modality_label == "RF & Electromagnetic (I/Q)":
    modality_enum = ModalityType.RF_ELECTROMAGNETIC
    sampling_rate = 1e6
    if source_type == "Synthetic Production Scenario":
        snr_target = st.slider("Target SNR (dB)", min_value=5.0, max_value=35.0, value=22.0)
        time_axis, raw_signal_array, q_signal_array = SignalDataGenerator.generate_rf_iq(
            duration_sec=0.002,
            sampling_rate_hz=sampling_rate,
            snr_db=snr_target,
        )
        doc_text = """# RADAR TELEMETRY SPECIFICATION
- Band: S-Band Baseband Receiver
- Sampling Rate: 1.00 MSps Complex I/Q
- Minimum Acceptable SNR: 12.0 dB
- Max Permissible Carrier Drift: ± 50.0 kHz
"""
    else:
        uploaded = st.file_uploader("Upload I/Q CSV (first 2 columns = I, Q)", type=["csv"])
        if uploaded:
            df_iq = pd.read_csv(uploaded)
            raw_signal_array = df_iq.iloc[:, 0].values
            q_signal_array = df_iq.iloc[:, 1].values if df_iq.shape[1] > 1 else np.zeros_like(raw_signal_array)
            time_axis = np.linspace(0, len(raw_signal_array) / sampling_rate, len(raw_signal_array))

elif modality_label == "Tabular Telemetry / Event Logs":
    modality_enum = ModalityType.TABULAR_LOGS
    if source_type == "Synthetic Production Scenario":
        df_input = SignalDataGenerator.generate_tabular_telemetry(num_records=150)
        doc_text = """# PUMP HYDRAULIC LOG THRESHOLDS
- Expected Pressure: 115 - 128 PSI
- Temperature Alert Threshold: > 80.0°C
- Vibration Outlier Tolerance: < 3.0 g RMS
"""
    else:
        uploaded = st.file_uploader("Upload tabular dataset (.csv)", type=["csv"])
        if uploaded:
            df_input = pd.read_csv(uploaded)

elif modality_label == "Technical Documents & Manuals":
    modality_enum = ModalityType.DOCUMENTS_SCANS
    default_text = SignalDataGenerator.sample_technical_document()
    doc_text = st.text_area("Input Document / Technical SOP Text", value=default_text, height=220)


# -------------------------------------------------------------
# 5. Header & Top Title
# -------------------------------------------------------------
st.markdown('<div class="main-title">⚡ Universal Multimodal & Signal AI Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Zero-egress DSP sandbox • Deterministic verification • Dual-store DuckDB memory • Plain-language synthesis</div>',
    unsafe_allow_html=True,
)

# Active deployment badge row
tier_badge = "🛡️ Air-Gapped / Deterministic Sandbox" if "Deterministic" in tier_choice else ("☁️ Cloud Gemini 2.5" if "Gemini" in tier_choice else "🖥️ Local Ollama")
st.markdown(
    f'<span class="badge-pill">{tier_badge}</span>'
    f'<span class="badge-pill">Modality: {modality_enum.value.upper()}</span>'
    f'<span class="badge-pill">Verification: 100% Deterministic DSP</span>',
    unsafe_allow_html=True,
)
st.write("")

# -------------------------------------------------------------
# 6. Primary Action: Feature Extraction & Execution
# -------------------------------------------------------------
col_action1, col_action2 = st.columns([1, 4])
with col_action1:
    run_btn = st.button("🚀 Process & Reason", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("Executing DSP Sandbox & Multimodal Reasoning Layer..."):
        # Configure LLM Client
        if "Gemini" in tier_choice:
            client = create_llm_client("gemini", gemini_api_key=api_key_input)
        elif "Ollama" in tier_choice:
            client = create_llm_client("ollama", ollama_url=ollama_url)
        else:
            client = create_llm_client("deterministic")

        engine = UniversalIntelligenceEngine(
            model_client=client,
            memory_store=st.session_state.memory_store,
        )

        # Dispatch Feature Extraction
        if modality_enum == ModalityType.TABULAR_LOGS and df_input is not None:
            features = UniversalSignalProcessor.extract_features(
                input_data=df_input,
                modality=modality_enum,
            )
        elif modality_enum == ModalityType.DOCUMENTS_SCANS:
            features = UniversalSignalProcessor.extract_features(
                input_data=doc_text,
                modality=modality_enum,
            )
        elif raw_signal_array is not None:
            extra_kwargs = {}
            if q_signal_array is not None:
                extra_kwargs["q_signal"] = q_signal_array

            features = UniversalSignalProcessor.extract_features(
                input_data=raw_signal_array,
                modality=modality_enum,
                sampling_rate_hz=sampling_rate,
                **extra_kwargs,
            )
        else:
            features = None

        if features:
            st.session_state.last_features = features
            output = engine.analyze(
                raw_data_summary=features,
                document_text=doc_text,
            )
            st.session_state.last_output = output
        else:
            st.error("No valid signal array or tabular dataset loaded.")


# -------------------------------------------------------------
# 7. Rendering Pipeline Outputs
# -------------------------------------------------------------
if st.session_state.last_output and st.session_state.last_features:
    output: EngineOutput = st.session_state.last_output
    features = st.session_state.last_features

    st.write("---")

    # KPI Metric Cards
    f_map = features.raw_feature_map or {}
    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        dur = f_map.get("duration_seconds", 0.0)
        st.markdown(f'<div class="metric-card"><div class="metric-val">{dur:.3f}s</div><div class="metric-lbl">Duration</div></div>', unsafe_allow_html=True)
    with kpi_cols[1]:
        rms_val = f_map.get("rms", f_map.get("mean_power", 0.0))
        st.markdown(f'<div class="metric-card"><div class="metric-val">{rms_val:.2f}</div><div class="metric-lbl">RMS / Power</div></div>', unsafe_allow_html=True)
    with kpi_cols[2]:
        dom_freq = f_map.get("dominant_frequency_hz", f_map.get("average_heart_rate_bpm", f_map.get("carrier_frequency_offset_hz", 0.0)))
        lbl = "Heart Rate" if "ecg" in str(f_map.get("modality_subtype")) else ("Carrier Offset" if "rf" in str(f_map.get("modality_subtype")) else "Peak Frequency")
        st.markdown(f'<div class="metric-card"><div class="metric-val">{dom_freq:.1f}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)
    with kpi_cols[3]:
        anom_cnt = f_map.get("anomalies_detected", f_map.get("outlier_counts", 0))
        if isinstance(anom_cnt, dict):
            anom_cnt = sum(anom_cnt.values())
        st.markdown(f'<div class="metric-card"><div class="metric-val">{anom_cnt}</div><div class="metric-lbl">Outliers / Anomalies</div></div>', unsafe_allow_html=True)
    with kpi_cols[4]:
        latency = output.metadata.get("processing_latency_sec", 0.0)
        st.markdown(f'<div class="metric-card"><div class="metric-val">{latency}s</div><div class="metric-lbl">Processing Latency</div></div>', unsafe_allow_html=True)

    st.write("")

    # Output Tabs
    tabs = st.tabs([
        "🎯 Executive Takeaways",
        "📈 Dynamic Visuals & Charts",
        "💡 Grounded Q&A Flashcards",
        "🗄️ DuckDB SQL Explorer",
        "📜 Structured JSON Audit",
    ])

    # Tab 1: Executive Bullet Points
    with tabs[0]:
        st.subheader("Executive Plain-Language Takeaways")
        st.caption("Distilled, active-verb takeaways translating technical metrics into business & operational impact.")
        for bullet in output.executive_takeaways:
            st.markdown(f'<div class="takeaway-card">{bullet}</div>', unsafe_allow_html=True)

    # Tab 2: Automated Visuals & Graphs
    with tabs[1]:
        st.subheader("Automated Declarative Visualizations")
        if output.charts:
            for chart in output.charts:
                st.markdown(f"**{chart.title}**")
                fig = None

                # Series-based line chart
                if chart.series:
                    fig = go.Figure()
                    x_vals = chart.series.get("time_axis")
                    for s_name, s_vals in chart.series.items():
                        if s_name == "time_axis":
                            continue
                        x_pts = x_vals if x_vals and len(x_vals) == len(s_vals) else list(range(len(s_vals)))
                        fig.add_trace(go.Scatter(x=x_pts, y=s_vals, mode="lines", name=s_name, line=dict(color="#1E88E5", width=2)))
                    fig.update_layout(
                        xaxis_title=chart.x_label,
                        yaxis_title=chart.y_label,
                        margin=dict(l=40, r=40, t=30, b=40),
                        template="plotly_white",
                        height=360,
                    )
                # Point-based bar/pie chart
                elif chart.data:
                    labels = [pt.label for pt in chart.data]
                    values = [pt.value for pt in chart.data]
                    if chart.chart_type == "bar":
                        fig = px.bar(x=labels, y=values, labels={"x": chart.x_label, "y": chart.y_label})
                        fig.update_traces(marker_color="#0284c7")
                    elif chart.chart_type == "pie":
                        fig = px.pie(names=labels, values=values)
                    else:
                        fig = px.line(x=labels, y=values, labels={"x": chart.x_label, "y": chart.y_label})
                    fig.update_layout(
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=30, b=40),
                        height=340,
                    )

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No explicit charts recommended by model for this input.")

        # Additional Raw Waveform Viewer if available
        if raw_signal_array is not None and time_axis is not None:
            with st.expander("🔬 High-Resolution Ingestion Waveform (DSP Sandbox Viewer)", expanded=False):
                step = max(1, len(raw_signal_array) // 1000)
                sub_t = time_axis[::step]
                sub_sig = raw_signal_array[::step]
                fig_raw = px.line(x=sub_t, y=sub_sig, labels={"x": "Time (s)", "y": "Signal Amplitude"})
                fig_raw.update_traces(line=dict(color="#0f766e", width=1.2))
                fig_raw.update_layout(template="plotly_white", height=280, margin=dict(l=40, r=40, t=20, b=40))
                st.plotly_chart(fig_raw, use_container_width=True)

    # Tab 3: Autonomous Q&A
    with tabs[2]:
        st.subheader("Autonomous Self-Assessment Q&A Pairs")
        st.caption("Key stakeholder questions answered with direct 1–2 sentence statements backed by input data.")
        for idx, qa in enumerate(output.qa_pairs, start=1):
            with st.container():
                st.markdown(f'<div class="qa-card"><div class="qa-q">Q{idx}: {qa.question}</div><div class="qa-a"><b>Answer:</b> {qa.answer}</div></div>', unsafe_allow_html=True)

    # Tab 4: DuckDB Time-Series Metric Store Explorer
    with tabs[3]:
        st.subheader("Embedded DuckDB Metric Store")
        st.caption("Run deterministic SQL queries over ingested signal summaries and time-series records.")

        query_col1, query_col2 = st.columns([3, 1])
        with query_col1:
            sql_input = st.text_area(
                "SQL Query",
                value="SELECT signal_id, modality, rms, dominant_frequency_hz, anomalies_detected, timestamp FROM signal_metrics ORDER BY timestamp DESC LIMIT 10",
                height=90,
            )
        with query_col2:
            st.write("")
            st.write("")
            execute_sql = st.button("Execute SQL", use_container_width=True)

        if execute_sql or True:
            try:
                res_df = st.session_state.memory_store.query_metrics_sql(sql_input)
                st.dataframe(res_df, use_container_width=True)
            except Exception as sql_err:
                st.error(f"SQL Error: {sql_err}")

        st.markdown("#### Hybrid Semantic Document Search")
        search_query = st.text_input("Search Indexed Manuals / Context", value="bearing defect lubrication")
        if search_query:
            results = st.session_state.memory_store.hybrid_retrieve(search_query, top_k=3)
            for r in results:
                st.markdown(f"**Chunk ID**: `{r['chunk_id']}` | **Hybrid Score**: `{r['score']}` (Dense: `{r['dense_score']}` / BM25: `{r['bm25_score']}`)")
                st.code(r["text"], language="markdown")

    # Tab 5: Raw JSON Audit Trail
    with tabs[4]:
        st.subheader("Verified Structured Output JSON")
        st.caption("Validated strictly against Pydantic EngineOutput schema.")
        json_str = output.model_dump_json(indent=2)
        st.code(json_str, language="json")
        st.download_button(
            "⬇️ Download EngineOutput JSON",
            data=json_str,
            file_name=f"signal_intelligence_output_{int(time.time())}.json",
            mime="application/json",
        )

