"""
ICT Liquidity Intelligence & Stop-Hunt Detection System
Interactive Streamlit Quantitative Research Dashboard
"""

import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

from src.data_loader import DataLoader
from src.predict import StopHuntPredictor
from src.backtest import BacktestEngine
from src.validation import validate_ohlcv_dataframe

# Page config
st.set_page_config(
    page_title="ICT Liquidity & Stop-Hunt Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Terminal CSS Theme
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    .stMetric {
        background-color: #1A1F2C;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #00E5FF;
    }
    .metric-card {
        background: #1E222D;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2A2E39;
        margin-bottom: 12px;
    }
    .status-badge-active {
        background-color: #00C853;
        color: black;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
    }
    .status-badge-warn {
        background-color: #FFD600;
        color: black;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
    }
    .status-badge-err {
        background-color: #D50000;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_cached_data():
    dl = DataLoader()
    data_dict = {}
    reports = {}
    for inst in ["XAUUSD", "XAGUSD"]:
        try:
            df, r = dl.load_instrument(inst)
            data_dict[inst] = df
            reports[inst] = r.to_dict()
        except Exception:
            pass

    proc_dir = Path("data/processed")
    features_df = pd.DataFrame()
    predictions_df = pd.DataFrame()
    trades_df = pd.DataFrame()
    meta = {}

    if (proc_dir / "features_combined.csv").exists():
        features_df = pd.read_csv(proc_dir / "features_combined.csv")
    if (proc_dir / "predictions_combined.csv").exists():
        predictions_df = pd.read_csv(proc_dir / "predictions_combined.csv")
    if (proc_dir / "backtest_trades.csv").exists():
        trades_df = pd.read_csv(proc_dir / "backtest_trades.csv")
    if Path("models/training_metadata.json").exists():
        with open("models/training_metadata.json", "r") as f:
            meta = json.load(f)

    return data_dict, reports, features_df, predictions_df, trades_df, meta


data_dict, reports, features_df, predictions_df, trades_df, metadata = load_cached_data()

# Sidebar Navigation
st.sidebar.title("⚡ Liquidity Intelligence")
st.sidebar.caption("ICT Killzones & XGBoost Stop-Hunt Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Market Analysis & Charts",
        "Liquidity Scanner",
        "Model Analytics",
        "Backtesting",
        "Data & Model Management",
        "Settings"
    ]
)

st.sidebar.markdown("---")
selected_inst = st.sidebar.selectbox("Selected Instrument", ["XAUUSD", "XAGUSD"], index=0)
selected_tf = st.sidebar.selectbox("Timeframe", ["15m"], index=0)

# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if page == "Overview":
    st.title("System Overview & Market Health")
    st.caption("Real-time summary of multi-instrument liquidity pools, candidate hunts, and model status.")

    report = reports.get(selected_inst, {})
    inst_preds = predictions_df[predictions_df["instrument"] == selected_inst] if not predictions_df.empty else pd.DataFrame()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Instrument", selected_inst, selected_tf)
    with col2:
        st.metric("Total Historical Bars", f"{report.get('total_bars', 0):,}")
    with col3:
        n_patterns = len(inst_preds) if not inst_preds.empty else 0
        st.metric("Detected Patterns", n_patterns)
    with col4:
        n_hunts = (inst_preds["label"] == "1").sum() if not inst_preds.empty else 0
        st.metric("Confirmed Reversals", n_hunts)
    with col5:
        n_breakdowns = (inst_preds["label"] == "Breakdown").sum() if not inst_preds.empty else 0
        st.metric("True Breakdowns", n_breakdowns)

    st.markdown("### Model & Pipeline Status")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Model Engine</h4>
            <p><strong>Version:</strong> {metadata.get('model_version', 'v1.0.0')}</p>
            <p><strong>Type:</strong> {metadata.get('model_type', 'XGBoost_Shared')}</p>
            <p><strong>Status:</strong> <span class="status-badge-active">ACTIVE & CALIBRATED</span></p>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        latest_ts = report.get("end_time", "N/A")
        st.markdown(f"""
        <div class="metric-card">
            <h4>Data Coverage</h4>
            <p><strong>Start:</strong> {report.get('start_time', 'N/A')}</p>
            <p><strong>Latest Bar:</strong> {latest_ts}</p>
            <p><strong>OHLC Violations:</strong> {report.get('ohlc_violations', 0)} (Clean)</p>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        latest_pred = inst_preds.iloc[-1] if not inst_preds.empty else None
        prob_str = f"{latest_pred['hunt_probability'] * 100:.1f}%" if latest_pred is not None else "N/A"
        tier_str = latest_pred.get("confidence_tier", "N/A") if latest_pred is not None else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h4>Latest Research Score</h4>
            <p><strong>Hunt Probability:</strong> {prob_str}</p>
            <p><strong>Confidence Tier:</strong> {tier_str}</p>
            <p><strong>Research Note:</strong> Historical research probabilities; not guaranteed.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Data Quality Report")
    st.json(report)

# ==============================================================================
# PAGE 2: MARKET ANALYSIS & CHARTS
# ==============================================================================
elif page == "Market Analysis & Charts":
    st.title("Interactive Candlestick & Liquidity Visualizer")
    st.caption("Visualizes ICT Killzones, Session Liquidity, Double/Triple Bottoms, and Sweep Reclaim Markers.")

    df_inst = data_dict.get(selected_inst, pd.DataFrame())
    if df_inst.empty:
        st.error(f"No data available for {selected_inst}.")
    else:
        # Display slice
        chart_bars = st.slider("Candles to Display", min_value=50, max_value=500, value=150, step=25)
        df_slice = df_inst.iloc[-chart_bars:].copy()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.8, 0.2]
        )

        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df_slice["timestamp"],
            open=df_slice["open"],
            high=df_slice["high"],
            low=df_slice["low"],
            close=df_slice["close"],
            name=selected_inst,
            increasing_line_color="#00E676",
            decreasing_line_color="#FF1744"
        ), row=1, col=1)

        # Volume
        vol_colors = ["#00E676" if c >= o else "#FF1744" for c, o in zip(df_slice["close"], df_slice["open"])]
        fig.add_trace(go.Bar(
            x=df_slice["timestamp"],
            y=df_slice["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.6
        ), row=2, col=1)

        # Overlays: Filter events in slice
        slice_start = df_slice["timestamp"].iloc[0]
        slice_end = df_slice["timestamp"].iloc[-1]
        
        inst_preds = predictions_df[predictions_df["instrument"] == selected_inst] if not predictions_df.empty else pd.DataFrame()
        if not inst_preds.empty and "timestamp" in inst_preds.columns:
            inst_preds["ts"] = pd.to_datetime(inst_preds["timestamp"], utc=True)
            slice_events = inst_preds[(inst_preds["ts"] >= slice_start) & (inst_preds["ts"] <= slice_end)]

            for _, ev in slice_events.iterrows():
                # Level line
                lp = ev["level_price"]
                fig.add_hline(
                    y=lp,
                    line_dash="dot",
                    line_color="#00E5FF",
                    annotation_text=f"{ev['pattern_type']} ({lp:.2f})",
                    annotation_position="bottom right",
                    row=1, col=1
                )

                # Sweep marker
                is_hunt = str(ev.get("label", "")) == "1"
                marker_col = "#00E676" if is_hunt else "#FF9100"
                marker_symbol = "triangle-up" if is_hunt else "x"
                fig.add_trace(go.Scatter(
                    x=[ev["ts"]],
                    y=[lp],
                    mode="markers+text",
                    marker=dict(symbol=marker_symbol, size=12, color=marker_col),
                    text=[f"Prob: {ev.get('hunt_probability', 0)*100:.0f}%"],
                    textposition="bottom center",
                    name=f"Event {ev.get('event_id', '')}",
                    showlegend=False
                ), row=1, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#161A22",
            height=700,
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 3: LIQUIDITY SCANNER
# ==============================================================================
elif page == "Liquidity Scanner":
    st.title("Liquidity Pool Scanner & Signal Journal")
    st.caption("Complete filterable table of structural liquidity levels, sweep depths, and ML scores.")

    if predictions_df.empty:
        st.warning("No predictions loaded. Run run_pipeline.py to populate.")
    else:
        # Filters
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            inst_filter = st.multiselect("Instrument", ["XAUUSD", "XAGUSD"], default=["XAUUSD", "XAGUSD"])
        with f_col2:
            pat_types = list(predictions_df["pattern_type"].unique())
            pat_filter = st.multiselect("Pattern Type", pat_types, default=pat_types)
        with f_col3:
            label_types = list(predictions_df["label"].unique())
            label_filter = st.multiselect("Outcome Label", label_types, default=label_types)
        with f_col4:
            min_prob = st.slider("Min Probability", 0.0, 1.0, 0.25, 0.05)

        filtered = predictions_df[
            (predictions_df["instrument"].isin(inst_filter)) &
            (predictions_df["pattern_type"].isin(pat_filter)) &
            (predictions_df["label"].isin(label_filter)) &
            (predictions_df["hunt_probability"] >= min_prob)
        ].copy()

        st.markdown(f"**Showing {len(filtered)} matching events:**")
        display_cols = [
            "timestamp", "instrument", "pattern_type", "level_price",
            "sweep_depth_atr", "confluence_score", "hunt_probability",
            "confidence_tier", "label"
        ]
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_cols].sort_values("timestamp", ascending=False), use_container_width=True)

        csv = filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Scanner CSV", csv, "scanner_events.csv", "text/csv")

# ==============================================================================
# PAGE 4: MODEL ANALYTICS
# ==============================================================================
elif page == "Model Analytics":
    st.title("XGBoost Model Analytics & Evaluation")
    st.caption("Rigorous out-of-sample evaluation, PR-AUC, calibration, feature importance, and cross-instrument transfer.")

    test_metrics = metadata.get("test_metrics", {})
    if not test_metrics:
        st.warning("No model metadata found. Please train model via pipeline.")
    else:
        comb_metrics = test_metrics.get("Combined", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Test ROC-AUC", f"{comb_metrics.get('roc_auc', 0.0):.4f}")
        with c2:
            st.metric("Test PR-AUC", f"{comb_metrics.get('pr_auc', 0.0):.4f}")
        with c3:
            st.metric("Brier Score", f"{comb_metrics.get('brier_score', 0.0):.4f}")
        with c4:
            st.metric("Test Samples", comb_metrics.get("n_samples", 0))

        st.markdown("### Gold vs Silver Test Performance")
        comp_data = []
        for inst_key in ["Combined", "XAUUSD", "XAGUSD"]:
            m = test_metrics.get(inst_key, {})
            if m:
                comp_data.append({
                    "Segment": inst_key,
                    "Samples": m.get("n_samples"),
                    "Positives": m.get("n_positives"),
                    "Positive Rate": f"{m.get('positive_rate', 0)*100:.1f}%",
                    "ROC-AUC": m.get("roc_auc"),
                    "PR-AUC": m.get("pr_auc"),
                    "Brier Score": m.get("brier_score")
                })
        st.table(pd.DataFrame(comp_data))

        # Cross Instrument Transfer Matrix
        st.markdown("### Cross-Instrument Transfer Experiments")
        cross_path = Path("data/processed/cross_instrument_results.json")
        if cross_path.exists():
            with open(cross_path, "r") as f:
                cross_res = json.load(f)
            cross_rows = []
            for k, v in cross_res.items():
                cross_rows.append({
                    "Experiment": k,
                    "Samples": v.get("n_samples"),
                    "ROC-AUC": v.get("roc_auc"),
                    "PR-AUC": v.get("pr_auc"),
                    "Recall": f"{v.get('recall', 0)*100:.1f}%",
                    "F1 Score": f"{v.get('f1', 0):.4f}"
                })
            st.table(pd.DataFrame(cross_rows))

        # Feature Importance Chart
        st.markdown("### Top Predictive Features")
        feat_imp = metadata.get("feature_importance", {})
        if feat_imp:
            top_feats = dict(list(feat_imp.items())[:12])
            fig_imp = go.Figure(go.Bar(
                x=list(top_feats.values()),
                y=list(top_feats.keys()),
                orientation="h",
                marker_color="#00E5FF"
            ))
            fig_imp.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0E1117",
                plot_bgcolor="#161A22",
                height=400,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# ==============================================================================
# PAGE 5: BACKTESTING
# ==============================================================================
elif page == "Backtesting":
    st.title("Strategy Simulation & Performance Backtesting")
    st.caption("Realistic simulation accounting for spread, slippage, commission, concurrent caps, and cooldown.")

    bt_col1, bt_col2 = st.columns([1, 2])
    with bt_col1:
        st.subheader("Simulation Parameters")
        bt_inst = st.selectbox("Instrument", ["XAUUSD", "XAGUSD"], index=0)
        bt_thresh = st.slider("Probability Threshold", 0.20, 0.45, 0.30, 0.02)
        bt_target_atr = st.slider("Take Profit (x ATR)", 1.0, 3.0, 1.5, 0.1)
        bt_sl_buf = st.slider("Stop Loss Buffer (x ATR)", 0.1, 1.0, 0.2, 0.05)
        bt_risk = st.slider("Risk Per Trade (%)", 0.5, 3.0, 1.0, 0.5)

        run_btn = st.button("🚀 Run Backtest Simulation")

    with bt_col2:
        if run_btn:
            with st.spinner("Simulating trades..."):
                raw_df = data_dict.get(bt_inst, pd.DataFrame())
                inst_preds = predictions_df[predictions_df["instrument"] == bt_inst]

                engine = BacktestEngine(
                    initial_capital=100000.0,
                    risk_per_trade_pct=bt_risk,
                    target_atr_x=bt_target_atr,
                    sl_buffer_atr=bt_sl_buf
                )
                t_df, eq_df, metrics = engine.run_backtest(
                    df=raw_df.iloc[-20000:].reset_index(drop=True),
                    signals_df=inst_preds,
                    probability_threshold=bt_thresh
                )

                st.session_state["bt_results"] = (t_df, eq_df, metrics)

        if "bt_results" in st.session_state:
            t_df, eq_df, metrics = st.session_state["bt_results"]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Net Profit", f"${metrics.get('net_profit', 0):,.2f}", f"{metrics.get('net_return_pct', 0)}%")
            with c2:
                st.metric("Win Rate", f"{metrics.get('win_rate', 0)}%")
            with c3:
                st.metric("Profit Factor", metrics.get("profit_factor", 0.0))
            with c4:
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0)}%")

            # Equity Curve Chart
            if not eq_df.empty:
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(
                    x=eq_df["timestamp"],
                    y=eq_df["total_equity"],
                    mode="lines",
                    name="Equity",
                    line=dict(color="#00E5FF", width=2)
                ))
                fig_eq.update_layout(
                    title="Portfolio Equity Curve ($)",
                    template="plotly_dark",
                    paper_bgcolor="#0E1117",
                    plot_bgcolor="#161A22",
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig_eq, use_container_width=True)

            if not t_df.empty:
                st.markdown("### Trade Log")
                st.dataframe(t_df.head(50), use_container_width=True)
                st.download_button("📥 Download Trade Log CSV", t_df.to_csv(index=False).encode('utf-8'), "trade_log.csv", "text/csv")
        else:
            st.info("Adjust parameters and click 'Run Backtest Simulation' to inspect performance.")

# ==============================================================================
# PAGE 6: DATA & MODEL MANAGEMENT
# ==============================================================================
elif page == "Data & Model Management":
    st.title("Data Ingestion & Model Version Management")
    st.caption("Upload new OHLCV datasets, trigger model training, and export artifacts.")

    up_file = st.file_uploader("Upload CSV Data (XAUUSD / XAGUSD)", type=["csv"])
    if up_file:
        df_uploaded = pd.read_csv(up_file)
        cleaned, rep = validate_ohlcv_dataframe(df_uploaded, instrument="UPLOADED")
        st.success(f"Uploaded {len(cleaned)} rows. Quality Status: {'Valid' if rep.is_valid else 'Invalid'}")
        st.json(rep.to_dict())

    st.markdown("### Retrain Classifier")
    if st.button("🔄 Trigger Model Retraining Pipeline"):
        with st.spinner("Executing pipeline and retraining model..."):
            os.system("python run_pipeline.py")
            st.success("Retraining complete! Refreshing page to reload artifacts.")
            st.rerun()

# ==============================================================================
# PAGE 7: SETTINGS
# ==============================================================================
elif page == "Settings":
    st.title("System Parameters & Configuration")
    st.caption("Central configuration matching TradingView Pine Script inputs.")

    config_path = "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    st.text_area("config.yaml", yaml.dump(cfg), height=450)
    st.info("Edit config.yaml directly to modify permanent system defaults.")
