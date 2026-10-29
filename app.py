"""
Credit Card Fraud Detection — Interactive Prototype
====================================================
A Streamlit dashboard built on ensemble ML models trained on the
ULB Credit Card Fraud dataset.

Models: Random Forest, XGBoost, Logistic Regression,
        Voting Classifier (soft), Stacking Classifier.

Run:  streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ───────────────────────────── paths ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "saved_artifacts")

# ───────────────────────────── page config ───────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────── custom CSS ────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .metric-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; }
    .fraud-alert {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255,65,108,0.3);
    }
    .legit-alert {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(56,239,125,0.3);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────── loaders ───────────────────────────
@st.cache_resource
def load_models():
    """Load all trained models from disk."""
    model_names = [
        "random_forest",
        "xgboost",
        "logistic_regression",
        "voting_classifier",
        "stacking_classifier",
    ]
    models = {}
    for name in model_names:
        path = os.path.join(ARTIFACT_DIR, f"{name}.pkl")
        if os.path.exists(path):
            display_name = name.replace("_", " ").title()
            models[display_name] = joblib.load(path)
    return models


@st.cache_data
def load_metrics():
    path = os.path.join(ARTIFACT_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_dataset_stats():
    path = os.path.join(ARTIFACT_DIR, "dataset_stats.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_demo_data():
    path = os.path.join(ARTIFACT_DIR, "demo_transactions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_feature_names():
    path = os.path.join(ARTIFACT_DIR, "feature_names.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


# ───────────────────────────── check artifacts ───────────────────
def artifacts_exist():
    return os.path.exists(ARTIFACT_DIR) and os.path.exists(
        os.path.join(ARTIFACT_DIR, "metrics.json")
    )


# ───────────────────────────── sidebar ───────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/shield.png", width=60)
    st.title("🛡️ Fraud Detection")
    st.caption("Ensemble ML Prototype")
    st.divider()

    if not artifacts_exist():
        st.error("⚠️ Models not trained yet!")
        st.info("Run `python train_models.py` first to train models and generate artifacts.")
        st.stop()

    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Model Comparison",
            "🔍 Live Prediction",
            "📋 Batch Analysis",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Built with Streamlit · Ensemble ML")
    st.caption("Dataset: ULB Credit Card Fraud")


# ───────────────────────────── load data ─────────────────────────
models = load_models()
metrics = load_metrics()
stats = load_dataset_stats()
demo_df = load_demo_data()
feature_names = load_feature_names()


# ═══════════════════════════════════════════════════════════════
#  PAGE: Overview
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🛡️ Credit Card Fraud Detection")
    st.markdown("**Ensemble Machine Learning Prototype** — Detecting fraudulent credit card transactions using multiple ML models.")

    st.divider()

    # ── Dataset Stats ──
    if stats:
        st.subheader("📊 Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Transactions", f"{stats.get('total_transactions', 0):,}")
        with c2:
            st.metric("Legitimate", f"{stats.get('total_legitimate', 0):,}")
        with c3:
            st.metric("Fraudulent", f"{stats.get('total_fraud', 0):,}")
        with c4:
            st.metric("Fraud Rate", f"{stats.get('fraud_percentage', 0):.3f}%")

        st.divider()

        # Class distribution chart
        col_a, col_b = st.columns(2)
        with col_a:
            fig_dist = go.Figure(data=[go.Pie(
                labels=["Legitimate", "Fraud"],
                values=[stats.get("total_legitimate", 0), stats.get("total_fraud", 0)],
                hole=0.55,
                marker=dict(colors=["#38ef7d", "#ff416c"]),
                textinfo="label+percent",
                textfont_size=14,
            )])
            fig_dist.update_layout(
                title="Class Distribution",
                height=350,
                margin=dict(t=50, b=30, l=30, r=30),
                showlegend=False,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_b:
            fig_split = go.Figure(data=[
                go.Bar(name="Train", x=["Legitimate", "Fraud"],
                       y=[stats.get("train_samples", 0) - stats.get("train_fraud", 0),
                          stats.get("train_fraud", 0)],
                       marker_color=["#667eea", "#667eea"]),
                go.Bar(name="Test", x=["Legitimate", "Fraud"],
                       y=[stats.get("test_samples", 0) - stats.get("test_fraud", 0),
                          stats.get("test_fraud", 0)],
                       marker_color=["#764ba2", "#764ba2"]),
            ])
            fig_split.update_layout(
                barmode="group",
                title="Train/Test Split",
                height=350,
                margin=dict(t=50, b=30, l=30, r=30),
            )
            st.plotly_chart(fig_split, use_container_width=True)

    st.divider()

    # ── Pipeline Overview ──
    st.subheader("🔧 Pipeline Architecture")

    col1, col2, col3, col4, col5 = st.columns(5)
    steps = [
        ("📥", "Data Loading", "284,807 txns\n31 features"),
        ("⚙️", "Preprocessing", "StandardScaler\non Amount & Time"),
        ("🔄", "SMOTE", "Balanced classes\n227K each"),
        ("🤖", "Model Training", "5 models\n(3 base + 2 ensemble)"),
        ("📈", "Evaluation", "Precision, Recall\nF1, AUPRC"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4, col5], steps):
        with col:
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.divider()

    # ── Model summary table ──
    st.subheader("🏆 Model Performance Summary")
    if metrics:
        summary_data = []
        for name, m in metrics.items():
            summary_data.append({
                "Model": name,
                "Accuracy": f"{m['accuracy']:.4f}",
                "Precision (Fraud)": f"{m['precision']:.4f}",
                "Recall (Fraud)": f"{m['recall']:.4f}",
                "F1 (Fraud)": f"{m['f1']:.4f}",
                "AUPRC": f"{m['auprc']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "Train Time (s)": f"{m['training_time']:.2f}",
            })
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Highlight best model
        best_f1 = max(metrics.items(), key=lambda x: x[1]["f1"])
        st.success(f"🏆 **Best model by F1-score**: {best_f1[0]} (F1 = {best_f1[1]['f1']:.4f})")


# ═══════════════════════════════════════════════════════════════
#  PAGE: Model Comparison
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("Compare all ensemble models across multiple evaluation metrics.")

    if not metrics:
        st.warning("No metrics found. Train models first.")
        st.stop()

    # ── Metric bar charts ──
    tab1, tab2, tab3 = st.tabs(["📊 Metrics Comparison", "🔲 Confusion Matrices", "📈 Performance Radar"])

    with tab1:
        model_names = list(metrics.keys())
        metric_keys = ["precision", "recall", "f1", "auprc", "roc_auc"]
        metric_labels = ["Precision", "Recall", "F1-Score", "AUPRC", "ROC-AUC"]

        fig_bars = make_subplots(rows=1, cols=len(metric_keys),
                                subplot_titles=metric_labels, shared_yaxes=True)

        colors = px.colors.qualitative.Vivid[:len(model_names)]

        for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            values = [metrics[m][key] for m in model_names]
            fig_bars.add_trace(
                go.Bar(
                    x=model_names, y=values,
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in values],
                    textposition="outside",
                    showlegend=False,
                ),
                row=1, col=i + 1,
            )

        fig_bars.update_layout(
            height=400,
            margin=dict(t=60, b=80, l=30, r=30),
            yaxis=dict(range=[0, 1.1]),
        )
        fig_bars.update_xaxes(tickangle=45, tickfont_size=9)
        st.plotly_chart(fig_bars, use_container_width=True)

    with tab2:
        st.subheader("Confusion Matrices")
        cols = st.columns(min(3, len(metrics)))
        for idx, (name, m) in enumerate(metrics.items()):
            col = cols[idx % len(cols)]
            with col:
                cm = np.array(m["confusion_matrix"])
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm, x=["Legit", "Fraud"], y=["Legit", "Fraud"],
                    text=[[str(v) for v in row] for row in cm],
                    texttemplate="%{text}",
                    colorscale="Blues",
                    showscale=False,
                ))
                fig_cm.update_layout(
                    title=dict(text=name, font_size=13),
                    xaxis_title="Predicted",
                    yaxis_title="Actual",
                    height=300,
                    margin=dict(t=50, b=50, l=60, r=30),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

    with tab3:
        st.subheader("Performance Radar Chart")
        radar_metrics = ["precision", "recall", "f1", "auprc", "roc_auc"]
        radar_labels = ["Precision", "Recall", "F1", "AUPRC", "ROC-AUC"]

        fig_radar = go.Figure()
        colors_radar = px.colors.qualitative.Vivid
        for idx, (name, m) in enumerate(metrics.items()):
            values = [m[k] for k in radar_metrics]
            values.append(values[0])  # close the polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                name=name,
                line_color=colors_radar[idx % len(colors_radar)],
                opacity=0.6,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Training time comparison ──
    st.divider()
    st.subheader("⏱️ Training Time Comparison")
    times = {name: m["training_time"] for name, m in metrics.items()}
    fig_time = go.Figure(data=[go.Bar(
        x=list(times.keys()),
        y=list(times.values()),
        marker_color=px.colors.qualitative.Vivid[:len(times)],
        text=[f"{t:.2f}s" for t in times.values()],
        textposition="outside",
    )])
    fig_time.update_layout(
        yaxis_title="Seconds",
        height=350,
        margin=dict(t=30, b=80),
    )
    fig_time.update_xaxes(tickangle=30)
    st.plotly_chart(fig_time, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: Live Prediction
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Live Prediction":
    st.title("🔍 Live Transaction Prediction")
    st.markdown("Select a model, input transaction features, and get a real-time fraud prediction.")

    if not models:
        st.warning("No models loaded.")
        st.stop()

    # Model selector
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[selected_model_name]

    st.divider()

    input_mode = st.radio("Input Mode", ["🎲 Random Demo Transaction", "✏️ Manual Input"], horizontal=True)

    if input_mode == "🎲 Random Demo Transaction":
        if demo_df is not None:
            if st.button("🎲 Load Random Transaction", type="primary"):
                row = demo_df.sample(1)
                st.session_state["demo_row"] = row

            if "demo_row" in st.session_state:
                row = st.session_state["demo_row"]
                actual = int(row["Actual_Class"].values[0])
                features = row.drop("Actual_Class", axis=1)

                st.subheader("Transaction Features")
                st.dataframe(features, use_container_width=True, hide_index=True)

                # Predict
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0]

                col_pred, col_actual = st.columns(2)
                with col_pred:
                    if prediction == 1:
                        st.markdown(f'<div class="fraud-alert">🚨 FRAUD DETECTED — Confidence: {probability[1]:.1%}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="legit-alert">✅ LEGITIMATE — Confidence: {probability[0]:.1%}</div>',
                                    unsafe_allow_html=True)

                with col_actual:
                    if actual == 1:
                        st.markdown('<div class="fraud-alert">Actual: 🚨 FRAUD</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="legit-alert">Actual: ✅ LEGITIMATE</div>', unsafe_allow_html=True)

                st.divider()

                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability[1] * 100,
                    title={"text": "Fraud Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#ff416c"},
                        "steps": [
                            {"range": [0, 30], "color": "#e8f5e9"},
                            {"range": [30, 70], "color": "#fff3e0"},
                            {"range": [70, 100], "color": "#ffebee"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                fig_gauge.update_layout(height=300, margin=dict(t=50, b=30))
                st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("Demo data not found. Run `train_models.py` first.")

    else:  # Manual Input
        st.info("Enter the 30 feature values (V1–V28, Amount_scaled, Time_scaled):")
        cols = st.columns(5)
        input_values = []
        for i, fname in enumerate(feature_names):
            with cols[i % 5]:
                val = st.number_input(fname, value=0.0, format="%.6f", key=f"input_{fname}")
                input_values.append(val)

        if st.button("🔍 Predict", type="primary"):
            input_df = pd.DataFrame([input_values], columns=feature_names)
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]

            if prediction == 1:
                st.markdown(f'<div class="fraud-alert">🚨 FRAUD DETECTED — Confidence: {probability[1]:.1%}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="legit-alert">✅ LEGITIMATE — Confidence: {probability[0]:.1%}</div>',
                            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: Batch Analysis
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Batch Analysis":
    st.title("📋 Batch Transaction Analysis")
    st.markdown("Analyze multiple transactions at once using the demo dataset or an uploaded CSV.")

    if not models:
        st.warning("No models loaded.")
        st.stop()

    selected_model_name = st.selectbox("Select Model", list(models.keys()), key="batch_model")
    model = models[selected_model_name]

    st.divider()

    data_source = st.radio("Data Source", ["📦 Demo Dataset", "📤 Upload CSV"], horizontal=True)

    if data_source == "📦 Demo Dataset":
        if demo_df is not None:
            batch_df = demo_df.copy()
        else:
            st.info("Demo data not found.")
            st.stop()
    else:
        uploaded = st.file_uploader("Upload CSV (must have same features as training data)", type="csv")
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
        else:
            st.info("Upload a CSV file to begin.")
            st.stop()

    if st.button("🚀 Run Batch Analysis", type="primary"):
        has_actual = "Actual_Class" in batch_df.columns
        if has_actual:
            actual = batch_df["Actual_Class"].values
            features_df = batch_df.drop("Actual_Class", axis=1)
        else:
            actual = None
            features_df = batch_df

        predictions = model.predict(features_df)
        probabilities = model.predict_proba(features_df)[:, 1]

        results_df = features_df.copy()
        results_df["Prediction"] = ["🚨 Fraud" if p == 1 else "✅ Legit" for p in predictions]
        results_df["Fraud_Probability"] = probabilities
        if has_actual:
            results_df["Actual"] = ["🚨 Fraud" if a == 1 else "✅ Legit" for a in actual]

        # Summary
        fraud_count = (predictions == 1).sum()
        legit_count = (predictions == 0).sum()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(predictions))
        with col2:
            st.metric("Predicted Fraud", fraud_count)
        with col3:
            st.metric("Predicted Legitimate", legit_count)

        if has_actual:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            st.divider()
            st.subheader("📈 Batch Evaluation Metrics")
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("Accuracy", f"{accuracy_score(actual, predictions):.4f}")
            with mc2:
                st.metric("Precision", f"{precision_score(actual, predictions, zero_division=0):.4f}")
            with mc3:
                st.metric("Recall", f"{recall_score(actual, predictions, zero_division=0):.4f}")
            with mc4:
                st.metric("F1-Score", f"{f1_score(actual, predictions, zero_division=0):.4f}")

        st.divider()

        # Distribution chart
        col_a, col_b = st.columns(2)
        with col_a:
            fig_pred_dist = go.Figure(data=[go.Pie(
                labels=["Legitimate", "Fraud"],
                values=[legit_count, fraud_count],
                hole=0.5,
                marker=dict(colors=["#38ef7d", "#ff416c"]),
            )])
            fig_pred_dist.update_layout(
                title="Prediction Distribution",
                height=300,
                margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig_pred_dist, use_container_width=True)

        with col_b:
            fig_prob_hist = go.Figure(data=[go.Histogram(
                x=probabilities,
                nbinsx=30,
                marker_color="#667eea",
            )])
            fig_prob_hist.update_layout(
                title="Fraud Probability Distribution",
                xaxis_title="Probability",
                yaxis_title="Count",
                height=300,
                margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig_prob_hist, use_container_width=True)

        # Results table
        st.subheader("📋 Detailed Results")
        display_cols = ["Prediction", "Fraud_Probability"]
        if has_actual:
            display_cols.append("Actual")
        st.dataframe(
            results_df[display_cols + [c for c in results_df.columns if c not in display_cols]],
            use_container_width=True,
            hide_index=True,
            height=400,
        )

        # Download
        csv_out = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV",
            csv_out,
            "fraud_detection_results.csv",
            "text/csv",
        )
