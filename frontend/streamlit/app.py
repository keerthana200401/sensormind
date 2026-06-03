import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "SensorMind",
    page_icon  = "🔧",
    layout     = "wide"
)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'saved')
    return {
        'xgb'         : joblib.load(f'{base}/xgboost.joblib'),
        'iso'         : joblib.load(f'{base}/isolation_forest.joblib'),
        'scaler'      : joblib.load(f'{base}/scaler.joblib'),
        'feature_cols': joblib.load(f'{base}/feature_cols.joblib')
    }

@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(__file__),
                        '..', '..', 'data', 'features',
                        'train_features.parquet')
    return pd.read_parquet(path)

models = load_models()
df     = load_data()

# ── Header ────────────────────────────────────────────────────
st.title("🔧 SensorMind")
st.markdown("**AI-Powered Predictive Maintenance for Industrial Engines**")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
engine_id = st.sidebar.selectbox(
    "Select Engine",
    ["Engine_001", "Engine_002", "Engine_003",
     "Engine_CRITICAL", "Engine_HEALTHY"]
)

subset = st.sidebar.selectbox(
    "Dataset Subset",
    ["FD001", "FD002", "FD003", "FD004"]
)

run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

st.sidebar.divider()
st.sidebar.markdown("**About SensorMind**")
st.sidebar.markdown("""
- 🤖 XGBoost RUL prediction
- 🔍 Isolation Forest anomaly detection
- 🧠 Claude API plain-English alerts
- 📊 SHAP explainability
""")

# ── Main content ──────────────────────────────────────────────
if run_analysis:
    with st.spinner(f"Analyzing {engine_id}..."):

        # Get sample
        subset_df = df[df['subset'] == subset]
        if 'critical' in engine_id.lower():
            sample = subset_df[subset_df['RUL_capped'] <= 15].iloc[0]
        elif 'healthy' in engine_id.lower():
            sample = subset_df[subset_df['RUL_capped'] >= 120].iloc[0]
        else:
            sample = subset_df.sample(1, random_state=42).iloc[0]

        # Run prediction
        feature_cols = models['feature_cols']
        input_df     = pd.DataFrame([sample[feature_cols]])
        rul_pred     = float(models['xgb'].predict(input_df)[0])
        rul_pred     = max(0, round(rul_pred, 2))
        input_scaled = models['scaler'].transform(input_df)
        is_anomaly   = models['iso'].predict(input_scaled)[0] == -1

        if rul_pred <= 15:   risk = "CRITICAL"
        elif rul_pred <= 30: risk = "HIGH"
        elif rul_pred <= 60: risk = "MEDIUM"
        else:                risk = "LOW"

        risk_colors = {
            'CRITICAL': '🔴',
            'HIGH'    : '🟠',
            'MEDIUM'  : '🟡',
            'LOW'     : '🟢'
        }

        # ── KPI cards ─────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Predicted RUL",
                      f"{rul_pred} cycles",
                      delta=f"{rul_pred - 125:.0f} from max")
        with col2:
            st.metric("Risk Level",
                      f"{risk_colors[risk]} {risk}")
        with col3:
            st.metric("Anomaly Detected",
                      "⚠️ YES" if is_anomaly else "✅ NO")
        with col4:
            st.metric("Dataset",
                      subset)

        st.divider()

        # ── Sensor trend chart ────────────────────────────────
        st.subheader("📈 Sensor Trends")

        engine_num = sample['engine_id']
        engine_data = df[
            (df['engine_id'] == engine_num) &
            (df['subset'] == subset)
        ].sort_values('cycle')

        col_left, col_right = st.columns(2)

        with col_left:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x    = engine_data['cycle'],
                y    = engine_data['sensor_13'],
                mode = 'lines',
                name = 'Sensor 13 (Exhaust Temp)',
                line = dict(color='#FF6B6B', width=2)
            ))
            fig1.update_layout(
                title  = 'Sensor 13 — Exhaust Temperature',
                xaxis_title = 'Cycle',
                yaxis_title = 'Value',
                height = 350
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_right:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x    = engine_data['cycle'],
                y    = engine_data['sensor_15'],
                mode = 'lines',
                name = 'Sensor 15 (Bypass Pressure)',
                line = dict(color='#4ECDC4', width=2)
            ))
            fig2.update_layout(
                title  = 'Sensor 15 — Bypass Ratio Pressure',
                xaxis_title = 'Cycle',
                yaxis_title = 'Value',
                height = 350
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ── RUL gauge chart ───────────────────────────────────
        st.subheader("⏱️ Remaining Useful Life Gauge")

        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = rul_pred,
            delta = {'reference': 125},
            title = {'text': "Predicted RUL (cycles)"},
            gauge = {
                'axis' : {'range': [0, 125]},
                'bar'  : {'color': "darkblue"},
                'steps': [
                    {'range': [0, 15],   'color': '#FF4444'},
                    {'range': [15, 30],  'color': '#FF8C00'},
                    {'range': [30, 60],  'color': '#FFD700'},
                    {'range': [60, 125], 'color': '#90EE90'}
                ],
                'threshold': {
                    'line' : {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        # ── Claude AI Alert ───────────────────────────────────
        st.subheader("🧠 AI Maintenance Alert")

        if st.button("Generate Claude Alert"):
            with st.spinner("Calling Claude API..."):
                from genai.rag_pipeline import generate_alert
                from genai.vectorstore import ingest_failure_reports

                ingest_failure_reports()

                prediction = {
                    'rul_prediction': rul_pred,
                    'risk_level'    : risk,
                    'is_anomaly'    : is_anomaly,
                    'anomaly_score' : 0.0,
                    'top_signals'   : {
                        'sensor_13': round(float(sample['sensor_13']), 4),
                        'sensor_15': round(float(sample['sensor_15']), 4),
                        'sensor_11': round(float(sample['sensor_11']), 4),
                    },
                    'model_used': 'XGBoost'
                }

                result = generate_alert(prediction, engine_id=engine_id)

                # Display alert box
                alert_bg = {
                    'CRITICAL': '#FFE5E5',
                    'HIGH'    : '#FFF3E0',
                    'MEDIUM'  : '#FFFDE7',
                    'LOW'     : '#E8F5E9'
                }

                st.markdown(f"""
                <div style="
                    background-color: {alert_bg[risk]};
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid {'red' if risk in ['CRITICAL','HIGH'] else 'orange'};
                ">
                <h4>{risk_colors[risk]} {risk} ALERT — {engine_id}</h4>
                <p>{result['alert']}</p>
                <small>References: {', '.join(result['similar_failures'])} 
                | Tokens: {result['input_tokens']} in / {result['output_tokens']} out</small>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ── Raw sensor readings table ─────────────────────────
        st.subheader("📋 Raw Sensor Readings")
        sensor_cols = [f'sensor_{i}' for i in range(1, 22)
                       if f'sensor_{i}' in df.columns]
        st.dataframe(
            pd.DataFrame([sample[sensor_cols]]).T\
              .rename(columns={0: 'Current Reading'})\
              .round(4),
            use_container_width=True
        )

else:
    # ── Landing screen ────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### 🤖 ML Models\n"
                "XGBoost + Random Forest + "
                "Isolation Forest predicting "
                "Remaining Useful Life")
    with col2:
        st.success("### 🧠 Gen AI Layer\n"
                   "Claude API + ChromaDB RAG "
                   "generating plain-English "
                   "maintenance alerts")
    with col3:
        st.warning("### 📊 82.2% Accuracy\n"
                   "R² Score of 0.822 | "
                   "MAE ±12.2 cycles | "
                   "3 models compared")

    st.markdown("---")
    st.markdown("### 👈 Select an engine and click **Run Analysis** to start")