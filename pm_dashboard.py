import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import joblib
import shap

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Asset Maintenance",
    page_icon="🔧",
    layout="wide"
)

# ── Load data & models ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    train    = pd.read_csv('data/train_featured.csv')
    test     = pd.read_csv('data/test_featured.csv')
    preds    = pd.read_csv('data/pm_predictions.csv')
    metrics  = pd.read_csv('data/pm_metrics.csv')
    shap_df  = pd.read_csv('data/pm_shap_values.csv')
    features = pd.read_csv('data/final_features.csv', header=None)[0].tolist()
    return train, test, preds, metrics, shap_df, features

@st.cache_resource
def load_models():
    rf     = joblib.load('data/model_rf.pkl')
    xgb    = joblib.load('data/model_xgb.pkl')
    scaler = joblib.load('data/scaler.pkl')
    return rf, xgb, scaler

train, test, preds, metrics, shap_df, features = load_data()
rf_model, xgb_model, scaler = load_models()

sensor_cols  = [f for f in features if f.startswith('s_') and 'rolling' not in f]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Controls")
selected_model = st.sidebar.selectbox("Model", ["XGBoost", "Random Forest", "Neural Network"])
failure_window = st.sidebar.slider("Failure window (cycles)", 10, 60, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** NASA CMAPSS FD001  \n**Engines:** 100 train / 100 test  \n**Sensors:** 21 (14 active)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Predictive Asset Maintenance Dashboard")
st.markdown("Predicting industrial engine failure using **Random Forest**, **XGBoost**, and a **Neural Network** — NASA CMAPSS turbofan dataset")
st.markdown("---")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.subheader("Model Performance — Test Set")

col1, col2, col3, col4, col5, col6 = st.columns(6)

def get_metric(metrics, model, metric):
    row = metrics[metrics['model'] == model]
    return round(row[metric].values[0], 4) if len(row) else "N/A"

col1.metric("RF AUROC",  get_metric(metrics, 'Random Forest',  'auroc'))
col2.metric("RF F1",     get_metric(metrics, 'Random Forest',  'f1'))
col3.metric("XGB AUROC", get_metric(metrics, 'XGBoost',        'auroc'))
col4.metric("XGB F1",    get_metric(metrics, 'XGBoost',        'f1'))
col5.metric("NN AUROC",  get_metric(metrics, 'Neural Network', 'auroc'))
col6.metric("NN F1",     get_metric(metrics, 'Neural Network', 'f1'))

st.markdown("---")

# ── Model Comparison ──────────────────────────────────────────────────────────
st.subheader("Model Comparison")

col_left, col_right = st.columns(2)

with col_left:
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    ax.bar(metrics['model'], metrics['auroc'], color=colors, alpha=0.85, edgecolor='white')
    ax.set_title('AUROC (higher is better)')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.5, 1.0)
    for i, v in enumerate(metrics['auroc']):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_right:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(metrics['model'], metrics['f1'], color=colors, alpha=0.85, edgecolor='white')
    ax.set_title('F1 Score (higher is better)')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(metrics['f1']):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Per-Engine Risk View ──────────────────────────────────────────────────────
st.subheader("Per-Engine Risk Assessment")

model_prob_col = {'XGBoost': 'xgb_prob', 'Random Forest': 'rf_prob', 'Neural Network': 'nn_prob'}
prob_col = model_prob_col[selected_model]

fig, ax = plt.subplots(figsize=(14, 4))
colors_engine = ['#e74c3c' if p >= 0.5 else '#2ecc71' for p in preds[prob_col]]
ax.bar(preds['engine_id'], preds[prob_col], color=colors_engine, alpha=0.85, edgecolor='white', linewidth=0.3)
ax.axhline(0.5, color='black', linestyle='--', linewidth=1, label='Decision threshold (0.5)')
ax.set_xlabel('Engine ID')
ax.set_ylabel('Failure Probability')
ax.set_title(f'{selected_model} — Predicted Failure Probability per Engine')
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.caption("Red = predicted near failure | Green = predicted healthy")

st.markdown("---")

# ── SHAP Explainability ───────────────────────────────────────────────────────
st.subheader("SHAP Feature Importance — XGBoost")
st.markdown("SHAP values show which sensor readings drove predictions the most. Higher mean |SHAP| = more important feature.")

mean_shap = shap_df.abs().mean().sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
mean_shap.sort_values().plot(kind='barh', ax=ax, color='#e74c3c', alpha=0.85, edgecolor='white')
ax.set_title('Top 15 Features by Mean |SHAP Value|')
ax.set_xlabel('Mean |SHAP Value|')
plt.tight_layout()
plt.savefig('data/pm_shap_dashboard.png', dpi=150, bbox_inches='tight')
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Sensor Explorer ───────────────────────────────────────────────────────────
st.subheader("Sensor Degradation Explorer")
st.markdown("Select an engine and sensor to see how readings change over its lifetime.")

col1, col2 = st.columns(2)
with col1:
    engine_id = st.selectbox("Engine ID", sorted(train['engine_id'].unique()))
with col2:
    sensor = st.selectbox("Sensor", sensor_cols)

engine_data = train[train['engine_id'] == engine_id].sort_values('cycle')
max_cycle   = engine_data['cycle'].max()
fail_start  = max_cycle - failure_window

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(engine_data['cycle'], engine_data[sensor],
        color='#3498db', linewidth=1.5, label='Raw reading')
ax.plot(engine_data['cycle'], engine_data[f'{sensor}_rolling'],
        color='#e74c3c', linewidth=2, linestyle='--', label='Rolling mean')
ax.axvline(fail_start, color='black', linestyle=':', linewidth=1.5,
           label=f'Failure window starts (cycle {int(fail_start)})')
ax.axvspan(fail_start, max_cycle, alpha=0.08, color='red')
ax.set_xlabel('Cycle')
ax.set_ylabel('Sensor Value')
ax.set_title(f'Engine {engine_id} — {sensor} over lifetime')
ax.legend()
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Class Distribution ────────────────────────────────────────────────────────
st.subheader("Training Data — Class Balance")

col1, col2 = st.columns(2)
counts = train['label'].value_counts()

with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Healthy (0)', 'Near Failure (1)'], counts.values,
           color=['#2ecc71', '#e74c3c'], alpha=0.85, edgecolor='white')
    ax.set_title('Class Distribution')
    ax.set_ylabel('Count')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha='center')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("### Why F1 over Accuracy?")
    st.markdown("""
The dataset is **imbalanced** — most cycles are healthy, only ~25% are near failure.

A naive model predicting everything as healthy would get high **accuracy** but be completely useless — it would miss every real failure.

**F1-score** balances precision and recall, penalising models that miss failures. This is why F1 is the right metric for predictive maintenance tasks.

**AUROC** measures the model's ability to rank near-failure engines above healthy ones regardless of the threshold chosen.
    """)

st.markdown("---")
st.caption("Data: NASA CMAPSS FD001 | Models: Random Forest, XGBoost, Neural Network (sklearn + xgboost) | Explainability: SHAP | Built for Shell Internship 2026")
