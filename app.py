"""
Smart Warehouse Demand Prediction — LPG Cylinders
Streamlit App | Mitra Bharatgas Agency, Murshidabad
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, io, os, joblib
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LPG Demand Predictor", page_icon="🔵",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,[class*="css"]{font-family:'Syne',sans-serif;}
[data-testid="stSidebar"]{background:#0a1628 !important;border-right:1px solid #1a2e4a;}
.hero{background:linear-gradient(135deg,#0d1b2a 0%,#1a2e4a 60%,#0d1b2a 100%);
  border:1px solid #1e3a5f;border-radius:14px;padding:2rem 2.5rem;margin-bottom:1.5rem;}
.hero-title{font-size:1.9rem;font-weight:800;color:#e8f4fd;margin:0;letter-spacing:-.5px;}
.hero-sub{font-size:.9rem;color:#6b8faf;margin-top:.4rem;font-family:'IBM Plex Mono',monospace;}
.hero-badge{display:inline-block;background:rgba(30,90,160,.25);border:1px solid #1e5aa0;
  color:#5ba3d9;padding:.15rem .7rem;border-radius:20px;font-size:.72rem;
  font-family:'IBM Plex Mono',monospace;margin-top:.8rem;}
.kpi{background:#0f1e30;border:1px solid #1a2e4a;border-radius:10px;padding:1rem 1.2rem;margin:.3rem 0;}
.kpi-label{font-size:.7rem;color:#5b7a99;text-transform:uppercase;letter-spacing:1px;}
.kpi-value{font-size:1.7rem;font-weight:700;color:#e8f4fd;margin:.15rem 0;}
.kpi-sub{font-size:.72rem;color:#3d7ab5;}
.sec{background:#0f1e30;border-left:4px solid #1e5aa0;border-radius:0 8px 8px 0;
  padding:.7rem 1rem;margin:1.2rem 0 .8rem 0;}
.sec h3{margin:0;color:#c8dcf0;font-size:1rem;font-weight:700;}
.sec p{margin:.15rem 0 0 0;color:#5b7a99;font-size:.78rem;}
.stTabs [data-baseweb="tab-list"]{background:#0f1e30;border-radius:8px;gap:3px;padding:3px;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#5b7a99;border-radius:6px;font-weight:600;font-size:.82rem;}
.stTabs [aria-selected="true"]{background:#1e3a5f !important;color:#5ba3d9 !important;}
div[data-testid="stSelectbox"] label,div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label{color:#8aacc8 !important;font-size:.83rem;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_FILE   = os.path.join(BASE_DIR, "Warehouse_Demand_5000_1.xlsx")

NUMERIC_COLS = ['Year','Month_Number','Is_Winter','Is_Festival_Month','Days_in_Month',
    'No_of_Family_Members','No_of_Adults','No_of_Children','Monthly_Income (₹)',
    'LPG_Price_per_Cylinder (₹)','Zone_Opening_Stock','Zone_Cylinders_Ordered',
    'Lead_Time_Days','Zone_Cylinders_Delivered','Zone_Safety_Stock','Zone_Reorder_Point',
    'Gas_Consumption_kg','Actual_Demand (cylinders)','Fulfilled_Demand (cylinders)',
    'Units_Short','Damaged_Cylinders','Zone_Closing_Stock','Avg_Daily_Demand']
DROP_COLS   = ['Record_ID','Family_ID','Month','Season','Festival','Stockout_Occurred','Units_Short']
EXCLUDE_TGT = ['Gas_Consumption_kg','Stockout_Risk','Actual_Demand (cylinders)',
                'Fulfilled_Demand (cylinders)','Avg_Daily_Demand',
                'Zone_Closing_Stock','Zone_Safety_Stock','Zone_Reorder_Point']
ZONES       = ['Raghunathganj','Kandi','Samserganj','Berhampore','Bali',
               'Domkal','Farakka','Suti','Lalbagh','Jangipur']
ZONE_ENC    = {z: i for i, z in enumerate(sorted(ZONES))}
SUBSIDY_ENC = {'Non-Subsidized': 0, 'PMUY': 1}
COLORS      = ['#1a5276','#1e8449','#b7950b','#c0392b','#7d3c98','#117a65']
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.rcParams.update({
    'font.family':'DejaVu Sans','figure.dpi':120,
    'figure.facecolor':'#0f1e30','axes.facecolor':'#0f1e30',
    'axes.labelcolor':'#8aacc8','xtick.color':'#5b7a99','ytick.color':'#5b7a99',
    'text.color':'#c8dcf0','grid.color':'#1a2e4a','grid.alpha':.4,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
})

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw():
    df = pd.read_excel(DATA_FILE, sheet_name='Warehouse_5000_Records', header=1)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Stockout_Risk'] = (
        (df['Zone_Closing_Stock'] < df['Zone_Safety_Stock'] * 1.5) |
        (df['Zone_Cylinders_Delivered'] < df['Zone_Cylinders_Ordered'] * 0.9)
    ).astype(int)
    return df

@st.cache_data(show_spinner=False)
def preprocess(_df):
    train = _df.sample(frac=0.8, random_state=42).reset_index(drop=True)
    test  = _df.drop(train.index).reset_index(drop=True)
    tr_raw_copy = train.copy()

    tr = train.drop(columns=DROP_COLS, errors='ignore')
    te = test.drop(columns=DROP_COLS,  errors='ignore')

    for col in tr.select_dtypes(include=[np.number]).columns:
        med = tr[col].median()
        tr[col] = tr[col].fillna(med)
        te[col] = te[col].fillna(med)

    enc = {}
    for col in ['Warehouse_Zone','Subsidy_Type']:
        if col in tr.columns:
            le = LabelEncoder()
            tr[col] = le.fit_transform(tr[col].astype(str))
            te[col] = le.transform(te[col].astype(str))
            enc[col] = le

    feat = [c for c in tr.columns if c not in EXCLUDE_TGT]
    Xtr = tr[feat]; Xte = te[feat]
    ytr_d = tr['Gas_Consumption_kg'];  yte_d = te['Gas_Consumption_kg']
    ytr_s = tr['Stockout_Risk'].astype(int); yte_s = te['Stockout_Risk'].astype(int)

    sc = StandardScaler()
    Xtr_s = pd.DataFrame(sc.fit_transform(Xtr), columns=feat)
    Xte_s = pd.DataFrame(sc.transform(Xte),     columns=feat)
    return Xtr_s, Xte_s, ytr_d, yte_d, ytr_s, yte_s, sc, enc, feat, tr_raw_copy

@st.cache_resource(show_spinner=False)
def train_all(_Xtr, _ytr_d, _ytr_s):
    regs = {
        'Linear Regression': LinearRegression(),
        'Decision Tree':     DecisionTreeRegressor(max_depth=8, random_state=42),
        'Random Forest':     RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
    }
    clfs = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
    }
    for m in regs.values(): m.fit(_Xtr, _ytr_d)
    for m in clfs.values(): m.fit(_Xtr, _ytr_s)
    return regs, clfs

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔵 LPG Predictor")
    st.caption("Mitra Bharatgas · Murshidabad")
    page = st.radio("Navigation", [
        "📊 Overview & Data",
        "🤖 Train Models",
        "📈 Visualizations",
        "🔮 Predict Demand",
        "📋 Batch Report",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("Dataset: 5,000 records · 10 zones\nJan 2022 – Dec 2024")
    st.caption("Targets:\n• Gas Consumption kg [Regression]\n• Zone Stockout Risk [Classification]")

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    raw_df = load_raw()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & Data":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🔵 Smart Warehouse Demand Prediction</div>
      <div class="hero-sub">LPG Cylinder Supply Chain · Mitra Bharatgas Agency · Murshidabad, West Bengal</div>
      <div class="hero-badge">ML-Powered · Random Forest · R²=0.97 · F1=1.00</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Total Records</div><div class="kpi-value">{len(raw_df):,}</div><div class="kpi-sub">Household transactions</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Warehouse Zones</div><div class="kpi-value">{raw_df["Warehouse_Zone"].nunique()}</div><div class="kpi-sub">Murshidabad district</div></div>', unsafe_allow_html=True)
    with c3:
        avg_gas = raw_df['Gas_Consumption_kg'].mean()
        st.markdown(f'<div class="kpi"><div class="kpi-label">Avg Gas Consumption</div><div class="kpi-value">{avg_gas:.1f} kg</div><div class="kpi-sub">Per household / month</div></div>', unsafe_allow_html=True)
    with c4:
        sr = raw_df['Stockout_Risk'].mean() * 100
        st.markdown(f'<div class="kpi"><div class="kpi-label">Zone Stockout Risk</div><div class="kpi-value">{sr:.1f}%</div><div class="kpi-sub">Records flagged at risk</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec"><h3>📂 Dataset Preview</h3><p>First 300 rows of warehouse records</p></div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(300), use_container_width=True, height=360)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec"><h3>📍 Records by Zone</h3></div>', unsafe_allow_html=True)
        zc = raw_df['Warehouse_Zone'].value_counts().reset_index()
        zc.columns = ['Zone', 'Count']
        st.dataframe(zc, use_container_width=True, hide_index=True)
    with col2:
        st.markdown('<div class="sec"><h3>📊 Key Statistics</h3></div>', unsafe_allow_html=True)
        st.dataframe(raw_df[['Gas_Consumption_kg','Monthly_Income (₹)',
                               'LPG_Price_per_Cylinder (₹)','Zone_Opening_Stock',
                               'No_of_Family_Members']].describe().round(2),
                     use_container_width=True)

    buf = io.BytesIO()
    raw_df.to_excel(buf, index=False, engine='openpyxl')
    st.download_button("⬇ Download Full Dataset (.xlsx)", buf.getvalue(),
                       "warehouse_data.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🤖 Model Training</div>
      <div class="hero-sub">80/20 split · 4 Regression models · 4 Classification models</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Preprocessing data (80/20 split)…"):
        Xtr, Xte, ytr_d, yte_d, ytr_s, yte_s, sc, enc, feat, tr_raw = preprocess(raw_df)

    st.success(f"✅ Preprocessed — Train: **{len(Xtr):,}** · Test: **{len(Xte):,}** · Features: **{len(feat)}**")

    with st.expander("🔍 Feature columns used"):
        st.code(str(feat))

    with st.spinner("Training 8 models… (~30 seconds on first run)"):
        regs, clfs = train_all(Xtr, ytr_d, ytr_s)

    # Regression table
    st.markdown('<div class="sec"><h3>📉 Task A — Gas Consumption Regression (kg)</h3><p>Predict monthly LPG gas usage per household</p></div>', unsafe_allow_html=True)
    reg_rows = []
    for name, m in regs.items():
        p = m.predict(Xte)
        reg_rows.append({"Model": name,
                          "R² Score": round(r2_score(yte_d, p), 4),
                          "MAE (kg)": round(mean_absolute_error(yte_d, p), 4),
                          "RMSE (kg)": round(np.sqrt(mean_squared_error(yte_d, p)), 4)})
    rdf = pd.DataFrame(reg_rows).sort_values("R² Score", ascending=False).reset_index(drop=True)
    best_reg_name = rdf.iloc[0]["Model"]

    def hi(row):
        return ['background-color:#1e3a5f;color:#5ba3d9;font-weight:bold' if row.name == 0 else '' for _ in row]

    st.dataframe(rdf.style.apply(hi, axis=1), use_container_width=True, hide_index=True)
    st.success(f"🏆 Best Regression: **{best_reg_name}** · R²={rdf.iloc[0]['R² Score']}")

    # Classification table
    st.markdown('<div class="sec"><h3>⚠️ Task B — Zone Stockout Risk Classification</h3><p>Predict whether a zone faces stockout pressure this month</p></div>', unsafe_allow_html=True)
    clf_rows = []
    for name, m in clfs.items():
        p = m.predict(Xte)
        clf_rows.append({"Model": name,
                          "Accuracy": round(accuracy_score(yte_s, p), 4),
                          "Precision": round(precision_score(yte_s, p, zero_division=0), 4),
                          "Recall": round(recall_score(yte_s, p, zero_division=0), 4),
                          "F1 Score": round(f1_score(yte_s, p, zero_division=0), 4)})
    cdf = pd.DataFrame(clf_rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    best_clf_name = cdf.iloc[0]["Model"]
    st.dataframe(cdf.style.apply(hi, axis=1), use_container_width=True, hide_index=True)
    st.success(f"🏆 Best Classification: **{best_clf_name}** · F1={cdf.iloc[0]['F1 Score']}")

    # Save to session state
    st.session_state.update({
        "trained": True, "regs": regs, "clfs": clfs,
        "rdf": rdf, "cdf": cdf,
        "best_reg": best_reg_name, "best_clf": best_clf_name,
        "Xte": Xte, "yte_d": yte_d, "yte_s": yte_s,
        "sc": sc, "enc": enc, "feat": feat, "tr_raw": tr_raw,
    })

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">📈 Visualizations</div>
      <div class="hero-sub">8 charts · Model performance · Demand trends · Zone analysis · Feature importance</div>
    </div>""", unsafe_allow_html=True)

    if "trained" not in st.session_state:
        st.warning("⚠️ Go to **🤖 Train Models** first.")
        st.stop()

    regs    = st.session_state["regs"];    clfs    = st.session_state["clfs"]
    rdf     = st.session_state["rdf"];     cdf     = st.session_state["cdf"]
    br_name = st.session_state["best_reg"]; bc_name = st.session_state["best_clf"]
    Xte     = st.session_state["Xte"];     yte_d   = st.session_state["yte_d"]
    yte_s   = st.session_state["yte_s"];   feat    = st.session_state["feat"]
    tr_raw  = st.session_state["tr_raw"]
    br = regs[br_name]; bc = clfs[bc_name]
    br_pred = br.predict(Xte)
    bc_pred = bc.predict(Xte)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "📉 Demand Analysis", "🏭 Zone & Stockout", "⭐ Feature Importance"])

    # Tab 1 — Model Comparison
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(rdf['Model'], rdf['R² Score'], color=COLORS[:4], height=0.5)
            ax.set_xlim(0, 1.1); ax.set_xlabel('R² Score')
            ax.set_title('Regression Model Comparison — R²', fontweight='bold', fontsize=11)
            ax.axvline(0.9, color='#e74c3c', linestyle='--', alpha=.6, lw=1.2, label='Target 0.90')
            ax.legend(fontsize=8)
            for bar, v in zip(bars, rdf['R² Score']):
                ax.text(bar.get_width()+.01, bar.get_y()+bar.get_height()/2,
                        f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(cdf['Model'], cdf['F1 Score'], color=COLORS[:4], height=0.5)
            ax.set_xlim(0, 1.1); ax.set_xlabel('F1 Score')
            ax.set_title('Classification Model Comparison — F1', fontweight='bold', fontsize=11)
            for bar, v in zip(bars, cdf['F1 Score']):
                ax.text(bar.get_width()+.01, bar.get_y()+bar.get_height()/2,
                        f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        col3, col4 = st.columns(2)
        with col3:
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            cm = confusion_matrix(yte_s, bc_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Risk', 'At Risk'],
                        yticklabels=['No Risk', 'At Risk'],
                        annot_kws={'size': 13, 'weight': 'bold'}, cbar=False)
            ax.set_title(f'Confusion Matrix\n{bc_name}', fontweight='bold', fontsize=11)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col4:
            errors = yte_d.values - br_pred
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.hist(errors, bins=35, color='#1a5276', alpha=.85, edgecolor='#0d1b2a')
            ax.axvline(0, color='#e74c3c', linestyle='--', lw=1.5, label='Zero Error')
            ax.axvline(errors.mean(), color='#f39c12', linestyle='--', lw=1.5,
                       label=f'Mean: {errors.mean():.3f}')
            ax.set_xlabel('Error (Actual − Predicted)'); ax.set_ylabel('Frequency')
            ax.set_title('Prediction Error Distribution', fontweight='bold', fontsize=11)
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # Tab 2 — Demand Analysis
    with tab2:
        r2 = r2_score(yte_d, br_pred)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(yte_d, br_pred, alpha=.3, color='#1a5276', s=12)
        lims = [yte_d.min() - .5, yte_d.max() + .5]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect Prediction')
        ax.set_xlabel('Actual Gas Consumption (kg)'); ax.set_ylabel('Predicted (kg)')
        ax.set_title(f'Actual vs Predicted Gas Consumption\n{br_name} | R²={r2:.4f}',
                     fontweight='bold', fontsize=12)
        ax.legend(fontsize=10); plt.tight_layout(); st.pyplot(fig); plt.close()

        tr_raw['Gas_Consumption_kg'] = pd.to_numeric(tr_raw['Gas_Consumption_kg'], errors='coerce')
        tr_raw['Month_Number'] = pd.to_numeric(tr_raw['Month_Number'], errors='coerce')
        monthly = tr_raw.groupby('Month_Number')['Gas_Consumption_kg'].mean().reset_index()
        s_colors = {1:'#5dade2',2:'#5dade2',3:'#f39c12',4:'#f39c12',5:'#f39c12',
                    6:'#27ae60',7:'#27ae60',8:'#27ae60',9:'#27ae60',
                    10:'#e67e22',11:'#e67e22',12:'#5dade2'}
        fig, ax = plt.subplots(figsize=(11, 5))
        bc2 = [s_colors.get(int(m), '#1a5276') for m in monthly['Month_Number']]
        bars = ax.bar([MONTH_NAMES[int(m)-1] for m in monthly['Month_Number']],
                      monthly['Gas_Consumption_kg'], color=bc2, edgecolor='#0d1b2a', width=0.65)
        ax.set_title('Average Monthly Gas Consumption (kg)', fontweight='bold', fontsize=13)
        ax.set_xlabel('Month'); ax.set_ylabel('Avg Gas (kg)')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.05,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
        ax.legend(handles=[mpatches.Patch(color='#5dade2', label='Winter'),
                            mpatches.Patch(color='#f39c12', label='Summer'),
                            mpatches.Patch(color='#27ae60', label='Monsoon'),
                            mpatches.Patch(color='#e67e22', label='Autumn')], fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Tab 3 — Zone & Stockout
    with tab3:
        tr_raw['Stockout_Risk'] = pd.to_numeric(tr_raw['Stockout_Risk'], errors='coerce')
        zs = tr_raw.groupby('Warehouse_Zone')['Stockout_Risk'].mean().sort_values(ascending=False) * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        bc3 = ['#c0392b' if v > zs.mean() else '#1e8449' for v in zs.values]
        bars = ax.bar(zs.index, zs.values, color=bc3, edgecolor='#0d1b2a', width=0.65)
        ax.axhline(zs.mean(), color='#f39c12', linestyle='--', lw=1.5, label=f'Avg: {zs.mean():.1f}%')
        ax.set_title('Stockout Risk Rate by Warehouse Zone', fontweight='bold', fontsize=13)
        ax.set_ylabel('Risk Rate (%)'); ax.set_xlabel('Zone')
        ax.set_xticklabels(zs.index, rotation=30, ha='right'); ax.legend(fontsize=10)
        for bar, v in zip(bars, zs.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.3,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        tr_raw['Gas_Consumption_kg'] = pd.to_numeric(tr_raw['Gas_Consumption_kg'], errors='coerce')
        zd = tr_raw.groupby('Warehouse_Zone')['Gas_Consumption_kg'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(zd.index, zd.values, color='#1a5276', height=0.6)
        ax.set_title('Avg Gas Consumption per Household by Zone', fontweight='bold', fontsize=12)
        ax.set_xlabel('Avg Gas Consumption (kg)')
        for i, (idx, v) in enumerate(zd.items()):
            ax.text(v+.05, i, f'{v:.2f} kg', va='center', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Tab 4 — Feature Importance
    with tab4:
        if hasattr(br, 'feature_importances_'):
            fi = pd.Series(br.feature_importances_, index=feat).sort_values(ascending=True).tail(12)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(fi.index, fi.values,
                    color=[COLORS[i % len(COLORS)] for i in range(len(fi))], height=0.6)
            ax.set_title(f'Feature Importance — {br_name}', fontweight='bold', fontsize=12)
            ax.set_xlabel('Importance Score')
            for i, (idx, v) in enumerate(fi.items()):
                ax.text(v+.001, i, f'{v:.4f}', va='center', fontsize=9)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.info("Feature importances not available for Linear Regression.")

        tr_raw['Monthly_Income (₹)'] = pd.to_numeric(tr_raw['Monthly_Income (₹)'], errors='coerce')
        sample = tr_raw.sample(min(1000, len(tr_raw)), random_state=42)
        fig, ax = plt.subplots(figsize=(9, 5))
        sc2 = ax.scatter(sample['Monthly_Income (₹)'], sample['Gas_Consumption_kg'],
                         alpha=.35, c=sample['Gas_Consumption_kg'],
                         cmap='YlOrRd', s=15)
        plt.colorbar(sc2, ax=ax, label='Gas (kg)')
        ax.set_xlabel('Monthly Income (₹)'); ax.set_ylabel('Gas Consumption (kg)')
        ax.set_title('Income vs Gas Consumption', fontweight='bold', fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT DEMAND
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Demand":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🔮 Predict Future Demand</div>
      <div class="hero-sub">Enter household & zone parameters → get gas forecast + stockout risk</div>
    </div>""", unsafe_allow_html=True)

    if "trained" not in st.session_state:
        st.warning("⚠️ Go to **🤖 Train Models** first.")
        st.stop()

    br   = st.session_state["regs"][st.session_state["best_reg"]]
    bc   = st.session_state["clfs"][st.session_state["best_clf"]]
    sc   = st.session_state["sc"]
    feat = st.session_state["feat"]

    with st.form("pred_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📍 Location & Time**")
            zone    = st.selectbox("Warehouse Zone", sorted(ZONES))
            year    = st.selectbox("Year", [2024, 2025, 2026])
            month   = st.selectbox("Month", list(range(1, 13)),
                                   format_func=lambda m: MONTH_NAMES[m-1])
            is_win  = st.checkbox("Winter Month?", value=(month in [12, 1, 2]))
            is_fest = st.checkbox("Festival Month?", value=(month in [10, 1]))
        with col2:
            st.markdown("**👨‍👩‍👧 Household Profile**")
            subsidy  = st.selectbox("Subsidy Type", ['PMUY', 'Non-Subsidized'])
            members  = st.slider("Family Members", 1, 10, 4)
            adults   = st.slider("Adults", 1, 8, 2)
            children = st.slider("Children", 0, 6, 2)
            income   = st.number_input("Monthly Income (₹)", 5000, 100000, 22000, step=1000)
        with col3:
            st.markdown("**🏭 Zone Operations**")
            lpg_price  = st.number_input("LPG Price/Cylinder (₹)", 700, 1200, 903, step=10)
            open_stock = st.number_input("Zone Opening Stock", 100, 2000, 800, step=50)
            ordered    = st.number_input("Cylinders Ordered", 100, 1500, 435, step=25)
            lead_time  = st.slider("Lead Time (days)", 1, 15, 5)
            delivered  = st.number_input("Cylinders Delivered", 0, 1500, 410, step=25)
            damaged    = st.number_input("Damaged Cylinders", 0, 50, 2)

        submitted = st.form_submit_button("🔮 Predict Now", use_container_width=True)

    if submitted:
        import calendar
        days = calendar.monthrange(year, month)[1]
        scen = {
            'Warehouse_Zone': ZONE_ENC[zone], 'Year': year, 'Month_Number': month,
            'Is_Winter': int(is_win), 'Is_Festival_Month': int(is_fest),
            'Days_in_Month': days, 'No_of_Family_Members': members,
            'No_of_Adults': adults, 'No_of_Children': children,
            'Monthly_Income (₹)': income, 'Subsidy_Type': SUBSIDY_ENC[subsidy],
            'LPG_Price_per_Cylinder (₹)': lpg_price,
            'Zone_Opening_Stock': open_stock, 'Zone_Cylinders_Ordered': ordered,
            'Lead_Time_Days': lead_time, 'Zone_Cylinders_Delivered': delivered,
            'Damaged_Cylinders': damaged,
        }
        inp    = pd.DataFrame([{c: scen.get(c, 0) for c in feat}])
        inp_sc = sc.transform(inp)

        gas_pred    = max(0.1, round(br.predict(inp_sc)[0], 2))
        risk_prob   = bc.predict_proba(inp_sc)[0][1] * 100
        cyl_needed  = round(gas_pred / 14.2, 2)
        recommended = max(1, round(cyl_needed * 1.15))
        safety_stk  = max(1, round(cyl_needed * 0.15))

        risk_label = "🔴 HIGH" if risk_prob > 60 else "🟡 MEDIUM" if risk_prob > 30 else "🟢 LOW"
        risk_color = "#e74c3c" if risk_prob > 60 else "#f39c12" if risk_prob > 30 else "#27ae60"

        st.markdown('<div class="sec"><h3>📊 Prediction Results</h3></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="kpi"><div class="kpi-label">Gas Consumption</div><div class="kpi-value" style="color:#5ba3d9">{gas_pred} kg</div><div class="kpi-sub">Predicted — {MONTH_NAMES[month-1]} {year}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi"><div class="kpi-label">Cylinders Needed</div><div class="kpi-value" style="color:#27ae60">{cyl_needed}</div><div class="kpi-sub">@ 14.2 kg per cylinder</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="kpi"><div class="kpi-label">Stockout Risk</div><div class="kpi-value" style="color:{risk_color}">{risk_prob:.1f}%</div><div class="kpi-sub">{risk_label} RISK</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="kpi"><div class="kpi-label">Recommended Order</div><div class="kpi-value" style="color:#f39c12">{recommended}</div><div class="kpi-sub">+15% buffer · Safety: {safety_stk} cyl</div></div>', unsafe_allow_html=True)

        # Risk gauge bar
        fig, ax = plt.subplots(figsize=(8, 1.6))
        ax.barh(['Risk'], [risk_prob], color=risk_color, height=0.4)
        ax.barh(['Risk'], [100 - risk_prob], left=risk_prob, color='#1a2e4a', height=0.4)
        ax.set_xlim(0, 100); ax.set_xlabel('Probability (%)')
        ax.set_title(f'Zone Stockout Risk: {risk_prob:.1f}%  {risk_label}', fontweight='bold')
        ax.text(min(risk_prob / 2, 90), 0, f'{risk_prob:.1f}%',
                ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        ax.axvline(30, color='#f39c12', linestyle=':', lw=1, alpha=.7)
        ax.axvline(60, color='#e74c3c', linestyle=':', lw=1, alpha=.7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.info(
            f"💡 **Interpretation**: This household in **{zone}** is expected to consume "
            f"**{gas_pred} kg** of LPG (≈ {cyl_needed} cylinders) in {MONTH_NAMES[month-1]} {year}. "
            f"The zone has a **{risk_prob:.1f}% stockout risk** — "
            f"recommended zone order: **{recommended} cylinders** (includes 15% safety buffer)."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BATCH REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch Report":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">📋 Batch Prediction Report</div>
      <div class="hero-sub">All 10 zones × selected months → heatmaps + Excel export</div>
    </div>""", unsafe_allow_html=True)

    if "trained" not in st.session_state:
        st.warning("⚠️ Go to **🤖 Train Models** first.")
        st.stop()

    br   = st.session_state["regs"][st.session_state["best_reg"]]
    bc   = st.session_state["clfs"][st.session_state["best_clf"]]
    sc   = st.session_state["sc"]
    feat = st.session_state["feat"]
    rdf  = st.session_state["rdf"]
    cdf  = st.session_state["cdf"]

    col1, col2 = st.columns(2)
    with col1:
        year_sel = st.selectbox("Year", [2024, 2025, 2026], index=1)
    with col2:
        month_sel = st.multiselect("Months", list(range(1, 13)), default=[1, 4, 7, 10],
                                   format_func=lambda m: MONTH_NAMES[m-1])

    defaults = {
        'No_of_Family_Members': 4, 'No_of_Adults': 2, 'No_of_Children': 2,
        'Monthly_Income (₹)': 22000, 'Subsidy_Type': 1,
        'LPG_Price_per_Cylinder (₹)': 903,
        'Zone_Opening_Stock': 794, 'Zone_Cylinders_Ordered': 435,
        'Lead_Time_Days': 5, 'Zone_Cylinders_Delivered': 409, 'Damaged_Cylinders': 2,
    }

    if st.button("🚀 Generate Batch Predictions", use_container_width=True) and month_sel:
        import calendar
        results = []
        prog  = st.progress(0)
        total = len(sorted(ZONES)) * len(month_sel)
        n = 0

        for zone in sorted(ZONES):
            for month in sorted(month_sel):
                days = calendar.monthrange(year_sel, month)[1]
                scen = {'Warehouse_Zone': ZONE_ENC[zone], 'Year': year_sel,
                        'Month_Number': month,
                        'Is_Winter': int(month in [12, 1, 2]),
                        'Is_Festival_Month': int(month in [10, 1, 3]),
                        'Days_in_Month': days, **defaults}
                inp    = pd.DataFrame([{c: scen.get(c, 0) for c in feat}])
                inp_sc = sc.transform(inp)
                gas_pred  = max(0.1, round(br.predict(inp_sc)[0], 2))
                risk_prob = bc.predict_proba(inp_sc)[0][1] * 100
                cyl_need  = round(gas_pred / 14.2, 2)
                rec       = max(1, round(cyl_need * 1.15))
                safety    = max(1, round(cyl_need * 0.15))
                risk      = "HIGH" if risk_prob > 60 else "MEDIUM" if risk_prob > 30 else "LOW"
                results.append({
                    'Zone': zone, 'Year': year_sel, 'Month': MONTH_NAMES[month-1],
                    'Gas_Consumption_kg': gas_pred,
                    'Cylinders_Needed': cyl_need,
                    'Stockout_Prob_%': round(risk_prob, 1),
                    'Risk_Level': risk,
                    'Recommended_Order': rec,
                    'Safety_Stock': safety,
                })
                n += 1; prog.progress(n / total)

        res_df = pd.DataFrame(results)

        def cr(v):
            if v == "HIGH":   return 'background-color:#3d0000;color:#e74c3c;font-weight:bold'
            if v == "MEDIUM": return 'background-color:#2d1d00;color:#f39c12;font-weight:bold'
            return 'background-color:#001a0d;color:#27ae60;font-weight:bold'

        st.markdown('<div class="sec"><h3>📊 Batch Results Table</h3></div>', unsafe_allow_html=True)
        st.dataframe(res_df.style.applymap(cr, subset=['Risk_Level']),
                     use_container_width=True, height=420)

        # Gas consumption heatmap
        st.markdown('<div class="sec"><h3>🌡️ Gas Consumption Heatmap</h3></div>', unsafe_allow_html=True)
        pivot = res_df.pivot_table(values='Gas_Consumption_kg', index='Zone', columns='Month')
        fig, ax = plt.subplots(figsize=(max(6, len(month_sel)*1.8), max(5, len(ZONES)*0.7)))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    annot_kws={'size': 9, 'weight': 'bold'},
                    cbar_kws={'label': 'Gas Consumption (kg)'})
        ax.set_title('Predicted Gas Consumption — Zone × Month', fontweight='bold', fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Stockout risk heatmap
        st.markdown('<div class="sec"><h3>⚠️ Stockout Risk Heatmap</h3></div>', unsafe_allow_html=True)
        pivot_r = res_df.pivot_table(values='Stockout_Prob_%', index='Zone', columns='Month')
        fig, ax = plt.subplots(figsize=(max(6, len(month_sel)*1.8), max(5, len(ZONES)*0.7)))
        sns.heatmap(pivot_r, annot=True, fmt='.0f', cmap='Reds', ax=ax,
                    annot_kws={'size': 9, 'weight': 'bold'},
                    cbar_kws={'label': 'Stockout Risk (%)'})
        ax.set_title('Stockout Risk Heatmap — Zone × Month (%)', fontweight='bold', fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Excel export
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as w:
            res_df.to_excel(w, sheet_name='Batch_Predictions', index=False)
            pd.DataFrame({
                'Metric': ['Best Demand Model', 'R² Score', 'MAE (kg)', 'RMSE (kg)',
                            'Best Stockout Model', 'F1 Score', 'Accuracy'],
                'Value':  [st.session_state["best_reg"],
                           f"{rdf.iloc[0]['R² Score']:.4f}",
                           f"{rdf.iloc[0]['MAE (kg)']:.4f}",
                           f"{rdf.iloc[0]['RMSE (kg)']:.4f}",
                           st.session_state["best_clf"],
                           f"{cdf.iloc[0]['F1 Score']:.4f}",
                           f"{cdf.iloc[0]['Accuracy']:.4f}"]
            }).to_excel(w, sheet_name='Model_Summary', index=False)

        st.download_button(
            "⬇ Download Batch Report (.xlsx)", buf.getvalue(),
            f"lpg_predictions_{year_sel}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
