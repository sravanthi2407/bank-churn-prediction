"""
PHASE 9: STREAMLIT APPLICATION
Bank Customer Churn Prediction Dashboard
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, sys, warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155; border-radius: 12px;
        padding: 18px; text-align: center; margin: 8px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #06b6d4; }
    .metric-label { font-size: 0.9rem; color: #94a3b8; margin-top: 4px; }
    .risk-high   { background: linear-gradient(135deg,#450a0a,#1e293b); border:2px solid #f43f5e; border-radius:12px; padding:18px; text-align:center; }
    .risk-medium { background: linear-gradient(135deg,#431407,#1e293b); border:2px solid #f97316; border-radius:12px; padding:18px; text-align:center; }
    .risk-low    { background: linear-gradient(135deg,#052e16,#1e293b); border:2px solid #10b981; border-radius:12px; padding:18px; text-align:center; }
    .section-header { font-size:1.5rem; font-weight:700; color:#06b6d4;
                      border-left:4px solid #06b6d4; padding-left:12px; margin:20px 0 12px; }
    div[data-testid="stSidebar"] { background-color: #1e293b; }
    .stSelectbox>div>div { background-color: #1e293b; color: #e2e8f0; }
    .stSlider .stSlider { background-color: #334155; }
    h1,h2,h3 { color: #e2e8f0; }
    .stDataFrame { background-color: #1e293b; }
</style>
""", unsafe_allow_html=True)

BASE = '/home/claude/Bank_Customer_Churn_Project'

@st.cache_resource
def load_artifacts():
    with open(f'{BASE}/models/best_model.pkl','rb') as f:
        model = pickle.load(f)
    with open(f'{BASE}/models/scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    with open(f'{BASE}/models/feature_cols.pkl','rb') as f:
        feature_cols = pickle.load(f)
    with open(f'{BASE}/models/label_encoders.pkl','rb') as f:
        encoders = pickle.load(f)
    return model, scaler, feature_cols, encoders

@st.cache_data
def load_data():
    df = pd.read_csv(f'{BASE}/data/European_Bank.csv')
    df_proc = pd.read_csv(f'{BASE}/data/processed.csv')
    risk    = pd.read_csv(f'{BASE}/reports/customer_risk_scores.csv')
    return df, df_proc, risk

def predict_churn(inputs, model, scaler, feature_cols):
    df_in = pd.DataFrame([inputs])
    # Feature engineering
    df_in['BalanceSalaryRatio'] = df_in['Balance'] / (df_in['EstimatedSalary'] + 1)
    df_in['ProductDensity']     = df_in['NumOfProducts'] / (df_in['Tenure'] + 1)
    df_in['EngagementProductInteraction'] = df_in['IsActiveMember'] * df_in['NumOfProducts']
    df_in['AgeTenureInteraction'] = df_in['Age'] * df_in['Tenure']
    df_in['IsZeroBalance'] = (df_in['Balance'] == 0).astype(int)
    df_in['AgeGroup'] = pd.cut(df_in['Age'], bins=[17,30,40,50,60,100], labels=[0,1,2,3,4]).astype(int)
    df_in['CreditRisk'] = pd.cut(df_in['CreditScore'], bins=[349,579,669,739,799,851],
                                  labels=[4,3,2,1,0]).astype(int)
    X = df_in[feature_cols]
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0,1]
    return float(prob)

try:
    model, scaler, feature_cols, encoders = load_artifacts()
    df_raw, df_proc, df_risk = load_data()
    artifacts_ok = True
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    artifacts_ok = False

# ── Sidebar Navigation ────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏦 Bank Churn AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home & KPIs",
    "🔮 Churn Predictor",
    "📊 Feature Importance",
    "🎮 What-If Simulator",
    "📈 Analytics Dashboard"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Gradient Boosting")
st.sidebar.markdown("**Dataset:** European Bank · 10,000 customers")
st.sidebar.markdown("**Best AUC:** 0.8664")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: HOME & KPIs
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Home & KPIs":
    st.markdown("# 🏦 Predictive Modeling & Risk Scoring for Bank Customer Churn")
    st.markdown("#### AI-Powered Customer Retention Intelligence Platform")
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">10,000</div><div class="metric-label">Total Customers</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-value">20.4%</div><div class="metric-label">Churn Rate</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-value">0.866</div><div class="metric-label">ROC-AUC Score</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-value">545</div><div class="metric-label">High-Risk Customers</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card"><div class="metric-value">86.8%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">🎯 Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        This platform applies **machine learning** to predict which bank customers are at risk of churning,
        enabling proactive retention strategies.

        **Business Impact:**
        - Identify high-risk customers **before** they leave
        - Prioritize retention budget on the **right customers**
        - Reduce churn rate by an estimated **15–25%**
        - Protect annual revenue by retaining high-value accounts

        **Models Trained:** Logistic Regression · Decision Tree · Random Forest · Gradient Boosting
        
        **Best Model:** Gradient Boosting with hyperparameter tuning via GridSearchCV
        """)

    with col2:
        st.markdown('<div class="section-header">📋 Dataset Summary</div>', unsafe_allow_html=True)
        summary = {
            'Metric': ['Rows','Columns','Missing Values','Duplicates','Churn Rate','Geography','Avg Age','Avg Balance'],
            'Value':  ['10,000','14','0 (0%)','0 (0%)','20.4%','France / Germany / Spain','38.9 years','€76,486']
        }
        st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📸 EDA Overview</div>', unsafe_allow_html=True)
    img_col1, img_col2 = st.columns(2)
    for path, col, title in [
        (f'{BASE}/reports/phase1_data_overview.png', img_col1, 'Data Overview'),
        (f'{BASE}/reports/phase2_bivariate.png', img_col2, 'Bivariate Analysis'),
    ]:
        if os.path.exists(path):
            col.image(path, caption=title, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2: CHURN PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif page == "🔮 Churn Predictor":
    st.markdown("# 🔮 Customer Churn Predictor")
    st.markdown("Enter customer details to get an instant churn risk assessment.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Demographics**")
        age         = st.slider("Age", 18, 92, 38)
        gender      = st.selectbox("Gender", ["Male","Female"])
        geography   = st.selectbox("Geography", ["France","Germany","Spain"])
    with col2:
        st.markdown("**Financial Profile**")
        credit_score   = st.slider("Credit Score", 350, 850, 650)
        balance        = st.number_input("Account Balance (€)", 0.0, 260000.0, 76000.0, step=1000.0)
        estimated_sal  = st.number_input("Estimated Salary (€)", 11.0, 200000.0, 100000.0, step=1000.0)
    with col3:
        st.markdown("**Banking Behaviour**")
        tenure         = st.slider("Tenure (years)", 0, 10, 5)
        num_products   = st.slider("Number of Products", 1, 4, 1)
        has_cr_card    = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        is_active      = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")

    if st.button("🚀 Predict Churn Risk", use_container_width=True):
        geo_enc = list(encoders['geo'].classes_).index(geography)
        gen_enc = list(encoders['gen'].classes_).index(gender)

        inputs = {
            'CreditScore': credit_score, 'Geography_enc': geo_enc,
            'Gender_enc': gen_enc, 'Age': age, 'Tenure': tenure,
            'Balance': balance, 'NumOfProducts': num_products,
            'HasCrCard': has_cr_card, 'IsActiveMember': is_active,
            'EstimatedSalary': estimated_sal
        }

        prob = predict_churn(inputs, model, scaler, feature_cols)
        score = int(prob * 1000)

        st.markdown("---")
        if prob >= 0.80:
            cat, css, emoji = "High Risk", "risk-high", "🚨"
            action = "🔴 IMMEDIATE ACTION REQUIRED: Assign dedicated relationship manager + exclusive retention offer."
        elif prob >= 0.50:
            cat, css, emoji = "Medium Risk", "risk-medium", "⚠️"
            action = "🟡 PROACTIVE RETENTION: Schedule personalised check-in call + product upgrade offer."
        else:
            cat, css, emoji = "Low Risk", "risk-low", "✅"
            action = "🟢 MONITOR: Include in quarterly loyalty programme. No immediate action required."

        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{prob:.1%}")
        r2.metric("Risk Score (0–1000)", str(score))
        r3.metric("Risk Category", f"{emoji} {cat}")

        st.markdown(f'<div class="{css}"><h3>{emoji} {cat}</h3><p style="font-size:1.1rem;color:#e2e8f0">{action}</p></div>', unsafe_allow_html=True)

        # Gauge chart
        fig, ax = plt.subplots(figsize=(7, 3.5), subplot_kw={'projection':'polar'})
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        theta = np.linspace(np.pi, 0, 200)
        for i, (lo, hi, color) in enumerate([(0,.5,'#10b981'),(.5,.8,'#f97316'),(.8,1,'#f43f5e')]):
            t1, t2 = np.pi * (1 - hi), np.pi * (1 - lo)
            t_seg = np.linspace(t2, t1, 50)
            ax.fill_between(t_seg, 0.6, 1, alpha=0.6, color=color)
        needle = np.pi * (1 - prob)
        ax.annotate('', xy=(needle, 0.85), xytext=(needle, 0.0),
                    arrowprops=dict(arrowstyle='->', color='white', lw=3))
        ax.set_ylim(0, 1.1); ax.set_xlim(0, np.pi)
        ax.axis('off')
        ax.text(np.pi/2, 0.2, f"{prob:.1%}", ha='center', va='center',
                fontsize=22, fontweight='bold', color='white',
                transform=ax.transData)
        ax.set_title(f"Churn Risk Gauge — {cat}", color='#e2e8f0', fontsize=13, pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════════════════════
# PAGE 3: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Feature Importance":
    st.markdown("# 📊 Feature Importance Dashboard")
    st.markdown("Understanding what drives customer churn.")
    st.markdown("---")

    img1 = f'{BASE}/reports/phase8_explainability.png'
    img2 = f'{BASE}/reports/phase8_pdp.png'
    if os.path.exists(img1):
        st.image(img1, caption='Feature Importance Analysis', use_container_width=True)
    if os.path.exists(img2):
        st.image(img2, caption='Partial Dependence Plots', use_container_width=True)

    fi_path = f'{BASE}/reports/feature_importance.csv'
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)
        st.markdown("### Feature Importance Table")
        st.dataframe(fi_df.round(4), hide_index=True, use_container_width=True)

    st.markdown("""
    **Key Insights:**
    - 🥇 **Age** is the strongest churn predictor — older customers (40+) churn more
    - 🥈 **Number of Products** — customers with 3–4 products show paradoxical high churn
    - 🥉 **Engagement × Products** — inactive customers with multiple products are highest risk
    - 📍 **Geography** — Germany customers churn at 32.4%, double other regions
    - 💰 **Balance** — higher balances correlate with churn (high-value customers leaving)
    """)

# ═══════════════════════════════════════════════════════════════
# PAGE 4: WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════
elif page == "🎮 What-If Simulator":
    st.markdown("# 🎮 What-If Churn Simulator")
    st.markdown("Adjust customer attributes and see real-time churn probability changes.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🎛️ Customer Parameters")
        age_s       = st.slider("Age", 18, 92, 40, key='sim_age')
        balance_s   = st.slider("Balance (€)", 0, 250000, 80000, step=5000, key='sim_bal')
        credit_s    = st.slider("Credit Score", 350, 850, 650, key='sim_cs')
        products_s  = st.slider("Num Products", 1, 4, 1, key='sim_prod')
        active_s    = st.slider("Is Active Member (0=No, 1=Yes)", 0, 1, 1, key='sim_act')
        tenure_s    = st.slider("Tenure (years)", 0, 10, 5, key='sim_ten')
        geo_s       = st.selectbox("Geography", ["France","Germany","Spain"], key='sim_geo')
        gender_s    = st.selectbox("Gender", ["Male","Female"], key='sim_gen')
        salary_s    = st.slider("Estimated Salary (€)", 0, 200000, 100000, step=5000, key='sim_sal')
        hascc_s     = st.selectbox("Has Credit Card", [1,0], format_func=lambda x:"Yes" if x else "No", key='sim_cc')

    with col2:
        st.markdown("### 📊 Live Prediction")
        geo_enc = list(encoders['geo'].classes_).index(geo_s)
        gen_enc = list(encoders['gen'].classes_).index(gender_s)
        inputs = {
            'CreditScore': credit_s,'Geography_enc': geo_enc,'Gender_enc': gen_enc,
            'Age': age_s,'Tenure': tenure_s,'Balance': balance_s,
            'NumOfProducts': products_s,'HasCrCard': hascc_s,
            'IsActiveMember': active_s,'EstimatedSalary': salary_s
        }
        prob_s = predict_churn(inputs, model, scaler, feature_cols)
        cat_s = "🚨 High Risk" if prob_s>=0.8 else ("⚠️ Medium Risk" if prob_s>=0.5 else "✅ Low Risk")
        clr   = "#f43f5e" if prob_s>=0.8 else ("#f97316" if prob_s>=0.5 else "#10b981")

        st.markdown(f"""
        <div style="background:#1e293b;border:2px solid {clr};border-radius:16px;padding:32px;text-align:center">
            <div style="font-size:4rem;font-weight:800;color:{clr}">{prob_s:.1%}</div>
            <div style="font-size:1.4rem;color:#e2e8f0;margin-top:8px">{cat_s}</div>
            <div style="font-size:0.9rem;color:#94a3b8;margin-top:8px">Risk Score: {int(prob_s*1000)}/1000</div>
        </div>
        """, unsafe_allow_html=True)

        # Impact simulation — vary age
        st.markdown("#### Sensitivity: Age vs Churn Probability")
        ages_sim = list(range(18, 93, 5))
        probs_sim = []
        for a in ages_sim:
            inp2 = inputs.copy(); inp2['Age'] = a
            probs_sim.append(predict_churn(inp2, model, scaler, feature_cols))
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        fig2.patch.set_facecolor('#0f172a'); ax2.set_facecolor('#1e293b')
        ax2.plot(ages_sim, [p*100 for p in probs_sim], color='#06b6d4', lw=2.5)
        ax2.axvline(age_s, color='#facc15', ls='--', lw=1.5, label=f'Current age: {age_s}')
        ax2.fill_between(ages_sim, [p*100 for p in probs_sim], alpha=0.15, color='#06b6d4')
        ax2.axhline(50, color='#f97316', ls=':', lw=1, label='Medium threshold')
        ax2.axhline(80, color='#f43f5e', ls=':', lw=1, label='High threshold')
        ax2.set_xlabel('Age', color='#94a3b8'); ax2.set_ylabel('Churn Probability (%)', color='#94a3b8')
        ax2.set_title('Age Sensitivity Analysis', color='#e2e8f0', fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(color='#334155', lw=0.5)
        ax2.tick_params(colors='#94a3b8')
        st.pyplot(fig2, use_container_width=True)
        plt.close()

# ═══════════════════════════════════════════════════════════════
# PAGE 5: ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Analytics Dashboard":
    st.markdown("# 📈 Analytics Dashboard")
    st.markdown("Interactive exploration of churn patterns across the customer base.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Churn Analysis","🗺️ Segmentation","🏆 Model Performance"])

    with tab1:
        st.image(f'{BASE}/reports/phase2_bivariate.png', caption='Bivariate Churn Analysis', use_container_width=True)
        st.image(f'{BASE}/reports/phase2_multivariate.png', caption='Multivariate Analysis', use_container_width=True)
    with tab2:
        st.image(f'{BASE}/reports/phase7_risk_scoring.png', caption='Risk Segmentation', use_container_width=True)
        st.image(f'{BASE}/reports/phase2_univariate.png', caption='Univariate Distributions', use_container_width=True)
    with tab3:
        st.image(f'{BASE}/reports/phase6_evaluation.png', caption='Model Evaluation Dashboard', use_container_width=True)
        results_path = f'{BASE}/reports/model_results.csv'
        if os.path.exists(results_path):
            res_df = pd.read_csv(results_path)
            st.markdown("### Model Comparison Table")
            display_cols = ['Model','Accuracy','Precision','Recall','F1','ROC-AUC','CV-AUC']
            st.dataframe(res_df[[c for c in display_cols if c in res_df.columns]].round(4),
                         hide_index=True, use_container_width=True)
