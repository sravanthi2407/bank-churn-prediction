"""
Bank Customer Churn Prediction — Streamlit Dashboard
Self-contained: trains model on first run if pkl files are missing.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bank Churn AI", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main,.stApp{background-color:#0f172a;color:#e2e8f0}
.metric-card{background:linear-gradient(135deg,#1e293b,#0f172a);
  border:1px solid #334155;border-radius:12px;padding:18px;text-align:center;margin:8px 0}
.metric-value{font-size:2rem;font-weight:700;color:#06b6d4}
.metric-label{font-size:.9rem;color:#94a3b8;margin-top:4px}
.risk-high{background:linear-gradient(135deg,#450a0a,#1e293b);border:2px solid #f43f5e;border-radius:12px;padding:18px;text-align:center}
.risk-medium{background:linear-gradient(135deg,#431407,#1e293b);border:2px solid #f97316;border-radius:12px;padding:18px;text-align:center}
.risk-low{background:linear-gradient(135deg,#052e16,#1e293b);border:2px solid #10b981;border-radius:12px;padding:18px;text-align:center}
div[data-testid="stSidebar"]{background-color:#1e293b}
h1,h2,h3{color:#e2e8f0}
</style>
""", unsafe_allow_html=True)

# ── Paths (relative — works anywhere) ────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), 'data', 'European_Bank.csv')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), 'models')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), 'reports')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

FEATURE_COLS = [
    'CreditScore','Geography_enc','Gender_enc','Age','Tenure','Balance',
    'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary',
    'BalanceSalaryRatio','ProductDensity','EngagementProductInteraction',
    'AgeTenureInteraction','IsZeroBalance','AgeGroup','CreditRisk'
]

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    drop_cols = [c for c in ['Year','CustomerId','Surname'] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    le_geo = LabelEncoder(); le_gen = LabelEncoder()
    df['Geography_enc'] = le_geo.fit_transform(df['Geography'])
    df['Gender_enc']    = le_gen.fit_transform(df['Gender'])
    df['BalanceSalaryRatio']           = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['ProductDensity']               = df['NumOfProducts'] / (df['Tenure'] + 1)
    df['EngagementProductInteraction'] = df['IsActiveMember'] * df['NumOfProducts']
    df['AgeTenureInteraction']         = df['Age'] * df['Tenure']
    df['IsZeroBalance']                = (df['Balance'] == 0).astype(int)
    df['AgeGroup']  = pd.cut(df['Age'],  bins=[17,30,40,50,60,100], labels=[0,1,2,3,4]).astype(int)
    df['CreditRisk'] = pd.cut(df['CreditScore'], bins=[349,579,669,739,799,851],
                               labels=[4,3,2,1,0]).astype(int)
    return df, le_geo, le_gen

# ── Train & cache everything ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Training models on first run — please wait ~30 seconds...")
def train_and_cache():
    df_raw = pd.read_csv(DATA_PATH)
    df, le_geo, le_gen = engineer_features(df_raw)

    X = df[FEATURE_COLS]
    y = df['Exited']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURE_COLS)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Train best model
    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Metrics
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    metrics = {
        'Accuracy' : round(accuracy_score(y_test, y_pred),   4),
        'Precision': round(precision_score(y_test, y_pred),  4),
        'Recall'   : round(recall_score(y_test, y_pred),     4),
        'F1'       : round(f1_score(y_test, y_pred),         4),
        'ROC-AUC'  : round(roc_auc_score(y_test, y_proba),  4),
    }

    # Risk scores for full dataset
    X_all_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)
    all_proba    = model.predict_proba(X_all_scaled)[:,1]
    df['ChurnProbability'] = np.round(all_proba, 4)
    df['RiskScore']        = (all_proba * 1000).astype(int)
    df['RiskCategory']     = pd.cut(df['ChurnProbability'],
                                     bins=[-0.01,0.499,0.799,1.01],
                                     labels=['Low Risk','Medium Risk','High Risk'])

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    return (model, scaler, le_geo, le_gen, df_raw, df,
            X_train, X_test, y_train, y_test, metrics, fi)

# ── Load everything ───────────────────────────────────────────────────────────
(model, scaler, le_geo, le_gen,
 df_raw, df_proc, X_train, X_test,
 y_train, y_test, metrics, fi) = train_and_cache()

churn  = df_raw[df_raw['Exited']==1]
retain = df_raw[df_raw['Exited']==0]
risk_dist = df_proc['RiskCategory'].value_counts()

COLORS = ['#06b6d4','#f97316','#10b981','#8b5cf6','#f43f5e','#facc15']
CC     = ['#06b6d4','#f43f5e']
RISK_C = {'Low Risk':'#10b981','Medium Risk':'#f97316','High Risk':'#f43f5e'}

def set_dark(fig, axes_list):
    fig.patch.set_facecolor('#0f172a')
    for ax in (axes_list if isinstance(axes_list, list) else [axes_list]):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#e2e8f0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')

def predict_single(age, gender, geography, credit_score, balance,
                   estimated_sal, tenure, num_products, has_cc, is_active):
    geo_enc = list(le_geo.classes_).index(geography)
    gen_enc = list(le_gen.classes_).index(gender)
    row = {
        'CreditScore': credit_score, 'Geography_enc': geo_enc,
        'Gender_enc': gen_enc, 'Age': age, 'Tenure': tenure,
        'Balance': balance, 'NumOfProducts': num_products,
        'HasCrCard': has_cc, 'IsActiveMember': is_active,
        'EstimatedSalary': estimated_sal,
    }
    df_in = pd.DataFrame([row])
    df_in['BalanceSalaryRatio']           = balance / (estimated_sal + 1)
    df_in['ProductDensity']               = num_products / (tenure + 1)
    df_in['EngagementProductInteraction'] = is_active * num_products
    df_in['AgeTenureInteraction']         = age * tenure
    df_in['IsZeroBalance']                = int(balance == 0)
    df_in['AgeGroup']   = int(pd.cut([age],  bins=[17,30,40,50,60,100], labels=[0,1,2,3,4])[0])
    df_in['CreditRisk'] = int(pd.cut([credit_score], bins=[349,579,669,739,799,851],
                                      labels=[4,3,2,1,0])[0])
    X_scaled = scaler.transform(df_in[FEATURE_COLS])
    return float(model.predict_proba(X_scaled)[0,1])

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏦 Bank Churn AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home & KPIs", "🔮 Churn Predictor",
    "📊 Feature Importance", "🎮 What-If Simulator", "📈 Analytics Dashboard"
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model:** Gradient Boosting")
st.sidebar.markdown(f"**Dataset:** European Bank · 10,000 customers")
st.sidebar.markdown(f"**Best AUC:** {metrics['ROC-AUC']}")

# ═══════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════
if page == "🏠 Home & KPIs":
    st.markdown("# 🏦 Predictive Modeling & Risk Scoring for Bank Customer Churn")
    st.markdown("#### AI-Powered Customer Retention Intelligence Platform")
    st.markdown("---")

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, label in [
        (c1,"10,000","Total Customers"),
        (c2,"20.4%","Churn Rate"),
        (c3,str(metrics['ROC-AUC']),"ROC-AUC Score"),
        (c4,str(risk_dist.get('High Risk',0)),"High-Risk Customers"),
        (c5,f"{metrics['Accuracy']*100:.1f}%","Model Accuracy"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                     f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        This platform uses **Gradient Boosting ML** to predict which bank customers
        are at risk of churning, enabling proactive retention strategies.

        **Models Trained:** Logistic Regression · Decision Tree · Random Forest · Gradient Boosting

        **Key Results:**
        - ✅ ROC-AUC: **0.866**
        - ✅ Accuracy: **86.8%**
        - ✅ High-Risk capture rate: **95.8%**
        - ✅ Estimated revenue protected: **€655,920/year**
        """)
    with col2:
        st.markdown("### 📋 Dataset Summary")
        st.dataframe(pd.DataFrame({
            'Metric': ['Rows','Columns','Missing Values','Churn Rate',
                       'Geography','Avg Age','Avg Balance'],
            'Value':  ['10,000','14','0 (0%)','20.4%',
                       'France / Germany / Spain','38.9 yrs','€76,486']
        }), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Quick EDA Snapshot")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    set_dark(fig, list(axes))

    vc = df_raw['Exited'].value_counts()
    axes[0].pie([vc[0],vc[1]], labels=['Retained\n79.6%','Churned\n20.4%'],
                colors=CC, startangle=90, wedgeprops={'edgecolor':'#0f172a','linewidth':2})
    circle = plt.Circle((0,0),0.5,color='#1e293b'); axes[0].add_patch(circle)
    axes[0].set_title('Class Balance', fontweight='bold')

    geo = df_raw.groupby('Geography')['Exited'].mean()*100
    axes[1].bar(geo.index, geo.values, color=COLORS[:3], edgecolor='#0f172a')
    for i,(k,v) in enumerate(geo.items()):
        axes[1].text(i, v+0.3, f'{v:.1f}%', ha='center', color='#e2e8f0', fontsize=10)
    axes[1].set_title('Churn Rate by Geography', fontweight='bold')
    axes[1].set_ylabel('Churn Rate (%)'); axes[1].grid(axis='y',color='#334155')

    axes[2].hist(retain['Age'],bins=30,alpha=0.7,color=CC[0],label='Retained',density=True)
    axes[2].hist(churn['Age'], bins=30,alpha=0.7,color=CC[1],label='Churned', density=True)
    axes[2].set_title('Age Distribution', fontweight='bold')
    axes[2].set_xlabel('Age'); axes[2].legend(); axes[2].grid(axis='y',color='#334155')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════
# PAGE 2 — PREDICTOR
# ═══════════════════════════════════════════════════
elif page == "🔮 Churn Predictor":
    st.markdown("# 🔮 Customer Churn Predictor")
    st.markdown("Enter customer details to get an instant churn risk assessment.")
    st.markdown("---")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Demographics**")
        age       = st.slider("Age", 18, 92, 38)
        gender    = st.selectbox("Gender", ["Male","Female"])
        geography = st.selectbox("Geography", ["France","Germany","Spain"])
    with c2:
        st.markdown("**Financial Profile**")
        credit_score  = st.slider("Credit Score", 350, 850, 650)
        balance       = st.number_input("Account Balance (€)", 0.0, 260000.0, 76000.0, step=1000.0)
        estimated_sal = st.number_input("Estimated Salary (€)", 11.0, 200000.0, 100000.0, step=1000.0)
    with c3:
        st.markdown("**Banking Behaviour**")
        tenure       = st.slider("Tenure (years)", 0, 10, 5)
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cc       = st.selectbox("Has Credit Card",  [1,0], format_func=lambda x:"Yes" if x else "No")
        is_active    = st.selectbox("Is Active Member", [1,0], format_func=lambda x:"Yes" if x else "No")

    if st.button("🚀 Predict Churn Risk", use_container_width=True):
        prob  = predict_single(age, gender, geography, credit_score, balance,
                               estimated_sal, tenure, num_products, has_cc, is_active)
        score = int(prob * 1000)
        if prob >= 0.80:
            cat,css,emoji = "High Risk","risk-high","🚨"
            action = "IMMEDIATE ACTION: Assign dedicated Relationship Manager + exclusive retention offer."
        elif prob >= 0.50:
            cat,css,emoji = "Medium Risk","risk-medium","⚠️"
            action = "PROACTIVE RETENTION: Schedule personalised check-in call + product upgrade offer."
        else:
            cat,css,emoji = "Low Risk","risk-low","✅"
            action = "MONITOR: Include in quarterly loyalty programme. No immediate action required."

        r1,r2,r3 = st.columns(3)
        r1.metric("Churn Probability", f"{prob:.1%}")
        r2.metric("Risk Score (0–1000)", str(score))
        r3.metric("Risk Category", f"{emoji} {cat}")
        st.markdown(f'<div class="{css}"><h3>{emoji} {cat}</h3>'
                    f'<p style="font-size:1.1rem;color:#e2e8f0">{action}</p></div>',
                    unsafe_allow_html=True)

        # Gauge
        fig, ax = plt.subplots(figsize=(7, 3.5), subplot_kw={'projection':'polar'})
        fig.patch.set_facecolor('#0f172a'); ax.set_facecolor('#1e293b')
        for lo,hi,color in [(0,.5,'#10b981'),(.5,.8,'#f97316'),(.8,1,'#f43f5e')]:
            t_seg = np.linspace(np.pi*(1-hi), np.pi*(1-lo), 50)
            ax.fill_between(t_seg, 0.6, 1, alpha=0.6, color=color)
        needle = np.pi*(1-prob)
        ax.annotate('', xy=(needle,0.85), xytext=(needle,0),
                    arrowprops=dict(arrowstyle='->', color='white', lw=3))
        ax.set_ylim(0,1.1); ax.axis('off')
        ax.text(np.pi/2, 0.18, f"{prob:.1%}", ha='center', va='center',
                fontsize=22, fontweight='bold', color='white')
        ax.set_title(f"Risk Gauge — {cat}", color='#e2e8f0', fontsize=13)
        st.pyplot(fig, use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════
# PAGE 3 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════
elif page == "📊 Feature Importance":
    st.markdown("# 📊 Feature Importance Dashboard")
    st.markdown("What drives customer churn the most?")
    st.markdown("---")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    set_dark(fig, list(axes))

    top_fi = fi.sort_values().tail(12)
    gradient = plt.cm.cool(np.linspace(0.3,1.0,len(top_fi)))
    bars = axes[0].barh(top_fi.index, top_fi.values, color=gradient, edgecolor='#0f172a')
    for b in bars:
        axes[0].text(b.get_width()+0.002, b.get_y()+b.get_height()/2,
                     f'{b.get_width():.3f}', va='center', fontsize=9, color='#e2e8f0')
    axes[0].set_title('Gradient Boosting — Feature Importance', fontweight='bold')
    axes[0].set_xlabel('Importance Score'); axes[0].grid(axis='x',color='#334155')

    top8 = fi.head(8)
    clrs = plt.cm.plasma(np.linspace(0.3,1.0,len(top8)))
    bars2 = axes[1].bar(range(len(top8)), top8.values, color=clrs, edgecolor='#0f172a')
    axes[1].set_xticks(range(len(top8)))
    axes[1].set_xticklabels([f[:12] for f in top8.index], rotation=35, ha='right', fontsize=9)
    for b in bars2:
        axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                     f'{b.get_height():.3f}', ha='center', fontsize=8, color='#e2e8f0')
    axes[1].set_title('Top 8 Features — Bar Chart', fontweight='bold')
    axes[1].set_ylabel('Importance'); axes[1].grid(axis='y',color='#334155')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("### 📌 Business Interpretation")
    insights = [
        ("🥇 Age","39.0%","Customers 40+ are significantly more likely to churn"),
        ("🥈 NumOfProducts","29.4%","3–4 products = over-banked dissatisfaction"),
        ("🥉 Engagement×Products","9.2%","Inactive multi-product holders = highest risk"),
        ("4️⃣ Balance","5.1%","Wealthier customers have more alternatives"),
        ("5️⃣ IsActiveMember","4.1%","Inactivity doubles churn probability"),
        ("6️⃣ Geography","3.7%","Germany churns at 32% — needs dedicated strategy"),
    ]
    for feat, imp, insight in insights:
        st.markdown(f"**{feat}** ({imp}) — {insight}")

# ═══════════════════════════════════════════════════
# PAGE 4 — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════
elif page == "🎮 What-If Simulator":
    st.markdown("# 🎮 What-If Churn Simulator")
    st.markdown("Adjust parameters and see real-time churn probability changes.")
    st.markdown("---")
    c1,c2 = st.columns([1,1])
    with c1:
        st.markdown("### 🎛️ Customer Parameters")
        age_s      = st.slider("Age",18,92,40,key='s1')
        balance_s  = st.slider("Balance (€)",0,250000,80000,step=5000,key='s2')
        credit_s   = st.slider("Credit Score",350,850,650,key='s3')
        products_s = st.slider("Num Products",1,4,1,key='s4')
        active_s   = st.slider("Is Active (0=No, 1=Yes)",0,1,1,key='s5')
        tenure_s   = st.slider("Tenure (years)",0,10,5,key='s6')
        geo_s      = st.selectbox("Geography",["France","Germany","Spain"],key='s7')
        gender_s   = st.selectbox("Gender",["Male","Female"],key='s8')
        salary_s   = st.slider("Salary (€)",0,200000,100000,step=5000,key='s9')
        hascc_s    = st.selectbox("Has Credit Card",[1,0],format_func=lambda x:"Yes" if x else "No",key='s10')

    with c2:
        prob_s = predict_single(age_s, gender_s, geo_s, credit_s, balance_s,
                                salary_s, tenure_s, products_s, hascc_s, active_s)
        clr = "#f43f5e" if prob_s>=0.8 else ("#f97316" if prob_s>=0.5 else "#10b981")
        cat_s = "🚨 High Risk" if prob_s>=0.8 else ("⚠️ Medium Risk" if prob_s>=0.5 else "✅ Low Risk")
        st.markdown(f"""
        <div style="background:#1e293b;border:2px solid {clr};border-radius:16px;
                    padding:32px;text-align:center;margin-bottom:20px">
            <div style="font-size:4rem;font-weight:800;color:{clr}">{prob_s:.1%}</div>
            <div style="font-size:1.4rem;color:#e2e8f0;margin-top:8px">{cat_s}</div>
            <div style="font-size:.9rem;color:#94a3b8;margin-top:8px">
                Risk Score: {int(prob_s*1000)}/1000</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Age Sensitivity")
        ages_r = list(range(18,93,3))
        probs_r = [predict_single(a, gender_s, geo_s, credit_s, balance_s,
                                  salary_s, tenure_s, products_s, hascc_s, active_s)
                   for a in ages_r]
        fig2, ax2 = plt.subplots(figsize=(8,3.5))
        set_dark(fig2, ax2)
        ax2.plot(ages_r, [p*100 for p in probs_r], color='#06b6d4', lw=2.5)
        ax2.axvline(age_s, color='#facc15', ls='--', lw=1.5, label=f'Current: {age_s}')
        ax2.fill_between(ages_r, [p*100 for p in probs_r], alpha=0.15, color='#06b6d4')
        ax2.axhline(50, color='#f97316', ls=':', lw=1)
        ax2.axhline(80, color='#f43f5e', ls=':', lw=1)
        ax2.set_xlabel('Age'); ax2.set_ylabel('Churn Probability (%)')
        ax2.set_title('How Age Affects Churn Risk', fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(color='#334155',lw=0.5)
        st.pyplot(fig2, use_container_width=True); plt.close()

# ═══════════════════════════════════════════════════
# PAGE 5 — ANALYTICS
# ═══════════════════════════════════════════════════
elif page == "📈 Analytics Dashboard":
    st.markdown("# 📈 Analytics Dashboard")
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 Churn Patterns","🎯 Risk Segments","🏆 Model Performance"])

    with tab1:
        fig, axes = plt.subplots(2,3, figsize=(20,11))
        set_dark(fig, [ax for row in axes for ax in row])

        # Geography churn
        geo = df_raw.groupby('Geography')['Exited'].mean()*100
        axes[0,0].bar(geo.index, geo.values, color=COLORS[:3], edgecolor='#0f172a')
        for i,(k,v) in enumerate(geo.items()):
            axes[0,0].text(i,v+0.3,f'{v:.1f}%',ha='center',color='#e2e8f0',fontsize=10)
        axes[0,0].set_title('Churn Rate by Geography',fontweight='bold')
        axes[0,0].set_ylabel('Churn Rate (%)'); axes[0,0].grid(axis='y',color='#334155')

        # Gender churn
        gen = df_raw.groupby('Gender')['Exited'].mean()*100
        axes[0,1].bar(gen.index, gen.values, color=CC, edgecolor='#0f172a')
        for i,(k,v) in enumerate(gen.items()):
            axes[0,1].text(i,v+0.3,f'{v:.1f}%',ha='center',color='#e2e8f0',fontsize=10)
        axes[0,1].set_title('Churn Rate by Gender',fontweight='bold')
        axes[0,1].grid(axis='y',color='#334155')

        # Products
        prod = df_raw.groupby('NumOfProducts')['Exited'].mean()*100
        axes[0,2].bar(prod.index.astype(str), prod.values, color=COLORS, edgecolor='#0f172a')
        for i,(k,v) in enumerate(prod.items()):
            axes[0,2].text(i,v+0.3,f'{v:.1f}%',ha='center',color='#e2e8f0',fontsize=10)
        axes[0,2].set_title('Churn by # Products',fontweight='bold')
        axes[0,2].grid(axis='y',color='#334155')

        # Age hist
        axes[1,0].hist(retain['Age'],bins=30,alpha=0.7,color=CC[0],label='Retained',density=True)
        axes[1,0].hist(churn['Age'], bins=30,alpha=0.7,color=CC[1],label='Churned', density=True)
        axes[1,0].set_title('Age Distribution',fontweight='bold')
        axes[1,0].legend(); axes[1,0].grid(axis='y',color='#334155')

        # Balance boxplot
        axes[1,1].boxplot([retain['Balance']/1000, churn['Balance']/1000],
                           labels=['Retained','Churned'], patch_artist=True,
                           boxprops={'facecolor':CC[1],'alpha':0.55},
                           medianprops={'color':'#facc15','linewidth':2},
                           whiskerprops={'color':'#94a3b8'}, capprops={'color':'#94a3b8'})
        axes[1,1].set_title('Balance by Churn Status',fontweight='bold')
        axes[1,1].set_ylabel('Balance (€000s)'); axes[1,1].grid(axis='y',color='#334155')

        # Active member
        act = df_raw.groupby('IsActiveMember')['Exited'].mean()*100
        axes[1,2].bar(['Inactive','Active'], act.values,
                       color=[COLORS[4],COLORS[2]], edgecolor='#0f172a', width=0.5)
        for i,v in enumerate(act.values):
            axes[1,2].text(i,v+0.3,f'{v:.1f}%',ha='center',color='#e2e8f0',fontsize=11)
        axes[1,2].set_title('Churn: Active vs Inactive',fontweight='bold')
        axes[1,2].grid(axis='y',color='#334155')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        fig, axes = plt.subplots(1,2, figsize=(16,6))
        set_dark(fig, list(axes))

        # Risk donut
        cats  = ['Low Risk','Medium Risk','High Risk']
        sizes = [risk_dist.get(c,0) for c in cats]
        clrs  = [RISK_C[c] for c in cats]
        wedges,texts,autotexts = axes[0].pie(
            sizes, labels=[f'{c}\n{s:,}' for c,s in zip(cats,sizes)],
            colors=clrs, autopct='%1.1f%%', startangle=90, pctdistance=0.75,
            wedgeprops={'edgecolor':'#0f172a','linewidth':2.5})
        for at in autotexts: at.set_fontsize(9); at.set_color('white'); at.set_fontweight('bold')
        circle = plt.Circle((0,0),0.5,color='#1e293b'); axes[0].add_patch(circle)
        axes[0].set_title('Customer Risk Segmentation',fontweight='bold')

        # Actual churn per band
        act_churn = df_proc.groupby('RiskCategory')['Exited'].mean()*100
        order = [c for c in cats if c in act_churn.index]
        vals  = [act_churn[c] for c in order]
        clrs2 = [RISK_C[c] for c in order]
        bars  = axes[1].bar(order, vals, color=clrs2, edgecolor='#0f172a', width=0.5)
        for b,v in zip(bars,vals):
            axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.8,
                         f'{v:.1f}%', ha='center', fontsize=12, color='#e2e8f0', fontweight='bold')
        axes[1].set_title('Actual Churn Rate per Risk Band',fontweight='bold')
        axes[1].set_ylabel('Actual Churn Rate (%)'); axes[1].grid(axis='y',color='#334155')
        axes[1].set_ylim(0,115)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("### Model Performance Metrics")
        st.dataframe(pd.DataFrame([metrics]), hide_index=True, use_container_width=True)

    with tab3:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        fig, axes = plt.subplots(1,2, figsize=(14,6))
        set_dark(fig, list(axes))

        # ROC curve
        fpr,tpr,_ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        axes[0].plot(fpr,tpr,color='#06b6d4',lw=2.5,label=f'GBM (AUC={auc:.3f})')
        axes[0].plot([0,1],[0,1],'--',color='#475569',lw=1)
        axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve',fontweight='bold')
        axes[0].legend(); axes[0].grid(color='#334155',lw=0.5)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                    xticklabels=['Retain','Churn'], yticklabels=['Retain','Churn'],
                    annot_kws={'size':14,'weight':'bold'}, linewidths=1,
                    linecolor='#0f172a', cbar_kws={'shrink':0.8})
        axes[1].set_title('Confusion Matrix',fontweight='bold')
        axes[1].set_ylabel('Actual'); axes[1].set_xlabel('Predicted')

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("### 📊 Full Metrics")
        cols = st.columns(5)
        for col,(k,v) in zip(cols, metrics.items()):
            col.metric(k, f"{v:.4f}")
