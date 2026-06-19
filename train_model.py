"""
PHASES 5 & 6: MODEL TRAINING, HYPERPARAMETER TUNING, AND EVALUATION
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, sys, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)

# Import preprocessing
sys.path.insert(0, '/home/claude/Bank_Customer_Churn_Project')
from preprocess import load_and_preprocess

plt.rcParams.update({
    'figure.facecolor':'#0f172a','axes.facecolor':'#1e293b',
    'axes.edgecolor':'#334155','axes.labelcolor':'#e2e8f0',
    'xtick.color':'#94a3b8','ytick.color':'#94a3b8',
    'text.color':'#e2e8f0','grid.color':'#334155','grid.linewidth':0.6,
    'font.family':'DejaVu Sans','font.size':10,
})
COLORS = ['#06b6d4','#f97316','#10b981','#8b5cf6','#f43f5e']

X_train, X_test, y_train, y_test, feature_cols, df_full = load_and_preprocess()

print("\n" + "="*65)
print("  PHASE 5: MODEL TRAINING WITH HYPERPARAMETER TUNING")
print("="*65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Model definitions with param grids ────────────────────────────────────────
models_config = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs','liblinear']}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [5,8,12,None], 'min_samples_split':[10,20,50]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {'n_estimators':[100,200], 'max_depth':[8,12,None], 'min_samples_leaf':[1,5]}
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators':[100,200], 'learning_rate':[0.05,0.1], 'max_depth':[3,5]}
    },
}

trained_models = {}
results = []

for name, cfg in models_config.items():
    print(f"\n🔧 Training: {name}")
    gs = GridSearchCV(cfg['model'], cfg['params'], cv=cv, scoring='roc_auc',
                      n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    trained_models[name] = best

    y_pred  = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cv_auc = cross_val_score(best, X_train, y_train, cv=cv, scoring='roc_auc').mean()

    results.append({'Model': name, 'Accuracy': acc, 'Precision': pre,
                    'Recall': rec, 'F1': f1, 'ROC-AUC': auc, 'CV-AUC': cv_auc,
                    'Best Params': gs.best_params_})

    print(f"  Best params: {gs.best_params_}")
    print(f"  Acc={acc:.4f} | Pre={pre:.4f} | Rec={rec:.4f} | F1={f1:.4f} | AUC={auc:.4f} | CV-AUC={cv_auc:.4f}")

# Save all models
for name, model in trained_models.items():
    fname = name.replace(' ','_').lower() + '.pkl'
    with open(f'/home/claude/Bank_Customer_Churn_Project/models/{fname}', 'wb') as f:
        pickle.dump(model, f)

# ── PHASE 6: Evaluation ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PHASE 6: MODEL EVALUATION & COMPARISON")
print("="*65)

results_df = pd.DataFrame(results)
print("\n📊 Model Comparison Table:")
print(results_df[['Model','Accuracy','Precision','Recall','F1','ROC-AUC','CV-AUC']].round(4).to_string(index=False))

best_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
best_model = trained_models[best_name]
print(f"\n🏆 Best Model: {best_name} (AUC = {results_df.loc[results_df['ROC-AUC'].idxmax(),'ROC-AUC']:.4f})")

# Save best model separately
with open('/home/claude/Bank_Customer_Churn_Project/models/best_model.pkl','wb') as f:
    pickle.dump(best_model, f)
results_df.to_csv('/home/claude/Bank_Customer_Churn_Project/reports/model_results.csv', index=False)

# ── Evaluation Charts ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.suptitle('Phase 6 · Model Evaluation Dashboard', fontsize=18, fontweight='bold', color='#e2e8f0')

gs_layout = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.35)

# 1. Metric comparison bars
metrics = ['Accuracy','Precision','Recall','F1','ROC-AUC']
ax1 = fig.add_subplot(gs_layout[0, :2])
x = np.arange(len(metrics))
w = 0.18
for i, (_, row) in enumerate(results_df.iterrows()):
    vals = [row[m] for m in metrics]
    ax1.bar(x + i*w, vals, w, label=row['Model'], color=COLORS[i], edgecolor='#0f172a', alpha=0.9)
ax1.set_xticks(x + w*1.5)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.set_ylabel('Score'); ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=9, ncol=2); ax1.grid(axis='y')
ax1.set_title('Model Metrics Comparison', fontsize=13, fontweight='bold')

# 2. ROC Curves
ax2 = fig.add_subplot(gs_layout[0, 2])
for i, (name, model) in enumerate(trained_models.items()):
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, color=COLORS[i], lw=2, label=f'{name[:12]} ({auc:.3f})')
ax2.plot([0,1],[0,1],'--', color='#475569', lw=1)
ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves', fontsize=13, fontweight='bold')
ax2.legend(fontsize=8); ax2.grid()

# 3–6. Confusion matrices for each model
positions = [gs_layout[1,0], gs_layout[1,1], gs_layout[1,2], gs_layout[2,0]]
for i, (name, model) in enumerate(trained_models.items()):
    ax = fig.add_subplot(positions[i])
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                linewidths=1, linecolor='#0f172a',
                xticklabels=['Retain','Churn'], yticklabels=['Retain','Churn'],
                annot_kws={'size':12, 'weight':'bold'})
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')

# 7. Feature importance (best model)
ax7 = fig.add_subplot(gs_layout[2, 1:])
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fi_top = fi.tail(12)
    bars = ax7.barh(fi_top.index, fi_top.values, color='#06b6d4', edgecolor='#0f172a', alpha=0.85)
    ax7.set_title(f'Feature Importance — {best_name}', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Importance Score'); ax7.grid(axis='x')

plt.savefig('/home/claude/Bank_Customer_Churn_Project/reports/phase6_evaluation.png',
            dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("\n✅ Phase 5–6 complete. Charts saved → reports/phase6_evaluation.png")
print(f"✅ Best model '{best_name}' saved → models/best_model.pkl")
