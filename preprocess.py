"""
PHASE 3 & 4: DATA PREPROCESSING + FEATURE ENGINEERING
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle, os
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess(path='/home/claude/Bank_Customer_Churn_Project/data/European_Bank.csv'):
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape}")

    # ── Phase 3: Preprocessing ────────────────────────────────────────────────
    # Drop irrelevant columns
    drop_cols = ['Year', 'CustomerId', 'Surname']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Remove duplicates (none expected but best practice)
    df.drop_duplicates(inplace=True)

    # Encode categorical columns
    le_geo = LabelEncoder()
    le_gen = LabelEncoder()
    df['Geography_enc'] = le_geo.fit_transform(df['Geography'])
    df['Gender_enc']    = le_gen.fit_transform(df['Gender'])

    # ── Phase 4: Feature Engineering ─────────────────────────────────────────
    # 1. BalanceSalaryRatio — relative wealth indicator
    df['BalanceSalaryRatio'] = np.where(
        df['EstimatedSalary'] > 0,
        df['Balance'] / (df['EstimatedSalary'] + 1),
        0
    )

    # 2. ProductDensity — products per tenure year (engagement density)
    df['ProductDensity'] = df['NumOfProducts'] / (df['Tenure'] + 1)

    # 3. EngagementProductInteraction — active member × products (loyalty proxy)
    df['EngagementProductInteraction'] = df['IsActiveMember'] * df['NumOfProducts']

    # 4. AgeTenureInteraction — lifestage-loyalty combination
    df['AgeTenureInteraction'] = df['Age'] * df['Tenure']

    # 5. IsZeroBalance — binary flag (30% of customers have 0 balance)
    df['IsZeroBalance'] = (df['Balance'] == 0).astype(int)

    # 6. AgeGroup — ordinal bucket
    df['AgeGroup'] = pd.cut(df['Age'], bins=[17,30,40,50,60,100],
                             labels=[0,1,2,3,4]).astype(int)

    # 7. CreditRisk — inverted credit score band
    df['CreditRisk'] = pd.cut(df['CreditScore'],
                               bins=[349,579,669,739,799,851],
                               labels=[4,3,2,1,0]).astype(int)

    print("Feature engineering complete. New features:")
    new_feats = ['BalanceSalaryRatio','ProductDensity','EngagementProductInteraction',
                 'AgeTenureInteraction','IsZeroBalance','AgeGroup','CreditRisk']
    for f in new_feats:
        print(f"  + {f}: mean={df[f].mean():.3f}, std={df[f].std():.3f}")

    # ── Prepare ML features ───────────────────────────────────────────────────
    feature_cols = [
        'CreditScore','Geography_enc','Gender_enc','Age','Tenure','Balance',
        'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary',
        'BalanceSalaryRatio','ProductDensity','EngagementProductInteraction',
        'AgeTenureInteraction','IsZeroBalance','AgeGroup','CreditRisk'
    ]

    X = df[feature_cols].copy()
    y = df['Exited'].copy()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # Train-test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean()*100:.1f}%")
    print(f"Test  churn rate: {y_test.mean()*100:.1f}%")

    # Save artefacts
    os.makedirs('/home/claude/Bank_Customer_Churn_Project/models', exist_ok=True)
    with open('/home/claude/Bank_Customer_Churn_Project/models/scaler.pkl','wb') as f:
        pickle.dump(scaler, f)
    with open('/home/claude/Bank_Customer_Churn_Project/models/label_encoders.pkl','wb') as f:
        pickle.dump({'geo': le_geo, 'gen': le_gen}, f)
    with open('/home/claude/Bank_Customer_Churn_Project/models/feature_cols.pkl','wb') as f:
        pickle.dump(feature_cols, f)

    df.to_csv('/home/claude/Bank_Customer_Churn_Project/data/processed.csv', index=False)
    print("\n✅ Preprocessed data + artefacts saved.")

    return X_train, X_test, y_train, y_test, feature_cols, df

if __name__ == '__main__':
    load_and_preprocess()
