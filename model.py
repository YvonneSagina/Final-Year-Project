import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, precision_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler

# LOAD FEATURE MATRIX 

def load_features():
    print("Loading feature matrix...")
    df = pd.read_csv('data/processed/feature_matrix.csv')
    print(f"Loaded {len(df)} records")
    print(f"Fraud: {df['fraud_label'].sum()} | Legitimate: {len(df) - df['fraud_label'].sum()}")
    return df

# PREPARE DATA FOR TRAINING 

def prepare_data(df):
    print("Preparing data for training...")

    # Define features and target
    X = df[['price_deviation_ratio',
            'repeat_award_frequency',
            'vendor_registration_age']]
    y = df['fraud_label']

    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)} records")
    print(f"Test set: {len(X_test)} records")

    return X_train, X_test, y_train, y_test

# TRAIN RANDOM FOREST MODEL 

def train_model(X_train, y_train):
    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,       # 100 decision trees
        max_depth=10,           # Maximum depth of each tree
        min_samples_split=5,    # Minimum samples to split a node
        class_weight='balanced',# Handles class imbalance automatically
        random_state=42         # For reproducibility
    )

    model.fit(X_train, y_train)
    print("Model training complete")
    return model

# EVALUATE MODEL 

def evaluate_model(model, X_test, y_test):
    print("\n── MODEL EVALUATION ─────────────────────────────")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Print metrics
    print(f"Precision:  {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:     {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:   {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:    {roc_auc_score(y_test, y_prob):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Legitimate', 'Fraudulent']))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred, y_prob

# FEATURE IMPORTANCE 

def print_feature_importance(model, feature_names):
    print("\n── FEATURE IMPORTANCE ───────────────────────────")
    importances = model.feature_importances_
    for name, score in sorted(zip(feature_names, importances),
                               key=lambda x: x[1], reverse=True):
        print(f"{name:<30} {score:.4f}  ")

# SAVE FRAUD SCORES TO DATABASE 

def save_scores(df, y_prob, y_pred):
    print("\nSaving fraud scores to database...")

    # Load original synthesized dataset to get full record details
    full_df = pd.read_csv('data/processed/synthesized_dataset.csv')

    # Create scores dataframe (FIXED: Now mapping directly to all OCIDs)
    scores_df = pd.DataFrame({
        'main_ocid': df['main_ocid'].values,
        'fraud_probability': y_prob,
        'risk_flag': y_pred,
        'risk_level': pd.cut(
            y_prob,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    })

    # Save to SQLite
    conn = sqlite3.connect('models/procurement_fraud.db')
    scores_df.to_sql('fraud_scores', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved {len(scores_df)} fraud scores to database")

    # Also save full scored dataset to CSV
    scores_df.to_csv('data/processed/fraud_scores.csv', index=False)
    print("Saved to data/processed/fraud_scores.csv")

# SAVE MODEL 

def save_model(model):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.pkl')
    print("Model saved to models/random_forest_model.pkl")

#  MAIN 

if __name__ == '__main__':
    feature_names = ['price_deviation_ratio',
                     'repeat_award_frequency',
                     'vendor_registration_age']

    df = load_features()
    
    # 1. Train and Evaluate as normal
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    print_feature_importance(model, feature_names)
    
    # 2. THE FIX: Generate predictions for the ENTIRE dataset
    print("\nScoring full dataset for dashboard...")
    X_all = df[feature_names]
    y_pred_all = model.predict(X_all)
    y_prob_all = model.predict_proba(X_all)[:, 1]
    
    # 3. Save the full batch of scores
    save_scores(df, y_prob_all, y_pred_all)
    save_model(model)
    
    print("\nModel training and evaluation complete!")