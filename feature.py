import pandas as pd
import numpy as np
import sqlite3

# LOAD SYNTHESIZED DATASET 

def load_data():
    print("Loading synthesized dataset...")
    df = pd.read_csv('Data/Processed/synthesized_dataset.csv')
    print(f"Loaded {len(df)} records")
    return df

# ENGINEER FEATURES 

def engineer_features(df):
    print("Engineering features...")

    # PRICE DEVIATION RATIO 
    df['contract_value'] = df['contract_value'].replace(0, 1) 
    df['price_deviation_ratio'] = df['award_value'] / df['contract_value']
    df['price_deviation_ratio'] = df['price_deviation_ratio'].round(2)
    sus_records = df[df['price_deviation_ratio'].between(2.5, 5.0)]
    print(f"Price deviation ratio — found {len(sus_records)} records inflated between 2.5x and 5.0x")

    # REPEAT AWARD FREQUENCY 
    supplier_counts = df['supplier_name'].value_counts()
    df['repeat_award_frequency'] = df['supplier_name'].map(supplier_counts)
    print(f"Repeat award frequency — max appearances: {df['repeat_award_frequency'].max()}")

    # VENDOR REGISTRATION AGE 
    df['vendor_registration_date'] = pd.to_datetime(
        df['vendor_registration_date'], errors='coerce', utc=True).dt.tz_localize(None)

    df['contractPeriod_startDate'] = pd.to_datetime(
        df['contractPeriod_startDate'], errors='coerce', utc=True).dt.tz_localize(None)

    df['vendor_registration_age'] = (
        df['contractPeriod_startDate'] - df['vendor_registration_date']).dt.days

    # Fill any missing registration age with the median
    median_age = df['vendor_registration_age'].median()
    df['vendor_registration_age'] = df['vendor_registration_age'].fillna(median_age)
    print(f"Vendor registration age — median days: {median_age:.0f}")

    return df

# SELECT FINAL FEATURE MATRIX 

def select_features(df):
    print("Selecting final feature matrix...")

    feature_cols = [
        'main_ocid',
        'price_deviation_ratio',
        'repeat_award_frequency',
        'vendor_registration_age',
        'fraud_label'
    ]

    df_features = df[feature_cols].copy()

    # Drop any remaining rows with missing values
    df_features.dropna(inplace=True)
    df_features.reset_index(drop=True, inplace=True)

    print(f"Final feature matrix: {len(df_features)} records")
    print(f"Fraud records: {df_features['fraud_label'].sum()}")
    print(f"Legitimate records: {len(df_features) - df_features['fraud_label'].sum()}")

    return df_features

# SAVE FEATURE MATRIX 

def save_features(df_features):
    print("Saving feature matrix...")

    # Save to CSV
    df_features.to_csv('Data/Processed/feature_matrix.csv', index=False)
    print("Saved to Data/Processed/feature_matrix.csv")

    # Save to SQLite
    conn = sqlite3.connect('models/procurement_fraud.db')
    df_features.to_sql('engineered_features', conn,
                       if_exists='replace', index=False)
    conn.close()
    print("Saved to SQLite database")

# PRINT FEATURE SUMMARY 

def print_summary(df_features):
    print("\n── FEATURE SUMMARY ──────────────────────────────")
    print(df_features[['price_deviation_ratio',
                        'repeat_award_frequency',
                        'vendor_registration_age',
                        'fraud_label']].describe())
    print("\nSample records:")
    print(df_features.head(10).to_string())

# MAIN 

if __name__ == '__main__':
    df = load_data()
    df = engineer_features(df)
    df_features = select_features(df)
    save_features(df_features)
    print_summary(df_features)
    print("\nFeature engineering complete!")