import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import timedelta

# LOAD AND MERGE CSV FILES 

def load_and_merge():
    print("Loading CSV files...")

    main = pd.read_csv('Data/Raw/2024/main.csv')
    awards = pd.read_csv('Data/Raw/2024/awards.csv')
    awards_suppliers = pd.read_csv('Data/Raw/2024/awards_suppliers.csv')
    parties = pd.read_csv('Data/Raw/2024/parties.csv')
    contracts = pd.read_csv('Data/Raw/2024/contracts.csv')

    print(f"Main: {len(main)} records")
    print(f"Awards: {len(awards)} records")
    print(f"Awards Suppliers: {len(awards_suppliers)} records")
    print(f"Parties: {len(parties)} records")
    print(f"Contracts: {len(contracts)} records")

    # Merge main with awards on main_ocid. So ocid data that matches main_ocid in awards will be merged.
    df = pd.merge(main, awards, left_on='ocid', right_on='main_ocid', how='inner', suffixes=('_main', '_award'))

    # Merge with awards_suppliers to get supplier name
    df = pd.merge(df, awards_suppliers[['main_ocid', 'name']],
                  on='main_ocid', how='left')
    df.rename(columns={'name': 'supplier_name'}, inplace=True)

    # Merge with contracts to get contract value
    df = pd.merge(df, contracts[['main_ocid', 'value_amount']],
                  on='main_ocid', how='left', suffixes=('', '_contract'))
    df.rename(columns={'value_amount_contract': 'contract_value'}, inplace=True)

    print(f"Merged dataset: {len(df)} records")
    return df

# CLEAN THE DATA 

def clean_data(df):
    print("Cleaning data...")

    cols_needed = [
        'main_ocid', 'buyer_name', 'supplier_name',
        'tender_procurementMethod', 'date',
        'value_amount', 'contract_value',
        'contractPeriod_startDate', 'contractPeriod_endDate'
    ]

    # Only keep columns that exist
    cols_needed = [c for c in cols_needed if c in df.columns]
    df = df[cols_needed].copy()

    # Rename for clarity
    df.rename(columns={
        'tender_procurementMethod': 'procurement_method',
        'date': 'tender_date',
        'value_amount': 'award_value'
    }, inplace=True)

    # Convert date columns
    for col in ['tender_date', 'contractPeriod_startDate', 'contractPeriod_endDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
            df[col] = df[col].dt.tz_localize(None)

    # Drop rows with missing award value 
    df.dropna(subset=['award_value'], inplace=True)
    # Identify which rows are missing a supplier name
    missing_mask = df['supplier_name'].isnull()
    
    # Give each missing supplier a unique numbered placeholder
    df.loc[missing_mask, 'supplier_name'] = [f'Unknown_Supplier_{i}' for i in range(missing_mask.sum())]
    
    df['contract_value'] = df['contract_value'].fillna(df['award_value'])
    df['procurement_method'] = df['procurement_method'].fillna('open')

    # Remove duplicates
    df.drop_duplicates(subset=['main_ocid'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Clean dataset: {len(df)} records")
    return df

# GENERATE VENDOR REGISTRATION DATES 

def generate_registration_dates(df):
    print("Generating vendor registration dates...")

    np.random.seed(42)
    registration_dates = []

    for _, row in df.iterrows():
        ref_date = row['contractPeriod_startDate'] if pd.notnull(
            row.get('contractPeriod_startDate')) else pd.Timestamp('2024-01-01')

        # Default: legitimate vendor registered 1-10 years before contract
        days_before = np.random.randint(365, 3650)
        reg_date = ref_date - timedelta(days=int(days_before))
        registration_dates.append(reg_date)

    df['vendor_registration_date'] = registration_dates
    return df

# INJECT FRAUD SIGNATURES 

def inject_fraud(df, fraud_ratio=0.20):
    print(f"Injecting fraud into {int(fraud_ratio * 100)}% of records...")

    np.random.seed(42)
    df['fraud_label'] = 0
    n_fraud = int(len(df) * fraud_ratio)
    fraud_indices = np.random.choice(df.index, size=n_fraud, replace=False)

    # Split fraud indices into three equal groups
    group_size = n_fraud // 3
    price_indices = fraud_indices[:group_size]
    repeat_indices = fraud_indices[group_size:group_size * 2]
    ghost_indices = fraud_indices[group_size * 2:]

    # Fraud Type 1: Price Inflation 
    for idx in price_indices:
        multiplier = np.random.uniform(2.5, 5.0)
        df.at[idx, 'award_value'] = df.at[idx, 'award_value'] * multiplier
        df.at[idx, 'fraud_label'] = 1

    # Fraud Type 2: Bid Rigging (repeat supplier) 
    # Find the top 3 most frequent suppliers already in the dataset
    top_suppliers = df['supplier_name'].value_counts().head(3).index.tolist()

    for i, idx in enumerate(repeat_indices):
        df.at[idx, 'supplier_name'] = top_suppliers[i % len(top_suppliers)]
        df.at[idx, 'fraud_label'] = 1

    # Fraud Type 3: Ghost Vendor
    for idx in ghost_indices:
        ref_date = df.at[idx, 'contractPeriod_startDate']
        if pd.isnull(ref_date):
            ref_date = pd.Timestamp('2024-06-01')
        days_before = np.random.randint(1, 30)
        df.at[idx, 'vendor_registration_date'] = ref_date - timedelta(days=int(days_before))
        df.at[idx, 'fraud_label'] = 1

    print(f"Fraud records: {df['fraud_label'].sum()}")
    print(f"Legitimate records: {len(df) - df['fraud_label'].sum()}")
    return df

# SAVE TO CSV AND SQLITE 

def save_data(df):
    print("Saving synthesized dataset...")

    # Save to CSV
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/synthesized_dataset.csv', index=False)
    print("Saved to data/processed/synthesized_dataset.csv")

    # Save to SQLite
    conn = sqlite3.connect('models/procurement_fraud.db')
    df.to_sql('procurement_records', conn, if_exists='replace', index=False)
    conn.close()
    print("Saved to SQLite database")

# MAIN 

if __name__ == '__main__':
    df = load_and_merge()
    df = clean_data(df)
    df = generate_registration_dates(df)
    df = inject_fraud(df)
    save_data(df)
    print("\nData synthesis complete!")
    print(df[['main_ocid', 'supplier_name', 'award_value',
              'vendor_registration_date', 'fraud_label']].head(10))