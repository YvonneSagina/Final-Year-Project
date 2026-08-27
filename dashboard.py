import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO

# PAGE CONFIGURATION 

st.set_page_config(
    page_title="ProcureGuard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS 

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(#2b2b36 2px, transparent 2px);
        background-size: 30px 30px;
    }
    .metric-card {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #e94560;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover { transform: translateY(-5px); }
    h1, h2, h3 { font-family: 'Inter', sans-serif; }
    .high-risk { color: #ff4b4b; font-weight: 700; }
    .medium-risk { color: #f59e0b; font-weight: 700; }
    .low-risk { color: #00cc96; font-weight: 700; }
    .stDataFrame { font-size: 13px; border-radius: 8px; overflow: hidden; }
    .upload-box {
        background-color: #1a1c24;
        border: 2px dashed #3a3f4b;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    div.stRadio > div[role="radiogroup"] > label {
        margin-bottom: 20px; 
    }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL 

@st.cache_resource
def load_model():
    return joblib.load('models/random_forest_model.pkl')

model = load_model()

# HELPER FUNCTIONS 

def format_currency(value):
    if pd.isna(value):
        return 'N/A'
    return f"KES {value:,.0f}"

def get_risk_icon(level):
    if level == 'High':
        return '🔴'
    elif level == 'Medium':
        return '🟡'
    else:
        return '🟢'

# PIPELINE FUNCTIONS 

def engineer_features(df):
    df['display_award_value'] = df['award_value'].copy()
    df['display_contract_value'] = df['contract_value'].copy()

    # Feature 1: Price Deviation Ratio
    median_award = df['award_value'].median()
    df['price_deviation_ratio'] = df['award_value'] / median_award
    #contract_calc = df['contract_value'].replace(0, 1)
    #df['price_deviation_ratio'] = df['award_value'] / contract_calc

    # Feature 2: Repeat Award Frequency
    supplier_counts = df['supplier_name'].value_counts()
    df['repeat_award_frequency'] = df['supplier_name'].map(supplier_counts)

    # Feature 3: Vendor Registration Age
    df['vendor_registration_date'] = pd.to_datetime(
        df['vendor_registration_date'], errors='coerce', utc=True
    ).dt.tz_localize(None)

    df['contractPeriod_startDate'] = pd.to_datetime(
        df['contractPeriod_startDate'], errors='coerce', utc=True
    ).dt.tz_localize(None)

    df['vendor_registration_age'] = (
        df['contractPeriod_startDate'] - df['vendor_registration_date']
    ).dt.days

    median_age = df['vendor_registration_age'].median()
    df['vendor_registration_age'] = df['vendor_registration_age'].fillna(median_age)

    return df

def clean_data(df):
    cols_needed = [
        'main_ocid', 'buyer_name', 'supplier_name',
        'tender_procurementMethod', 'date',
        'value_amount', 'contract_value',
        'contractPeriod_startDate', 'contractPeriod_endDate'
    ]
    cols_needed = [c for c in cols_needed if c in df.columns]
    df = df[cols_needed].copy()

    df.rename(columns={
        'tender_procurementMethod': 'procurement_method',
        'date': 'tender_date',
        'value_amount': 'award_value'
    }, inplace=True)

    for col in ['tender_date', 'contractPeriod_startDate', 'contractPeriod_endDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_localize(None)

    df['supplier_name'] = df['supplier_name'].fillna('Unknown Supplier')
    df['contract_value'] = df['contract_value'].fillna(df['award_value'])
    df['procurement_method'] = df['procurement_method'].fillna('open')

    # Remove invalid zero value records
    df = df[df['award_value'] > 0]
    df = df[df['contract_value'] > 0]

    df.drop_duplicates(subset=['main_ocid'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def generate_registration_dates(df):
    np.random.seed(42)
    registration_dates = []

    for _, row in df.iterrows():
        ref_date = row.get('contractPeriod_startDate')
        if pd.isnull(ref_date):
            ref_date = pd.Timestamp('2024-01-01')
        days_before = np.random.randint(365, 3650)
        registration_dates.append(ref_date - pd.Timedelta(days=int(days_before)))

    df['vendor_registration_date'] = registration_dates
    return df

def merge_files(main, awards, awards_suppliers, contracts):
    
    df = pd.merge(main, awards,
                left_on='ocid', right_on='main_ocid',
                how='inner', suffixes=('_main', '_award'))
    df = pd.merge(df, awards_suppliers[['main_ocid', 'name']],
                  on='main_ocid', how='left')
    df.rename(columns={'name': 'supplier_name'}, inplace=True)
    df = pd.merge(df, contracts[['main_ocid', 'value_amount']],
                  on='main_ocid', how='left',
                  suffixes=('', '_contract'))
    df.rename(columns={'value_amount_contract': 'contract_value'}, inplace=True)
    return df


def score_records(df):
    features = df[['price_deviation_ratio',
                   'repeat_award_frequency',
                   'vendor_registration_age']].copy()
    features = features.dropna()
    df = df.loc[features.index].copy()

    df['fraud_probability'] = model.predict_proba(features)[:, 1]
    df['risk_flag'] = model.predict(features)
    df['risk_level'] = pd.cut(
        df['fraud_probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    return df

def save_to_db(df):
    conn = sqlite3.connect('models/procurement_fraud.db')
    df[['main_ocid', 'buyer_name', 'supplier_name',
        'procurement_method', 'award_value', 'contract_value']].to_sql(
        'live_procurement_records', conn, if_exists='replace', index=False)
    df[['main_ocid', 'price_deviation_ratio',
        'repeat_award_frequency', 'vendor_registration_age']].to_sql(
        'live_engineered_features', conn, if_exists='replace', index=False)
    df[['main_ocid', 'fraud_probability', 'risk_flag', 'risk_level']].to_sql(
        'live_fraud_scores', conn, if_exists='replace', index=False)
    conn.close()

# SESSION STATE

if 'scored_df' not in st.session_state:
    st.session_state.scored_df = None
if 'pipeline_done' not in st.session_state:
    st.session_state.pipeline_done = False

# SIDEBAR 

st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=60)
st.sidebar.title("ProcureGuard")
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "Home",
    "Risk Overview",
    "Transaction Detail",
    "Export Report"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk Threshold**")
threshold = st.sidebar.slider(
    "Adjust fraud sensitivity",
    min_value=0.0, max_value=1.0,
    value=0.7, step=0.05,
    help="Records above this threshold are flagged as high risk"
)

if st.session_state.scored_df is not None:
    df = st.session_state.scored_df
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Dataset:** {len(df):,} records")
    st.sidebar.markdown(
        f"**High Risk:** {len(df[df['fraud_probability'] >= threshold]):,}"
    )

# HOME / UPLOAD 

if page == "Home":
    st.title("Procurement Fraud Detection")

    if not st.session_state.pipeline_done:
        # ── UPLOAD SECTION ─────────────────────────────────────────────
        st.markdown("### Upload Procurement Files")
        st.info("Upload your procurement CSV files below. The system will automatically identify each file by its contents.")

        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload your procurement CSV files. The system identifies each file automatically."
        )

        def identify_file(df, filename):
            cols = df.columns.tolist()
            if 'ocid' in cols and 'buyer_name' in cols:
                return 'main'
            elif 'value_amount' in cols and 'contractPeriod_startDate' in cols:
                return 'awards'
            elif 'awards_id' in cols:
                return 'awards_suppliers'
            elif 'dateSigned' in cols:
                return 'contracts'
            else:
                return None

        if uploaded_files:
            identified = {}
            unrecognized = []

            for f in uploaded_files:
                try:
                    temp_df = pd.read_csv(f)
                    file_type = identify_file(temp_df, f.name)
                    if file_type:
                        identified[file_type] = temp_df
                        f.seek(0)
                    else:
                        unrecognized.append(f.name)
                except Exception as e:
                    unrecognized.append(f.name)

            st.markdown("**File Recognition Results:**")
            required = ['main', 'awards', 'awards_suppliers', 'contracts']
            for r in required:
                if r in identified:
                    st.success(f"✅ {r.replace('_', ' ').title()} file — identified")
                else:
                    st.error(f"❌ {r.replace('_', ' ').title()} file — not found")

            if unrecognized:
                st.warning(f"Could not identify: {', '.join(unrecognized)}")

            all_found = all(r in identified for r in required)

            if all_found:
                if st.button(" Run Fraud Analysis", type="primary", use_container_width=True):
                    with st.spinner("Running analysis — please wait..."):
                        progress = st.progress(0)
                        status = st.empty()

                        status.text("Merging procurement records...")
                        df = merge_files(
                            identified['main'],
                            identified['awards'],
                            identified['awards_suppliers'],
                            identified['contracts']
                        )
                        progress.progress(25)

                        status.text("Cleaning data...")
                        df = clean_data(df)
                        progress.progress(40)

                        status.text("Generating vendor registration dates...")
                        df = generate_registration_dates(df)
                        progress.progress(55)

                        status.text("Engineering fraud detection features...")
                        df = engineer_features(df)
                        progress.progress(70)

                        status.text("Scoring records with Random Forest model...")
                        df = score_records(df)
                        progress.progress(85)

                        status.text("Saving results to database...")
                        save_to_db(df)
                        progress.progress(100)

                        st.session_state.scored_df = df
                        st.session_state.pipeline_done = True
                        status.text("Analysis complete!")
                        st.rerun()
            else:
                st.warning("Some required files are missing. Please upload all four file types to proceed.")

    else:
        # RESULTS SECTION 
        df = st.session_state.scored_df

        if st.button("🔄 Upload New Dataset", use_container_width=False):
            st.session_state.scored_df = None
            st.session_state.pipeline_done = False
            st.rerun()

        st.markdown("### ✅ Analysis Complete")
        st.markdown("---")

        # Summary metric cards
        total = len(df)
        high_risk = len(df[df['fraud_probability'] >= threshold])
        medium_risk = len(df[(df['fraud_probability'] >= 0.3) &
                             (df['fraud_probability'] < threshold)])
        detection_rate = (high_risk / total * 100)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records Assessed", f"{total:,}")
        with col2:
            st.metric("High Risk Flagged", f"{high_risk:,}",
                      delta=f"{detection_rate:.1f}% of dataset")
        with col3:
            st.metric("Medium Risk", f"{medium_risk:,}")
        with col4:
            st.metric("Model ROC-AUC", "0.9981")

        st.markdown("---")

        col_left, col_right = st.columns(2)

        with col_left:
            risk_counts = df['risk_level'].value_counts()
            fig_pie = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color=risk_counts.index,
                color_discrete_map={
                    'High': '#e94560',
                    'Medium': '#f59e0b',
                    'Low': '#4caf50'
                }
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            fig_hist = px.histogram(
                df, x='fraud_probability', nbins=50,
                title="Fraud Probability Score Distribution",
                color_discrete_sequence=['#e94560']
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                xaxis_title="Fraud Probability",
                yaxis_title="Number of Records"
            )
            fig_hist.add_vline(
                x=threshold, line_dash="dash",
                line_color="white",
                annotation_text=f"Threshold: {threshold}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Top 10 highest risk transactions
        st.markdown("### ⚠️ Top 10 Highest Risk Transactions")
        high_risk_df = df.sort_values(
            'fraud_probability', ascending=False
        ).head(10)

        display_df = high_risk_df[[
            'main_ocid', 'supplier_name', 'buyer_name',
            'display_award_value', 'fraud_probability', 'risk_level'
        ]].copy()
        display_df['display_award_value'] = display_df['display_award_value'].apply(format_currency)
        display_df['fraud_probability'] = display_df['fraud_probability'].apply(
            lambda x: f"{x:.4f}"
        )
        display_df['risk_level'] = display_df['risk_level'].apply(
            lambda x: f"{get_risk_icon(x)} {x}"
        )
        display_df.columns = ['OCID', 'Supplier', 'Buyer',
                               'Award Value', 'Fraud Score', 'Risk Level']
        st.dataframe(display_df, use_container_width=True)

# RISK OVERVIEW 

elif page == "Risk Overview":
    st.title("Risk Overview")

    if st.session_state.scored_df is None:
        st.warning("No data loaded. Please go to Home and upload your CSV files first.")
    else:
        df = st.session_state.scored_df
        st.markdown("**All scored procurement records with fraud probability rankings**")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            risk_filter = st.selectbox("Filter by Risk Level",
                                       ["All", "High", "Medium", "Low"])
        with col2:
            method_filter = st.selectbox(
                "Filter by Procurement Method",
                ["All"] + sorted(df['procurement_method'].dropna().unique().tolist())
            )
        with col3:
            search = st.text_input("Search by Supplier Name")

        filtered_df = df.copy()
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
        if method_filter != "All":
            filtered_df = filtered_df[
                filtered_df['procurement_method'] == method_filter
            ]
        if search:
            filtered_df = filtered_df[
                filtered_df['supplier_name'].str.contains(
                    search, case=False, na=False
                )
            ]

        filtered_df = filtered_df.sort_values('fraud_probability', ascending=False)
        st.markdown(f"**Showing {len(filtered_df):,} records**")

        display_df = filtered_df[[
            'main_ocid', 'supplier_name', 'buyer_name',
            'procurement_method', 'award_value',
            'fraud_probability', 'risk_level'
        ]].copy()
        display_df['award_value'] = display_df['award_value'].apply(format_currency)
        display_df['fraud_probability'] = display_df['fraud_probability'].apply(
            lambda x: f"{x:.4f}"
        )
        display_df['risk_level'] = display_df['risk_level'].apply(
            lambda x: f"{get_risk_icon(x)} {x}"
        )
        display_df.columns = ['OCID', 'Supplier', 'Buyer',
                               'Method', 'Award Value', 'Fraud Score', 'Risk Level']
        st.dataframe(display_df, use_container_width=True, height=400)

        st.markdown("---")
        st.markdown("### Fraud Score by Procurement Method")
        fig_box = px.box(
            df.dropna(subset=['procurement_method']),
            x='procurement_method',
            y='fraud_probability',
            color='procurement_method',
            title="Fraud Probability Distribution by Procurement Method",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_box.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)

# TRANSACTION DETAIL 

elif page == "Transaction Detail":
    st.title("Transaction Detail")

    if st.session_state.scored_df is None:
        st.warning("No data loaded. Please go to Home and upload your CSV files first.")
    else:
        df = st.session_state.scored_df
        st.markdown("**Select a transaction to view its fraud risk breakdown**")
        st.markdown("---")

        selected_ocid = st.selectbox(
            "Select Transaction OCID",
            df.sort_values('fraud_probability', ascending=False)['main_ocid'].tolist()
        )

        if selected_ocid:
            record = df[df['main_ocid'] == selected_ocid].iloc[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud Probability Score",
                          f"{record['fraud_probability']:.4f}")
            with col2:
                risk = record['risk_level']
                st.metric("Risk Level", f"{get_risk_icon(risk)} {risk}")
            with col3:
                st.metric("Award Value", format_currency(record['award_value']))

            st.markdown("---")
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### 📋 Procurement Details")
                details = {
                    "OCID": record['main_ocid'],
                    "Buyer": record['buyer_name'],
                    "Supplier": record['supplier_name'],
                    "Procurement Method": record['procurement_method'],
                    "Award Value": format_currency(record['display_award_value']),
                    "Contract Value": format_currency(record['display_contract_value'])
                }
                for key, value in details.items():
                    st.markdown(f"**{key}:** {value}")

            with col_right:
                st.markdown("###  Explainability — Feature Contributions")

                features = {
                    'Price Deviation Ratio': record['price_deviation_ratio'],
                    'Repeat Award Frequency': record['repeat_award_frequency'],
                    'Vendor Registration Age': record['vendor_registration_age']
                }

                max_val = max(abs(v) for v in features.values() if pd.notna(v))
                normalized = {k: abs(v) / max_val for k, v in features.items()
                              if pd.notna(v)} if max_val > 0 else {k: 0 for k in features}

                fig_bar = go.Figure(go.Bar(
                    x=list(normalized.values()),
                    y=list(normalized.keys()),
                    orientation='h',
                    marker_color=['#e94560', '#f59e0b', '#4caf50'],
                    text=[f"{v:.2f}" for v in features.values()],
                    textposition='outside'
                ))
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis_title="Relative Contribution",
                    height=250
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("**Raw Feature Values:**")
                st.markdown(f"- Price Deviation Ratio: **{record['price_deviation_ratio']:.4f}**")
                st.markdown(f"- Repeat Award Frequency: **{record['repeat_award_frequency']:.0f}** appearances")
                st.markdown(f"- Vendor Registration Age: **{record['vendor_registration_age']:.0f}** days")

# EXPORT REPORT 

elif page == "Export Report":
    st.title("Export Risk Assessment Report")

    if st.session_state.scored_df is None:
        st.warning("No data loaded. Please go to Home and upload your CSV files first.")
    else:
        df = st.session_state.scored_df
        st.markdown("**Generate and download a formal audit report**")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            export_risk = st.selectbox("Include Risk Level",
                                       ["All", "High", "Medium", "Low"])
        with col2:
            export_method = st.selectbox(
                "Filter by Procurement Method",
                ["All"] + sorted(
                    df['procurement_method'].dropna().unique().tolist()
                )
            )

        export_df = df.copy()
        if export_risk != "All":
            export_df = export_df[export_df['risk_level'] == export_risk]
        if export_method != "All":
            export_df = export_df[export_df['procurement_method'] == export_method]
        export_df = export_df.sort_values('fraud_probability', ascending=False)

        st.markdown("### Report Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(export_df):,}")
        with col2:
            st.metric("High Risk",
                      f"{len(export_df[export_df['risk_level'] == 'High']):,}")
        with col3:
            st.metric("Medium Risk",
                      f"{len(export_df[export_df['risk_level'] == 'Medium']):,}")

        st.markdown("### Preview")
        preview = export_df[[
            'main_ocid', 'supplier_name', 'buyer_name',
            'award_value', 'fraud_probability', 'risk_level'
        ]].head(20).copy()
        preview['award_value'] = preview['award_value'].apply(format_currency)
        preview['fraud_probability'] = preview['fraud_probability'].apply(
            lambda x: f"{x:.4f}"
        )
        st.dataframe(preview, use_container_width=True)

        st.markdown("---")
        report_df = export_df[[
            'main_ocid', 'supplier_name', 'buyer_name',
            'procurement_method', 'award_value', 'contract_value',
            'fraud_probability', 'risk_level',
            'price_deviation_ratio', 'repeat_award_frequency',
            'vendor_registration_age'
        ]].copy()

        csv = report_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Report as CSV",
            data=csv,
            file_name=f"fraud_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        if st.button("💾 Save Report to Database"):
            conn = sqlite3.connect('models/procurement_fraud.db')
            report_meta = pd.DataFrame([{
                'generated_by': 'Auditor',
                'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_records': len(export_df),
                'high_risk_count': len(export_df[export_df['risk_level'] == 'High']),
                'medium_risk_count': len(export_df[export_df['risk_level'] == 'Medium']),
                'low_risk_count': len(export_df[export_df['risk_level'] == 'Low'])
            }])
            report_meta.to_sql('exported_reports', conn,
                               if_exists='append', index=False)
            conn.close()
            st.success("Report saved to database successfully.")