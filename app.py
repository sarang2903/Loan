import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Loan Prediction EDA",
    page_icon="💰",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>💰 Loan Prediction - Exploratory Data Analysis</h1>",
    unsafe_allow_html=True
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("LP_Train.csv")
    return df

df = load_data()

# ================= SIDEBAR =================
st.sidebar.header("📌 Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Dataset Overview", "Data Cleaning", "EDA Visualizations", "Insights"]
)

# ================= DATA CLEANING FUNCTION =================
def clean_data(df):
    df.Gender.fillna('Male', inplace=True)
    df.Married.fillna('Yes', inplace=True)
    df.Dependents.fillna(0, inplace=True)
    df.Self_Employed.fillna('No', inplace=True)
    df.LoanAmount.fillna(128.0, inplace=True)
    df.Loan_Amount_Term.fillna(360.0, inplace=True)
    df.Credit_History.fillna(1.0, inplace=True)
    df.Dependents = df.Dependents.replace('[+]', '', regex=True).astype(int)
    return df

df = clean_data(df)

# ================= OUTLIER REMOVAL =================
def remove_outliers(df, col):
    q1 = np.percentile(df[col], 25)
    q3 = np.percentile(df[col], 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']:
    df = remove_outliers(df, col)

# ================= DATASET OVERVIEW =================
if section == "Dataset Overview":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])

    st.subheader("📌 Data Types")
    st.write(df.dtypes)

    st.subheader("❗ Missing Values")
    st.write(df.isnull().sum())

# ================= DATA CLEANING =================
elif section == "Data Cleaning":
    st.subheader("🧹 Cleaning Summary")
    st.success("✔ Missing values handled\n✔ Dependents cleaned\n✔ Outliers removed")

    st.subheader("📊 Statistical Summary")
    st.write(
        df[['ApplicantIncome','CoapplicantIncome','LoanAmount',
            'Loan_Amount_Term','Credit_History']].describe()
    )

# ================= EDA VISUALIZATIONS =================
elif section == "EDA Visualizations":

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Categorical Analysis", "Income Analysis", "Loan Analysis", "Property Area"]
    )

    # -------- TAB 1 --------
    with tab1:
        st.subheader("Loan Status vs Categorical Features")
        cols = ['Gender','Married','Education','Self_Employed']

        for col in cols:
            fig, ax = plt.subplots()
            pd.crosstab(df[col], df['Loan_Status']).plot(kind='bar', ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

    # -------- TAB 2 --------
    with tab2:
        st.subheader("Income Distribution")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sb.boxplot(x=df.Loan_Status, y=df.ApplicantIncome, ax=ax)
            ax.set_title("Applicant Income vs Loan Status")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            sb.boxplot(x=df.Loan_Status, y=df.CoapplicantIncome, ax=ax)
            ax.set_title("Coapplicant Income vs Loan Status")
            st.pyplot(fig)

    # -------- TAB 3 --------
    with tab3:
        st.subheader("Loan Amount Analysis")

        fig, ax = plt.subplots()
        sb.boxplot(x=df.Education, y=df.LoanAmount, ax=ax)
        ax.set_title("Loan Amount vs Education")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        sb.barplot(x=df.Loan_Status, y=df.Loan_Amount_Term, ax=ax)
        ax.set_title("Loan Term vs Loan Status")
        st.pyplot(fig)

    # -------- TAB 4 --------
    with tab4:
        st.subheader("Property Area Impact")

        fig, ax = plt.subplots()
        pd.crosstab(df.Property_Area, df.Loan_Status).plot(kind='bar', ax=ax)
        ax.set_title("Property Area vs Loan Status")
        plt.xticks(rotation=0)
        st.pyplot(fig)

# ================= INSIGHTS =================
else:
    st.subheader("📌 Key Insights")
    st.markdown("""
    ✔ Applicants with **Credit History = 1** have higher approval rate  
    ✔ **Graduate applicants** tend to get higher loan amounts  
    ✔ **Urban & Semiurban** areas show more loan approvals  
    ✔ Higher applicant income increases approval probability  
    """)

    st.balloons()
