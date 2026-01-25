import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Loan EDA", layout="wide")

st.title("💰 Loan Prediction EDA")

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("LP_Train.csv")

df = load_data()

# ---------- CLEAN DATA ----------
df['Gender'] = df['Gender'].fillna('Male')
df['Married'] = df['Married'].fillna('Yes')
df['Self_Employed'] = df['Self_Employed'].fillna('No')
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(360)
df['Credit_History'] = df['Credit_History'].fillna(1)

df['Dependents'] = df['Dependents'].astype(str)
df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Dependents'] = df['Dependents'].fillna('0').astype(int)

# ---------- SIDEBAR ----------
section = st.sidebar.radio(
    "Navigation",
    ["Overview", "Categorical Analysis", "Income Analysis"]
)

# ---------- OVERVIEW ----------
if section == "Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# ---------- CATEGORICAL ----------
elif section == "Categorical Analysis":
    col = st.selectbox(
        "Select Categorical Column",
        ['Gender','Married','Education','Self_Employed','Property_Area']
    )

    fig, ax = plt.subplots()
    pd.crosstab(df[col], df['Loan_Status']).plot(kind='bar', ax=ax)
    st.pyplot(fig)
    plt.close()

# ---------- INCOME ----------
else:
    fig, ax = plt.subplots()
    sb.boxplot(x=df['Loan_Status'], y=df['ApplicantIncome'], ax=ax)
    st.pyplot(fig)
    plt.close()
