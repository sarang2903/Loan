import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Loan Prediction EDA", layout="wide")

st.title("📊 Loan Prediction Dataset - EDA")

# ======================
# Load Data
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("LP_Train.csv")
    return df

df = load_data()

st.subheader("Raw Dataset")
st.dataframe(df)

# ======================
# Data Cleaning
# ======================
st.subheader("Missing Values")
st.write(df.isnull().sum())

df['Gender'] = df['Gender'].fillna('Male')
df['Married'] = df['Married'].fillna('Yes')
df['Dependents'] = df['Dependents'].fillna(0)
df['Self_Employed'] = df['Self_Employed'].fillna('No')
df['LoanAmount'] = df['LoanAmount'].fillna(113.73)
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(344.53)
df['Credit_History'] = df['Credit_History'].fillna(1.0)

df['Dependents'] = df['Dependents'].replace('[+]','', regex=True).astype('int64')

st.success("Missing values handled successfully!")

# ======================
# Descriptive Statistics
# ======================
st.subheader("Numerical Feature Summary")
st.write(df[['ApplicantIncome','CoapplicantIncome',
             'LoanAmount','Loan_Amount_Term','Credit_History']].describe())

# ======================
# Categorical Value Counts
# ======================
st.subheader("Categorical Feature Distribution")
cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']

for col in cat_cols:
    st.write(f"### {col}")
    st.write(df[col].value_counts())

# ======================
# Crosstab Plots
# ======================
st.subheader("Loan Status vs Categorical Features")

cols = ['Gender','Married','Dependents','Education','Self_Employed']
for col in cols:
    fig, ax = plt.subplots()
    pd.crosstab(df[col], df['Loan_Status']).plot(kind='bar', ax=ax)
    plt.title(f"{col} vs Loan Status")
    st.pyplot(fig)

# ======================
# Boxplot - Applicant Income
# ======================
st.subheader("Applicant Income vs Loan Status")
fig, ax = plt.subplots()
sns.boxplot(x=df['Loan_Status'], y=df['ApplicantIncome'], ax=ax)
st.pyplot(fig)

# ======================
# Outlier Removal (ApplicantIncome)
# ======================
i = 'ApplicantIncome'
q1 = np.percentile(df[i], 25)
q3 = np.percentile(df[i], 75)
iqr = q3 - q1

c1 = q1 - 1.5 * iqr
c2 = q3 + 1.5 * iqr

outliers = df[(df[i] < c1) | (df[i] > c2)].index
df = df.drop(outliers)

st.info(f"Outliers removed from {i}: {len(outliers)} rows")

# ======================
# Barplot - Coapplicant Income
# ======================
st.subheader("Coapplicant Income vs Loan Status")
fig, ax = plt.subplots()
sns.barplot(x=df['Loan_Status'], y=df['CoapplicantIncome'], ax=ax)
st.pyplot(fig)

# ======================
# Correlation
# ======================
st.subheader("Correlation Matrix")
st.write(df[['ApplicantIncome','CoapplicantIncome','LoanAmount']].corr())

# ======================
# GroupBy Analysis
# ======================
st.subheader("Loan Amount by Gender, Marital Status & Education")
st.write(df.groupby(['Gender','Married','Education'])['LoanAmount'].sum())

# ======================
# Credit History vs Loan Status
# ======================
st.subheader("Credit History vs Loan Status")
fig, ax = plt.subplots()
pd.crosstab(df['Loan_Status'], df['Credit_History']).plot(kind='bar', ax=ax)
plt.xticks(rotation=0)
st.pyplot(fig)

# ======================
# Loan Amount Term Analysis
# ======================
st.subheader("Loan Amount Term Analysis")

fig, ax = plt.subplots()
sns.barplot(x=df['Loan_Status'], y=df['Loan_Amount_Term'], ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
sns.lineplot(x=df['Loan_Amount_Term'], y=df['Credit_History'], ax=ax)
st.pyplot(fig)

# ======================
# Property Area Analysis
# ======================
st.subheader("Property Area vs Loan Status")
fig, ax = plt.subplots()
pd.crosstab(df['Property_Area'], df['Loan_Status']).plot(kind='bar', ax=ax)
plt.xticks(rotation=0)
st.pyplot(fig)

st.subheader("Loan Amount & Term by Property Area")
st.write(df.groupby('Property_Area')[['LoanAmount','Loan_Amount_Term']].sum())

st.success("EDA Completed 🎉")
