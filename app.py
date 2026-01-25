import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Loan Prediction EDA", layout="wide")
st.title("📊 Loan Prediction Dataset - EDA")

# Load data
df = pd.read_csv("LP_Train.csv")

st.subheader("Raw Dataset")
st.dataframe(df)

# Missing values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# =========================
# Data Cleaning
# =========================
df['Gender'].fillna('Male', inplace=True)
df['Married'].fillna('Yes', inplace=True)
df['Self_Employed'].fillna('No', inplace=True)
df['LoanAmount'].fillna(128.0, inplace=True)
df['Loan_Amount_Term'].fillna(360.0, inplace=True)
df['Credit_History'].fillna(1.0, inplace=True)

# 🔥 Correct Dependents handling
df['Dependents'] = df['Dependents'].fillna('0')
df['Dependents'] = df['Dependents'].str.replace('+', '', regex=False)
df['Dependents'] = df['Dependents'].astype(int)

# =========================
# Statistical Summary
# =========================
st.subheader("Statistical Summary")
st.write(
    df[['ApplicantIncome','CoapplicantIncome',
        'LoanAmount','Loan_Amount_Term','Credit_History']].describe()
)

# =========================
# Categorical Distributions
# =========================
st.subheader("Categorical Feature Distribution")
cat_cols = ['Gender','Married','Dependents','Education',
            'Self_Employed','Property_Area']

for col in cat_cols:
    st.write(f"**{col}**")
    st.write(df[col].value_counts())

# =========================
# Loan Status vs Categorical
# =========================
st.subheader("Loan Status vs Categorical Features")
cols = ['Gender','Married','Dependents','Education','Self_Employed']

for col in cols:
    fig, ax = plt.subplots()
    pd.crosstab(df[col], df['Loan_Status']).plot(kind='bar', ax=ax)
    ax.set_title(col)
    st.pyplot(fig)
    plt.close(fig)

# =========================
# Applicant Income Boxplot
# =========================
st.subheader("Applicant Income vs Loan Status")
fig, ax = plt.subplots()
sb.boxplot(data=df, x='Loan_Status', y='ApplicantIncome', ax=ax)
st.pyplot(fig)
plt.close(fig)

# =========================
# Outlier Removal Function
# =========================
def remove_outliers(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    return data[(data[col] >= q1 - 1.5*iqr) & (data[col] <= q3 + 1.5*iqr)]

df = remove_outliers(df, 'ApplicantIncome')
df = remove_outliers(df, 'CoapplicantIncome')
df = remove_outliers(df, 'LoanAmount')
df = remove_outliers(df, 'Loan_Amount_Term')

# =========================
# Coapplicant Income
# =========================
st.subheader("Coapplicant Income vs Loan Status")
fig, ax = plt.subplots()
sb.boxplot(data=df, x='Loan_Status', y='CoapplicantIncome', ax=ax)
st.pyplot(fig)
plt.close(fig)

# =========================
# Correlation
# =========================
st.subheader("Correlation Matrix")
st.write(df[['ApplicantIncome','CoapplicantIncome','LoanAmount']].corr())

# =========================
# Loan Amount Comparisons
# =========================
for col in ['Gender','Married','Education']:
    st.subheader(f"Loan Amount vs {col}")
    fig, ax = plt.subplots()
    sb.boxplot(data=df, x=col, y='LoanAmount', ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# =========================
# Loan Term vs Loan Status
# =========================
st.subheader("Loan Amount Term vs Loan Status")
fig, ax = plt.subplots()
sb.barplot(data=df, x='Loan_Status', y='Loan_Amount_Term', ax=ax)
st.pyplot(fig)
plt.close(fig)

# =========================
# Scatter Plot
# =========================
st.subheader("Loan Amount Term vs Credit History")
fig, ax = plt.subplots()
sb.scatterplot(data=df, x='Loan_Amount_Term', y='Credit_History', ax=ax)
st.pyplot(fig)
plt.close(fig)

# =========================
# Property Area vs Loan Status
# =========================
st.subheader("Property Area vs Loan Status")
fig, ax = plt.subplots()
pd.crosstab(df['Property_Area'], df['Loan_Status']).plot(kind='bar', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
st.pyplot(fig)
plt.close(fig)

st.success("EDA Completed Successfully ✅")
