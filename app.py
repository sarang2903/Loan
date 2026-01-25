import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Loan Prediction EDA", layout="wide")
st.title("📊 Loan Prediction Dataset - EDA")

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("LP_Train.csv")

df = load_data()

st.subheader("Raw Dataset")
st.dataframe(df)

# =========================
# Missing Values
# =========================
st.subheader("Missing Values")
st.write(df.isnull().sum())

# =========================
# Data Cleaning
# =========================
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Dependents
df['Dependents'] = df['Dependents'].fillna('0')
df['Dependents'] = df['Dependents'].str.replace('+', '', regex=False).astype(int)

# =========================
# Outlier Removal
# =========================
def remove_outliers(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    return data[(data[col] >= q1 - 1.5 * iqr) & (data[col] <= q3 + 1.5 * iqr)]

for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']:
    df = remove_outliers(df, col)

# =========================
# Statistical Summary
# =========================
st.subheader("Statistical Summary")
st.write(df.describe())

# =========================
# Categorical Distributions
# =========================
st.subheader("Categorical Feature Distribution")
cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'Property_Area']

for col in cat_cols:
    st.write(f"**{col}**")
    st.write(df[col].value_counts())

# =========================
# Loan Status vs Categorical
# =========================
st.subheader("Loan Status vs Categorical Features")

for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed']:
    fig, ax = plt.subplots()
    pd.crosstab(df[col], df['Loan_Status']).plot(kind='bar', ax=ax)
    ax.set_title(col)
    st.pyplot(fig)
    plt.close(fig)

# =========================
# Income Analysis
# =========================
st.subheader("Applicant Income vs Loan Status")
fig, ax = plt.subplots()
sb.boxplot(data=df, x='Loan_Status', y='ApplicantIncome', ax=ax)
st.pyplot(fig)
plt.close(fig)

st.subheader("Coapplicant Income vs Loan Status")
fig, ax = plt.subplots()
sb.boxplot(data=df, x='Loan_Status', y='CoapplicantIncome', ax=ax)
st.pyplot(fig)
plt.close(fig)

# =========================
# Correlation Matrix
# =========================
st.subheader("Correlation Matrix")
num_cols = df.select_dtypes(include=np.number)
fig, ax = plt.subplots(figsize=(8, 6))
sb.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)
plt.close(fig)

# =========================
# Loan Amount Comparisons
# =========================
for col in ['Gender', 'Married', 'Education']:
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
# Credit History vs Loan Status
# =========================
st.subheader("Credit History vs Loan Status")
fig, ax = plt.subplots()
pd.crosstab(df['Credit_History'], df['Loan_Status']).plot(kind='bar', ax=ax)
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