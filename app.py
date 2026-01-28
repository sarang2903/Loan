import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Loan Prediction EDA", layout="wide")

st.title("📊 Loan Prediction Dataset - EDA (Streamlit)")

# =========================
# Load Dataset
# =========================
df = pd.read_csv("LP_Train.csv")

st.subheader("📌 Raw Dataset")
st.dataframe(df)

# =========================
# Missing Values
# =========================
st.subheader("❓ Missing Values Count")
st.write(df.isnull().sum())

# =========================
# Data Types
# =========================
st.subheader("📌 Data Types")
st.write(df.dtypes)

# =========================
# Fill Missing Values
# =========================
df["Gender"] = df["Gender"].fillna("Male")
df["Married"] = df["Married"].fillna("Yes")
df["Dependents"] = df["Dependents"].fillna(0)
df["Self_Employed"] = df["Self_Employed"].fillna("No")
df["LoanAmount"] = df["LoanAmount"].fillna(113.73)
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(344.53)
df["Credit_History"] = df["Credit_History"].fillna(1.0)

# Convert Dependents
df["Dependents"] = df["Dependents"].replace("[+]", "", regex=True).astype("int64")

# =========================
# Describe Numerical Columns
# =========================
st.subheader("📌 Numerical Summary")
st.write(df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']].describe())

# =========================
# Categorical Value Counts
# =========================
st.subheader("📌 Categorical Columns Value Counts")

cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
for col in cat_cols:
    st.write(f"### {col}")
    st.write(df[col].value_counts())

# =========================
# Crosstab Barplots
# =========================
st.subheader("📊 Crosstab Barplots (Category vs Loan_Status)")

cols = ['Gender','Married','Dependents','Education','Self_Employed']

for col in cols:
    st.write(f"### {col} vs Loan_Status")
    fig, ax = plt.subplots()
    pd.crosstab(df[col], df['Loan_Status']).plot(kind='bar', ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

# =========================
# Boxplot ApplicantIncome vs Loan_Status
# =========================
st.subheader("📦 Boxplot: ApplicantIncome vs Loan_Status")
fig, ax = plt.subplots()
sb.boxplot(x=df["Loan_Status"], y=df["ApplicantIncome"], ax=ax)
st.pyplot(fig)

# =========================
# Remove Outliers from ApplicantIncome (IQR)
# =========================
st.subheader("🚫 Removing Outliers (ApplicantIncome using IQR)")

i = "ApplicantIncome"
q1 = np.percentile(df[i], 25)
q3 = np.percentile(df[i], 75)
iqr = q3 - q1

c1 = q1 - 1.5 * iqr
c2 = q3 + 1.5 * iqr

w = df[(df[i] > c2) | (df[i] < c1)].index
st.write("Outliers Removed:", len(w))

df = df.drop(labels=w, axis=0)

st.success("✅ Outliers Removed Successfully!")

# =========================
# Barplot CoapplicantIncome vs Loan_Status
# =========================
st.subheader("📊 Barplot: CoapplicantIncome vs Loan_Status")
fig, ax = plt.subplots()
sb.barplot(x=df["Loan_Status"], y=df["CoapplicantIncome"], ax=ax)
st.pyplot(fig)

# =========================
# Correlation
# =========================
st.subheader("📌 Correlation (Numerical)")
st.write(df[['ApplicantIncome','CoapplicantIncome','LoanAmount']].corr(numeric_only=True))

# =========================
# Groupby Sum LoanAmount
# =========================
st.subheader("📌 LoanAmount Sum (Gender + Married + Education)")
st.write(df.groupby(['Gender','Married','Education'])['LoanAmount'].sum())

# =========================
# Credit_History vs Loan_Status Crosstab
# =========================
st.subheader("📊 Loan_Status vs Credit_History")
fig, ax = plt.subplots()
pd.crosstab(df['Loan_Status'], df['Credit_History']).plot(kind='bar', ax=ax)
plt.xticks(rotation=0)
st.pyplot(fig)

# =========================
# Loan Amount Term Barplot
# =========================
st.subheader("📊 Loan_Amount_Term vs Loan_Status")
fig, ax = plt.subplots()
sb.barplot(x=df["Loan_Status"], y=df["Loan_Amount_Term"], ax=ax)
st.pyplot(fig)

# =========================
# Lineplot Loan_Amount_Term vs Credit_History
# =========================
st.subheader("📈 Lineplot: Loan_Amount_Term vs Credit_History")
fig, ax = plt.subplots()
sb.lineplot(x=df["Loan_Amount_Term"], y=df["Credit_History"], ax=ax)
st.pyplot(fig)

# =========================
# Property Area vs Loan Status
# =========================
st.subheader("📊 Property_Area vs Loan_Status")
fig, ax = plt.subplots()
pd.crosstab(df['Property_Area'], df['Loan_Status']).plot(kind='bar', ax=ax)
plt.xticks(rotation=0)
st.pyplot(fig)

# =========================
# Groupby Property_Area Sum
# =========================
st.subheader("📌 Property_Area wise Sum (LoanAmount + Loan_Amount_Term)")
st.write(df.groupby('Property_Area')[['LoanAmount','Loan_Amount_Term']].sum())
