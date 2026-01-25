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

# Data cleaning
df.Gender = df.Gender.fillna('Male')
df.Married = df.Married.fillna('Yes')
df.Dependents = df.Dependents.fillna(0)
df.Self_Employed = df.Self_Employed.fillna('No')
df.LoanAmount = df.LoanAmount.fillna(128.0)
df.Loan_Amount_Term = df.Loan_Amount_Term.fillna(360.0)
df.Credit_History = df.Credit_History.fillna(1.0)

df.Dependents = df.Dependents.replace('[+]', '', regex=True).astype('int64')

# Describe
st.subheader("Statistical Summary")
st.write(df[['ApplicantIncome','CoapplicantIncome','LoanAmount',
             'Loan_Amount_Term','Credit_History']].describe())

# Categorical value counts
st.subheader("Categorical Feature Distribution")
cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
for col in cat_cols:
    st.write(f"**{col}**")
    st.write(df[col].value_counts())

# Bar plots (Loan Status vs categorical)
st.subheader("Loan Status vs Categorical Features")
cols = ['Gender','Married','Dependents','Education','Self_Employed']

for col in cols:
    fig, ax = plt.subplots()
    pd.crosstab(df[col], df['Loan_Status']).plot(kind='bar', ax=ax)
    plt.title(col)
    st.pyplot(fig)

# Boxplot Applicant Income
st.subheader("Applicant Income vs Loan Status")
fig, ax = plt.subplots()
sb.boxplot(x=df.Loan_Status, y=df.ApplicantIncome, ax=ax)
st.pyplot(fig)

# Outlier removal function
def remove_outliers(df, col):
    q1 = np.percentile(df[col], 25)
    q3 = np.percentile(df[col], 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[col] >= lower) & (df[col] <= upper)]

df = remove_outliers(df, 'ApplicantIncome')
df = remove_outliers(df, 'CoapplicantIncome')
df = remove_outliers(df, 'LoanAmount')
df = remove_outliers(df, 'Loan_Amount_Term')

# Coapplicant Income
st.subheader("Coapplicant Income vs Loan Status")
fig, ax = plt.subplots()
sb.boxplot(x=df.Loan_Status, y=df.CoapplicantIncome, ax=ax)
st.pyplot(fig)

# Correlation
st.subheader("Correlation Matrix")
st.write(df[['ApplicantIncome','CoapplicantIncome','LoanAmount']].corr())

# Loan Amount vs Gender
st.subheader("Loan Amount vs Gender")
fig, ax = plt.subplots()
sb.boxplot(x=df.Gender, y=df.LoanAmount, ax=ax)
st.pyplot(fig)

# Loan Amount vs Married
st.subheader("Loan Amount vs Married")
fig, ax = plt.subplots()
sb.boxplot(x=df.Married, y=df.LoanAmount, ax=ax)
st.pyplot(fig)

# Loan Amount vs Education
st.subheader("Loan Amount vs Education")
fig, ax = plt.subplots()
sb.boxplot(x=df.Education, y=df.LoanAmount, ax=ax)
st.pyplot(fig)

# Loan Amount Term vs Loan Status
st.subheader("Loan Amount Term vs Loan Status")
fig, ax = plt.subplots()
sb.barplot(x=df.Loan_Status, y=df.Loan_Amount_Term, ax=ax)
st.pyplot(fig)

# Scatter Plot
st.subheader("Loan Amount Term vs Credit History")
fig, ax = plt.subplots()
sb.scatterplot(x=df.Loan_Amount_Term, y=df.Credit_History, ax=ax)
st.pyplot(fig)

# Property Area vs Loan Status
st.subheader("Property Area vs Loan Status")
fig, ax = plt.subplots()
pd.crosstab(df['Property_Area'], df['Loan_Status']).plot(kind='bar', ax=ax)
plt.xticks(rotation=0)
st.pyplot(fig)

st.success("EDA Completed Successfully ✅")
