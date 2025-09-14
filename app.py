import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import joblib
import os

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

# -------------------------------
# Title
# -------------------------------
st.title("📊 Big Data Project - Streamlit App")

# -------------------------------
# Load Dataset: Fixed file or fallback uploader
# -------------------------------

st.header("1️⃣ Load Dataset (Fixed)")

DATA_FILE = "Blockchain excel.xlsx"   
xls = pd.ExcelFile(DATA_FILE)

xls = pd.ExcelFile(DATA_FILE)
SHEET = xls.sheet_names[0]
df = pd.read_excel(DATA_FILE, sheet_name=SHEET, engine="openpyxl")

st.success(f"Dataset loaded! Shape = {df.shape}")
st.write(df.head())
# -------------------------------
# EDA
# -------------------------------
st.header("2️⃣ Exploratory Data Analysis (EDA)")

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

st.subheader("Numeric Summary")
desc = df[num_cols].describe().T
desc["skew"] = df[num_cols].skew()
desc["kurt"] = df[num_cols].kurt()
st.write(desc)

if len(num_cols) > 1:
    st.subheader("Correlation Heatmap")
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax)
    st.pyplot(fig)

# -------------------------------
# Preprocessing
# -------------------------------
st.header("3️⃣ Preprocessing")

threshold = 0.2
df = df.loc[:, df.isnull().mean() <= threshold]

for c in num_cols:
    df[c] = SimpleImputer(strategy="mean").fit_transform(df[[c]]).ravel()
for c in cat_cols:
    df[c] = SimpleImputer(strategy="most_frequent").fit_transform(df[[c]]).ravel()

for c in cat_cols:
    freq = df[c].value_counts(normalize=True)
    rare = freq[freq < 0.01].index
    df[c] = df[c].replace(rare, "Other")

skewness = df[num_cols].skew()
for c in skewness[abs(skewness) > 1].index:
    df[c] = np.log1p(df[c] - df[c].min() + 1)

st.success(f"Preprocessing complete! Shape = {df.shape}")

# -------------------------------
# CTGAN Training
# -------------------------------
st.header("4️⃣ Train CTGAN Model")

if st.button("Start Training"):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    synth = CTGANSynthesizer(metadata, epochs=50, batch_size=200)
    synth.fit(df)

    synthetic = synth.sample(num_rows=len(df))
    st.success("✅ CTGAN training complete!")
    st.write(synthetic.head())

    joblib.dump(synth, "ctgan_model.pkl")

    # -------------------------------
    # Validation
    # -------------------------------
    st.header("5️⃣ Validation and Comparison")

    quality = evaluate_quality(df, synthetic, metadata)
    diagnostic = run_diagnostic(df, synthetic, metadata)

    st.write("Quality Score:", quality.get_score())
    st.write("Diagnostics:", diagnostic)

    comp = []
    for col in df.select_dtypes(include=np.number).columns:
        ks_stat, ks_p = stats.ks_2samp(df[col].dropna(), synthetic[col].dropna())
        comp.append({
            "Column": col,
            "Real_Mean": df[col].mean(),
            "Synthetic_Mean": synthetic[col].mean(),
            "KS_Stat": ks_stat,
            "KS_p": ks_p
        })
    comp_df = pd.DataFrame(comp)
    st.subheader("Numeric Column Comparison")
    st.write(comp_df.head())

    # -------------------------------
    # PCA Projection
    # -------------------------------
    st.header("6️⃣ PCA Projection (Real vs Synthetic)")

    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) >= 2:
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(df[num_cols].dropna())
        synth_scaled = scaler.transform(synthetic[num_cols].dropna())

        pca = PCA(n_components=2)
        real_pca = pca.fit_transform(real_scaled)
        synth_pca = pca.transform(synth_scaled)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(real_pca[:,0], real_pca[:,1], alpha=0.5, label="Real", color="blue")
        ax.scatter(synth_pca[:,0], synth_pca[:,1], alpha=0.5, label="Synthetic", color="orange")
        ax.set_title("PCA Projection")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for PCA.")