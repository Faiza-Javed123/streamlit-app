# %% [markdown]
# # Setup and Upload

# %%
!pip -q install --upgrade pip
!pip -q install pandas numpy matplotlib seaborn scikit-learn scipy sdv rdt openpyxl joblib

import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import warnings, re, os
warnings.filterwarnings("ignore")

print("Versions -> pandas:", pd.__version__, "| numpy:", np.__version__)

# ▶ Upload dataset
from google.colab import files
print("Choose your .xlsx / .xls file (e.g., 'Blockchain excel.xlsx')")
uploaded = files.upload()
excel_path = list(uploaded.keys())[0]

# Load first sheet
xls = pd.ExcelFile(excel_path)
print("Sheets found:", xls.sheet_names)
SHEET = xls.sheet_names[0]
df = pd.read_excel(excel_path, sheet_name=SHEET, engine="openpyxl")

print("Shape:", df.shape)
df.head()


# %% [markdown]
# # Exploratory Data Analysis (EDA)

# %%
def maybe_datetime(col: pd.Series):
    if pd.api.types.is_datetime64_any_dtype(col):
        return True
    if pd.api.types.is_numeric_dtype(col):
        return False
    if any(k in (col.name or "").lower() for k in ["date", "time", "timestamp"]):
        try:
            parsed = pd.to_datetime(col, errors="coerce")
            return parsed.notna().mean() > 0.95
        except:
            return False
    return False

def maybe_identifier(s: pd.Series):
    name = (s.name or "").lower()
    unique_ratio = s.nunique(dropna=False) / max(len(s), 1)
    return any(k in name for k in ["id", "code", "name"]) or unique_ratio > 0.95

datetime_cols = [c for c in df.columns if maybe_datetime(df[c])]
id_cols = [c for c in df.columns if maybe_identifier(df[c])]
cat_cols = [c for c in df.columns if df[c].dtype == 'object']
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

print("Datetime cols:", datetime_cols)
print("Identifier cols:", id_cols[:10], "...")
print("Categorical cols:", cat_cols[:10], "...")
print("Numeric cols:", num_cols[:10], "...")

# Categorical cardinality
cardinality = pd.Series({c: df[c].nunique() for c in cat_cols}).sort_values(ascending=False)
display(cardinality.head(10))

# Plot top categories (up to 5 cols)
for c in cardinality.index[:5]:
    vc = df[c].value_counts().head(8)
    plt.figure(figsize=(7,4))
    sns.barplot(x=vc.values, y=vc.index, palette="coolwarm")
    plt.title(f"Top categories - {c}")
    plt.tight_layout()
    plt.show()

# Numeric summary
desc = df[num_cols].describe().T
desc["skew"] = df[num_cols].skew()
desc["kurt"] = df[num_cols].kurt()
display(desc.head(10))

# Histograms
for c in num_cols[:5]:
    plt.figure(figsize=(6,3))
    sns.histplot(df[c].dropna(), bins="auto", kde=True, color="skyblue")
    plt.title(f"{c} (Skew={desc.loc[c,'skew']:.2f})")
    plt.show()

# Correlation heatmap
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0)
    plt.title("Correlation Heatmap")
    plt.show()


# %% [markdown]
# # Preprocessing

# %%
from sklearn.impute import SimpleImputer

# Drop >20% missing
threshold = 0.2
df = df.loc[:, df.isnull().mean() <= threshold]

# Impute numeric + categorical
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

for c in num_cols:
    df[c] = SimpleImputer(strategy="mean").fit_transform(df[[c]]).ravel()
for c in cat_cols:
    df[c] = SimpleImputer(strategy="most_frequent").fit_transform(df[[c]]).ravel()

# Group rare categories
for c in cat_cols:
    freq = df[c].value_counts(normalize=True)
    rare = freq[freq < 0.01].index
    df[c] = df[c].replace(rare, "Other")

# Log-transform skewed numerics
skewness = df[num_cols].skew()
for c in skewness[abs(skewness) > 1].index:
    df[c] = np.log1p(df[c] - df[c].min() + 1)

print("Preprocessing complete. Shape:", df.shape)

# Save cleaned + metadata
from sdv.metadata import SingleTableMetadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
metadata.save_to_json("metadata.json")
df.to_csv("cleaned_dataset.csv", index=False)
print("✅ Saved cleaned_dataset.csv + metadata.json")


# %% [markdown]
# # CTGAN Training

# %%
from sdv.single_table import CTGANSynthesizer

cleaned_df = pd.read_csv("cleaned_dataset.csv")
metadata = SingleTableMetadata.load_from_json("metadata.json")

synth = CTGANSynthesizer(metadata, epochs=300, batch_size=500)
synth.fit(cleaned_df)
print("CTGAN training complete.")

synthetic = synth.sample(num_rows=len(cleaned_df))
synthetic.to_csv("synthetic_dataset.csv", index=False)

import joblib
joblib.dump(synth, "ctgan_model.pkl")
print("✅ Synthetic dataset + model saved")


# %% [markdown]
# # Validation and Comparission

# %%
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from scipy import stats

real_df = pd.read_excel(excel_path, sheet_name=SHEET, engine="openpyxl")
cleaned_df = pd.read_csv("cleaned_dataset.csv")
synthetic_df = pd.read_csv("synthetic_dataset.csv")

# Align columns
common_cols = list(set(cleaned_df.columns) & set(synthetic_df.columns))
real_df, cleaned_df, synthetic_df = real_df[common_cols], cleaned_df[common_cols], synthetic_df[common_cols]

print("Columns compared:", len(common_cols))

# Quality & diagnostic
quality = evaluate_quality(real_df, synthetic_df, metadata)
diagnostic = run_diagnostic(real_df, synthetic_df, metadata)

print("Quality Score:", quality.get_score())
print("Diagnostics:", diagnostic)

# Numeric comparisons
comp = []
for col in real_df.select_dtypes(include=np.number).columns:
    ks_stat, ks_p = stats.ks_2samp(real_df[col].dropna(), synthetic_df[col].dropna())
    comp.append({
        "Column": col,
        "Real_Mean": real_df[col].mean(),
        "Synthetic_Mean": synthetic_df[col].mean(),
        "KS_Stat": ks_stat,
        "KS_p": ks_p
    })
comp_df = pd.DataFrame(comp)
comp_df.to_csv("numeric_comparison.csv", index=False)
display(comp_df.head())


# %% [markdown]
# # Visualizations

# %%
# Numeric distributions
num_cols = real_df.select_dtypes(include=np.number).columns[:5]
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.kdeplot(real_df[col].dropna(), label="Real", fill=True, color="blue", alpha=0.4)
    sns.kdeplot(synthetic_df[col].dropna(), label="Synthetic", fill=True, color="orange", alpha=0.4)
    plt.title(f"Distribution: {col}")
    plt.legend()
    plt.show()

# Categorical comparisons
cat_cols = real_df.select_dtypes(exclude=np.number).columns[:3]
for col in cat_cols:
    plt.figure(figsize=(7,3))
    real_df[col].value_counts(normalize=True).head(5).plot(kind="bar", alpha=0.6, label="Real")
    synthetic_df[col].value_counts(normalize=True).head(5).plot(kind="bar", alpha=0.6, label="Synthetic")
    plt.title(f"Category Comparison: {col}")
    plt.legend()
    plt.show()

# Correlation heatmaps
plt.figure(figsize=(10,8))
sns.heatmap(real_df.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Real Data Correlation")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(synthetic_df.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Synthetic Data Correlation")
plt.show()


# %% [markdown]
# # PCA Projection (Real vs Synthetic)

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Use only numeric columns
num_cols = real_df.select_dtypes(include=np.number).columns
if len(num_cols) >= 2:
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_df[num_cols].dropna())
    synth_scaled = scaler.transform(synthetic_df[num_cols].dropna())

    # PCA to 2D
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_scaled)
    synth_pca = pca.transform(synth_scaled)

    plt.figure(figsize=(8,6))
    plt.scatter(real_pca[:,0], real_pca[:,1], alpha=0.5, label="Real", color="blue")
    plt.scatter(synth_pca[:,0], synth_pca[:,1], alpha=0.5, label="Synthetic", color="orange")
    plt.title("PCA Projection (First 2 Components)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()
else:
    print("❌ Not enough numeric columns for PCA projection.")



