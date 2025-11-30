import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ks_2samp

# SDV Imports
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Data Synthesize Generator",
    page_icon="📊",
    layout="wide"
)

# ------------------- HEADER -------------------
st.markdown("<h1 style='text-align: center; color: #00D4FF;'>Data Synthesize Generator</h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------- SIDEBAR -------------------
st.sidebar.title("EXPLORER")
page = st.sidebar.radio("Select Section", ["HOME", "ABOUT", "GENERATE", "POST PROCESSING", "ACCURACY"])

# ------------------- HOME -------------------
if page == "HOME":
    st.header("Synthetic Data Generator")
    st.success("Upload → Generate → Validate → Download — All in seconds!")
    st.markdown("""
    ### Key Features
    - Generate synthetic data using **CTGAN** or **Gaussian Copula**
    - Automatic cleaning (duplicates + missing values)
    - Statistical Validation (KS Test, Histograms, Mean/Median/Std)
    - Full Post-Processing options
    - Download options for cleaned & synthetic data
    """)

# ------------------- ABOUT -------------------
elif page == "ABOUT":
    st.header("Model Information")
    st.info("""
    ### **CTGAN**
    Best for mixed data (categorical + numerical).  
    Produces realistic synthetic rows using GAN architecture.

    ### **Gaussian Copula**
    Best for numerical data where correlation structure matters.
    """)

# ------------------- GENERATE -------------------
elif page == "GENERATE":
    st.header("Generate Synthetic Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

            # Drop columns with >20% missing values
            df = df.loc[:, df.isnull().mean() < 0.20]

            st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(), use_container_width=True)

            col1, col2 = st.columns(2)
            model_name = col1.selectbox("Select Model", ["CTGAN", "Gaussian Copula"])
            samples = col2.number_input("Synthetic Rows", min_value=100, value=len(df), step=100)

            epochs = st.slider("Training Epochs", 100, 800, 300)
            batch_size = st.slider("Batch Size", 128, 1024, 500)

            data_focus = "Mixed"
            generator_decay = None
            if model_name == "CTGAN":
                data_focus = st.radio(
                    "Data Type Focus (CTGAN Only)",
                    ["Mixed (Recommended)", "Numerical", "Categorical"], horizontal=True
                )
                generator_decay = st.number_input(
                    "Generator Decay",
                    min_value=1e-8, max_value=1e-5,
                    value=1e-6, step=1e-7, format="%.8f"
                )

            if st.button("Generate Synthetic Data", type="primary"):
                with st.spinner("Training the model... Please wait"):
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(df)

                    if model_name == "CTGAN":
                        model = CTGANSynthesizer(
                            metadata=metadata,
                            enforce_min_max_values=(data_focus != "Numerical"),
                            epochs=epochs,
                            batch_size=batch_size,
                            generator_decay=generator_decay,
                            verbose=False
                        )
                    else:
                        model = GaussianCopulaSynthesizer(metadata=metadata)

                    model.fit(df)
                    synthetic = model.sample(num_rows=samples)

                st.session_state.original = df
                st.session_state.synthetic = synthetic

                st.success("Synthetic Data Generated Successfully!")
                st.balloons()

                colA, colB = st.columns(2)
                colA.subheader("Original Data (Top 10)")
                colA.dataframe(df.head(10))
                colB.subheader("Synthetic Data (Top 10)")
                colB.dataframe(synthetic.head(10))

                st.download_button(
                    "Download Synthetic Data",
                    data=synthetic.to_csv(index=False).encode(),
                    file_name="synthetic_data.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Error loading file: {e}")

# ------------------- POST PROCESSING -------------------
elif page == "POST PROCESSING":
    st.header("Post Processing (Data Cleaning + Comparison)")
    
    if "original" not in st.session_state or "synthetic" not in st.session_state:
        st.warning("Generate synthetic data first!")
        st.stop()

    orig = st.session_state.original.copy()
    synth = st.session_state.synthetic.copy()

    # ---------------- 1. Remove Columns with > 20% Missing ----------------
    st.subheader("1. Remove Columns with > 20% Missing Values")
    thresh_orig = orig.shape[0] * 0.20
    thresh_synth = synth.shape[0] * 0.20
    orig_clean = orig.dropna(thresh=orig.shape[0] - thresh_orig, axis=1)
    synth_clean = synth.dropna(thresh=synth.shape[0] - thresh_synth, axis=1)

    st.write("Original Columns Before:", orig.shape[1])
    st.write("Original Columns After:", orig_clean.shape[1])
    st.write("Synthetic Columns Before:", synth.shape[1])
    st.write("Synthetic Columns After:", synth_clean.shape[1])
    st.success("Columns with more than 20% missing values removed!")

    # ---------------- 2. Numerical Columns Analysis ----------------
    st.subheader("2. Numerical Columns Analysis")

    num_cols = orig_clean.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) == 0:
        st.info("No numerical columns found.")
    else:
        st.write("### Histograms & Statistics")
        for col in num_cols:
            if col in synth_clean.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=orig_clean[col], name="Original", opacity=0.6))
                fig.add_trace(go.Histogram(x=synth_clean[col], name="Synthetic", opacity=0.6))
                fig.update_layout(title=f"Histogram: {col}", barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)

                # Stats: Mean, Median, Mode, Std
                st.write(f"**Statistics for {col}:**")
                stats_df = pd.DataFrame({
                    "Original": [orig_clean[col].mean(), orig_clean[col].median(), orig_clean[col].mode()[0], orig_clean[col].std()],
                    "Synthetic": [synth_clean[col].mean(), synth_clean[col].median(), synth_clean[col].mode()[0], synth_clean[col].std()]
                }, index=["Mean", "Median", "Mode", "Std Dev"])
                st.dataframe(stats_df)

        # Correlation Matrix
        st.write("### Correlation Matrix")
        st.write("**Original Data Correlation:**")
        st.dataframe(orig_clean[num_cols].corr(), use_container_width=True)
        st.write("**Synthetic Data Correlation:**")
        st.dataframe(synth_clean[num_cols].corr(), use_container_width=True)

    # ---------------- 3. Categorical Columns Analysis ----------------
    st.subheader("3. Categorical Columns Analysis")

    cat_cols = orig_clean.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) == 0:
        st.info("No categorical columns found.")
    else:
        for col in cat_cols:
            if col in synth_clean.columns:
                # Histogram
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=orig_clean[col].value_counts().index,
                    y=orig_clean[col].value_counts().values,
                    name="Original"
                ))
                fig2.add_trace(go.Bar(
                    x=synth_clean[col].value_counts().index,
                    y=synth_clean[col].value_counts().values,
                    name="Synthetic"
                ))
                fig2.update_layout(title=f"Category Frequency: {col}", barmode='group')
                st.plotly_chart(fig2, use_container_width=True)

                # Mode & Frequency
                st.write(f"**Mode & Frequency for {col}:**")
                mode_orig = orig_clean[col].mode()[0]
                mode_synth = synth_clean[col].mode()[0]
                freq_orig = orig_clean[col].value_counts()[mode_orig]
                freq_synth = synth_clean[col].value_counts()[mode_synth]
                st.write(f"Original → Mode: {mode_orig}, Count: {freq_orig}")
                st.write(f"Synthetic → Mode: {mode_synth}, Count: {freq_synth}")

    # ---------------- 4. KS Test for Numerical Columns ----------------
    st.subheader("4. KS Test (Numerical Columns)")
    ks_results = []
    for col in num_cols:
        if col in synth_clean.columns:
            ks_stat, p_val = ks_2samp(orig_clean[col].dropna(), synth_clean[col].dropna())
            ks_results.append([col, round(ks_stat, 4), round(1 - ks_stat, 4)])
    ks_df = pd.DataFrame(ks_results, columns=["Column", "KS Statistic", "Match Score (0-1)"])
    st.dataframe(ks_df, use_container_width=True)

# ------------------- ACCURACY -------------------
elif page == "ACCURACY":
    st.header("Model Accuracy Evaluation")
    if "original" not in st.session_state or "synthetic" not in st.session_state:
        st.warning("Generate data first to calculate accuracy.")
        st.stop()

    orig = st.session_state.original
    synth = st.session_state.synthetic

    def calculate_accuracy(original, synthetic):
        num_cols = original.select_dtypes(include=['int64', 'float64']).columns
        scores = []
        for col in num_cols:
            if col in synthetic.columns:
                o = original[col].dropna()
                s = synthetic[col].dropna()
                if len(o) > 0 and len(s) > 0:
                    ks_stat, p_val = ks_2samp(o, s)
                    scores.append(1 - ks_stat)
        return round((sum(scores) / len(scores)) * 100, 2) if scores else 0

    if st.button("Calculate Accuracy", type="primary"):
        with st.spinner("Calculating..."):
            accuracy = calculate_accuracy(orig, synth)
        st.success("Accuracy Calculated!")
        st.metric("Model Accuracy", f"{accuracy}%")

        st.subheader("Column-wise Similarity Breakdown")
        summary = []
        for col in orig.select_dtypes(include=['int64', 'float64']).columns:
            if col in synth.columns:
                ks_stat, _ = ks_2samp(orig[col].dropna(), synth[col].dropna())
                summary.append([col, round(1 - ks_stat, 3)])
        st.dataframe(pd.DataFrame(summary, columns=["Column", "Match Score (0-1)"]))

# ------------------- FOOTER -------------------
st.markdown("---")
st.caption("© 2025 Data Synthesizer – Privacy-First Synthetic Data Platform")
