import streamlit as st
import pandas as pd
import joblib
import os

import train_model  # ✅ direct import (no subprocess)

st.set_page_config(page_title="House Price Prediction", layout="wide")

MODEL_FILE = "models/model.pkl"
PIPELINE_FILE = "models/pipeline.pkl"

# -------------------- HEADER --------------------
st.title("🏠 House Price Prediction Dashboard")
st.markdown("Upload your dataset → Validate → Predict → Analyze → Download")

# -------------------- AUTO TRAIN --------------------
if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    st.warning("⚠️ Model not found. Training model... ⏳")

    try:
        train_model.train()
        st.success("✅ Model trained successfully!")
    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        st.stop()

# -------------------- CACHED LOAD --------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline

model, pipeline = load_artifacts()

# -------------------- SIDEBAR --------------------
st.sidebar.title("📋 Required Columns")

required_columns = list(pipeline.feature_names_in_)

st.sidebar.markdown("Ensure your dataset contains:")

for col in required_columns:
    st.sidebar.markdown(f"- `{col}`")

st.sidebar.markdown("---")
st.sidebar.info("Upload CSV → Validate → Predict → Download")

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("👀 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # -------------------- VALIDATION --------------------
        st.subheader("🔍 Column Validation")

        missing_cols = list(set(required_columns) - set(df.columns))
        extra_cols = list(set(df.columns) - set(required_columns))

        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
            st.stop()

        if extra_cols:
            st.warning(f"⚠️ Extra columns removed: {extra_cols}")
            df = df[required_columns]

        st.success("✅ Dataset is valid")

        # -------------------- PREDICTION BUTTON --------------------
        if st.button("🚀 Run Prediction", use_container_width=True):

            with st.spinner("Generating predictions..."):

                # ✅ Use predict function from train_model
                result_df = train_model.predict(df.copy())

            # -------------------- KPIs --------------------
            avg_price = result_df["prediction"].mean()
            max_price = result_df["prediction"].max()
            min_price = result_df["prediction"].min()

            col1, col2, col3 = st.columns(3)

            col1.metric("💰 Avg Price", f"₹ {avg_price:,.0f}")
            col2.metric("📈 Max Price", f"₹ {max_price:,.0f}")
            col3.metric("📉 Min Price", f"₹ {min_price:,.0f}")

            # -------------------- RESULTS TABLE --------------------
            st.subheader("📈 Prediction Results Preview")

            def highlight_prediction(col):
                if col.name == "prediction":
                    return ['background-color: #145a32; color: white; font-weight: bold'] * len(col)
                return [''] * len(col)

            styled_df = result_df.head().style \
                .apply(highlight_prediction, axis=0) \
                .format({"prediction": "{:,.2f}"})

            st.dataframe(styled_df, use_container_width=True)

            # -------------------- CHART --------------------
            st.subheader("📊 Price Distribution")
            st.bar_chart(result_df["prediction"])

            # -------------------- DOWNLOAD --------------------
            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.success("✅ Prediction completed successfully!")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")