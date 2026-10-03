import os
import joblib
import numpy as np
import pandas as pd

import joblib
joblib.dump(model_rfr, "random_forest_model.pkl")
joblib.dump(model_dt, "decision_tree_model.pkl")
joblib.dump(model_lr, "linear_regression_model.pkl")
import streamlit as st

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="House Rent Prediction App",
    page_icon="🏠",
    layout="wide"
)


# ==================================================
# FILE PATHS
# ==================================================
DATASET_PATH = r"Ed\Lab 21 House_Rent_Dataset.csv"

MODEL_FILES = {
    "Linear Regression": "linear_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}


# ==================================================
# LOAD DATASET
# ==================================================
@st.cache_data
def load_data():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found: {DATASET_PATH}")
        st.stop()

    dataset = pd.read_csv(DATASET_PATH)

    required_columns = [
        "BHK",
        "Size",
        "Floor",
        "Area Type",
        "City",
        "Furnishing Status",
        "Tenant Preferred",
        "Bathroom",
        "Rent"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        st.error(
            f"Required dataset columns are missing: "
            f"{missing_columns}"
        )
        st.stop()

    return dataset


df = load_data()


# ==================================================
# LOAD SAVED MODELS
# ==================================================
@st.cache_resource
def load_models():
    loaded_models = {}

    for model_name, model_path in MODEL_FILES.items():

        if not os.path.exists(model_path):
            loaded_models[model_name] = None
            continue

        try:
            loaded_models[model_name] = joblib.load(model_path)

        except Exception as error:
            loaded_models[model_name] = error

    return loaded_models


models = load_models()


# ==================================================
# ENCODE DATASET
# ==================================================
@st.cache_data
def prepare_dataset(dataset):
    """
    Label-encode the categorical features and create the
    eight input features expected by the models.
    """

    model_df = dataset[
        [
            "BHK",
            "Size",
            "Floor",
            "Area Type",
            "City",
            "Furnishing Status",
            "Tenant Preferred",
            "Bathroom",
            "Rent"
        ]
    ].copy()

    # Remove rows with missing target values
    model_df = model_df.dropna(subset=["Rent"])

    numerical_columns = [
        "BHK",
        "Size",
        "Bathroom",
        "Rent"
    ]

    categorical_columns = [
        "Floor",
        "Area Type",
        "City",
        "Furnishing Status",
        "Tenant Preferred"
    ]

    # Convert numerical columns to numbers
    for column in numerical_columns:
        model_df[column] = pd.to_numeric(
            model_df[column],
            errors="coerce"
        )

    # Fill missing numerical values using median
    for column in ["BHK", "Size", "Bathroom"]:
        model_df[column] = model_df[column].fillna(
            model_df[column].median()
        )

    # Remove rows where Rent could not be converted
    model_df = model_df.dropna(subset=["Rent"])

    encoders = {}

    # Create a separate encoder for each categorical feature
    for column in categorical_columns:
        model_df[column] = (
            model_df[column]
            .fillna("Unknown")
            .astype(str)
        )

        encoder = LabelEncoder()
        model_df[column] = encoder.fit_transform(
            model_df[column]
        )

        encoders[column] = encoder

    feature_columns = [
        "BHK",
        "Size",
        "Floor",
        "Area Type",
        "City",
        "Furnishing Status",
        "Tenant Preferred",
        "Bathroom"
    ]

    X = model_df[feature_columns]
    y = model_df["Rent"]

    return X, y, encoders, feature_columns


X_all, y_all, encoders, feature_columns = prepare_dataset(df)


# ==================================================
# HELPER FUNCTIONS
# ==================================================
def encode_value(encoder, value, column_name):
    value = str(value)

    if value not in encoder.classes_:
        raise ValueError(
            f"The value '{value}' is not recognized for "
            f"'{column_name}'."
        )

    return int(encoder.transform([value])[0])


def align_input_with_model(model, input_dataframe):
    """
    Align the input columns with the columns used by the
    selected model.
    """

    if hasattr(model, "feature_names_in_"):
        expected_columns = list(model.feature_names_in_)

        missing_columns = [
            column
            for column in expected_columns
            if column not in input_dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"The selected model requires these missing "
                f"features: {missing_columns}"
            )

        return input_dataframe[expected_columns]

    if hasattr(model, "n_features_in_"):
        expected_count = int(model.n_features_in_)
        received_count = input_dataframe.shape[1]

        if expected_count != received_count:
            raise ValueError(
                f"The model expects {expected_count} features, "
                f"but the application created {received_count}."
            )

    return input_dataframe


def calculate_metrics(model, X, y):
    """
    Generate predictions and calculate regression metrics.
    """

    aligned_X = align_input_with_model(model, X)

    predictions = model.predict(aligned_X)

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    return metrics, predictions


# ==================================================
# TITLE
# ==================================================
st.title("🏠 House Rent Prediction App")

st.markdown(
    """
    Predict house rent using pre-trained **Linear Regression**,
    **Decision Tree**, and **Random Forest** models.
    """
)


# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header("⚙️ Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

active_model = models.get(selected_model_name)

st.sidebar.subheader("🏡 House Information")

bhk = st.sidebar.selectbox(
    "BHK",
    sorted(df["BHK"].dropna().unique())
)

size = st.sidebar.slider(
    "Size in Square Feet",
    min_value=int(df["Size"].min()),
    max_value=int(df["Size"].max()),
    value=int(df["Size"].median())
)

floor = st.sidebar.selectbox(
    "Floor",
    sorted(
        df["Floor"]
        .dropna()
        .astype(str)
        .unique()
    )
)

area_type = st.sidebar.selectbox(
    "Area Type",
    sorted(
        df["Area Type"]
        .dropna()
        .astype(str)
        .unique()
    )
)

city = st.sidebar.selectbox(
    "City",
    sorted(
        df["City"]
        .dropna()
        .astype(str)
        .unique()
    )
)

furnishing_status = st.sidebar.selectbox(
    "Furnishing Status",
    sorted(
        df["Furnishing Status"]
        .dropna()
        .astype(str)
        .unique()
    )
)

tenant_preferred = st.sidebar.selectbox(
    "Tenant Preferred",
    sorted(
        df["Tenant Preferred"]
        .dropna()
        .astype(str)
        .unique()
    )
)

bathroom = st.sidebar.selectbox(
    "Bathroom",
    sorted(df["Bathroom"].dropna().unique())
)


# ==================================================
# DATASET AND MODEL STATUS
# ==================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Dataset Overview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.write(f"Dataset rows: **{df.shape[0]}**")
    st.write(f"Dataset columns: **{df.shape[1]}**")


with col2:
    st.subheader("🤖 Model Status")

    if active_model is None:
        st.error(
            f"Model file not found: "
            f"{MODEL_FILES[selected_model_name]}"
        )

    elif isinstance(active_model, Exception):
        st.error(
            f"Model loading failed: {active_model}"
        )

    else:
        st.success(
            f"{selected_model_name} loaded successfully."
        )

        st.write(
            f"Model class: "
            f"**{type(active_model).__name__}**"
        )

        if hasattr(active_model, "n_features_in_"):
            st.write(
                f"Expected features: "
                f"**{active_model.n_features_in_}**"
            )

        if hasattr(active_model, "feature_names_in_"):
            with st.expander("View training feature names"):
                st.write(
                    active_model.feature_names_in_.tolist()
                )


# ==================================================
# USER INPUT SUMMARY
# ==================================================
st.markdown("---")
st.subheader("📝 Selected House Details")

input_summary = pd.DataFrame({
    "Feature": [
        "BHK",
        "Size",
        "Floor",
        "Area Type",
        "City",
        "Furnishing Status",
        "Tenant Preferred",
        "Bathroom"
    ],
    "Selected Value": [
        bhk,
        size,
        floor,
        area_type,
        city,
        furnishing_status,
        tenant_preferred,
        bathroom
    ]
})

st.dataframe(
    input_summary,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# CALCULATE PREDICTION
# ==================================================
st.markdown("---")
st.subheader("🔮 Prediction and Metrics")

calculate_button = st.button(
    "Calculate Estimated Rent",
    type="primary",
    use_container_width=True
)

if calculate_button:

    if active_model is None:
        st.error(
            "Prediction cannot be performed because the "
            "selected model file is missing."
        )

    elif isinstance(active_model, Exception):
        st.error(
            f"The model could not be loaded: {active_model}"
        )

    else:
        try:
            with st.spinner(
                "Calculating prediction and model metrics..."
            ):
                # Encode categorical inputs
                encoded_floor = encode_value(
                    encoders["Floor"],
                    floor,
                    "Floor"
                )

                encoded_area_type = encode_value(
                    encoders["Area Type"],
                    area_type,
                    "Area Type"
                )

                encoded_city = encode_value(
                    encoders["City"],
                    city,
                    "City"
                )

                encoded_furnishing = encode_value(
                    encoders["Furnishing Status"],
                    furnishing_status,
                    "Furnishing Status"
                )

                encoded_tenant = encode_value(
                    encoders["Tenant Preferred"],
                    tenant_preferred,
                    "Tenant Preferred"
                )

                # Create exactly eight features
                user_input = pd.DataFrame(
                    [{
                        "BHK": bhk,
                        "Size": size,
                        "Floor": encoded_floor,
                        "Area Type": encoded_area_type,
                        "City": encoded_city,
                        "Furnishing Status": encoded_furnishing,
                        "Tenant Preferred": encoded_tenant,
                        "Bathroom": bathroom
                    }]
                )

                # Match the model's expected column order
                final_input = align_input_with_model(
                    active_model,
                    user_input
                )

                # Generate user prediction
                predicted_rent = active_model.predict(
                    final_input
                )[0]

                predicted_rent = max(
                    0,
                    float(predicted_rent)
                )

                # Calculate model metrics
                metrics, dataset_predictions = calculate_metrics(
                    active_model,
                    X_all,
                    y_all
                )

            # ------------------------------------------
            # Display predicted rent
            # ------------------------------------------
            st.success(
                f"## Estimated Monthly Rent: "
                f"₹{predicted_rent:,.2f}"
            )

            # ------------------------------------------
            # Display metrics
            # ------------------------------------------
            st.subheader(
                f"📈 {selected_model_name} Metrics Report"
            )

            metric_col1, metric_col2, metric_col3, metric_col4 = (
                st.columns(4)
            )

            metric_col1.metric(
                "MAE",
                f"₹{metrics['MAE']:,.2f}"
            )

            metric_col2.metric(
                "MSE",
                f"{metrics['MSE']:,.2f}"
            )

            metric_col3.metric(
                "RMSE",
                f"₹{metrics['RMSE']:,.2f}"
            )

            metric_col4.metric(
                "R² Score",
                f"{metrics['R2']:.4f}"
            )

            # ------------------------------------------
            # Metrics explanation
            # ------------------------------------------
            with st.expander(
                "How to understand these metrics"
            ):
                st.markdown(
                    """
                    - **MAE:** Average absolute difference between
                      actual and predicted rent. Lower is better.

                    - **MSE:** Average squared prediction error.
                      Lower is better.

                    - **RMSE:** Typical prediction error measured in
                      rent units. Lower is better.

                    - **R² score:** Shows how much variation in rent
                      is explained by the model. A value closer to
                      `1.0` is better.
                    """
                )

            # ------------------------------------------
            # Actual and predicted table
            # ------------------------------------------
            report_df = pd.DataFrame({
                "Actual Rent": y_all.to_numpy(),
                "Predicted Rent": dataset_predictions
            })

            report_df["Absolute Error"] = (
                report_df["Actual Rent"]
                - report_df["Predicted Rent"]
            ).abs()

            st.subheader("📋 Actual vs Predicted Report")

            st.dataframe(
                report_df.head(100).style.format({
                    "Actual Rent": "₹{:,.2f}",
                    "Predicted Rent": "₹{:,.2f}",
                    "Absolute Error": "₹{:,.2f}"
                }),
                use_container_width=True
            )

            # ------------------------------------------
            # Actual vs predicted chart
            # ------------------------------------------
            st.subheader("📊 Actual vs Predicted Chart")

            chart_df = report_df[
                ["Actual Rent", "Predicted Rent"]
            ].head(100)

            st.line_chart(chart_df)

            # ------------------------------------------
            # Prediction input
            # ------------------------------------------
            with st.expander(
                "View encoded prediction input"
            ):
                st.dataframe(
                    final_input,
                    use_container_width=True
                )

        except Exception as error:
            st.error(
                f"Prediction or metric calculation failed: "
                f"{error}"
            )

            st.exception(error)


# ==================================================
# COMPARE ALL AVAILABLE MODELS
# ==================================================
st.markdown("---")
st.subheader("🏆 Compare All Models")

if st.button(
    "Generate Model Comparison Report",
    use_container_width=True
):
    comparison_results = []

    with st.spinner("Comparing available models..."):

        for model_name, model in models.items():

            if model is None or isinstance(model, Exception):
                continue

            try:
                model_metrics, _ = calculate_metrics(
                    model,
                    X_all,
                    y_all
                )

                comparison_results.append({
                    "Model": model_name,
                    "MAE": model_metrics["MAE"],
                    "MSE": model_metrics["MSE"],
                    "RMSE": model_metrics["RMSE"],
                    "R² Score": model_metrics["R2"]
                })

            except Exception as error:
                st.warning(
                    f"{model_name} could not be evaluated: "
                    f"{error}"
                )

    if comparison_results:
        comparison_df = pd.DataFrame(
            comparison_results
        )

        comparison_df = comparison_df.sort_values(
            by="R² Score",
            ascending=False
        )

        st.dataframe(
            comparison_df.style.format({
                "MAE": "{:,.2f}",
                "MSE": "{:,.2f}",
                "RMSE": "{:,.2f}",
                "R² Score": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        best_model = comparison_df.iloc[0]["Model"]
        best_r2 = comparison_df.iloc[0]["R² Score"]

        st.success(
            f"Best model based on R² score: "
            f"**{best_model}** with R² = "
            f"**{best_r2:.4f}**"
        )

        st.subheader("R² Score Comparison")

        comparison_chart = (
            comparison_df
            .set_index("Model")[["R² Score"]]
        )

        st.bar_chart(comparison_chart)

    else:
        st.error(
            "No models were available for comparison."
        )