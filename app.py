
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MODEL_PATHS = {
    "Decision Tree": MODELS_DIR / "decision_tree_baseline.joblib",
    "Random Forest": MODELS_DIR / "random_forest_multiclass.joblib",
}


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Network IDS | Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 16px;
            border-radius: 12px;
        }
        .subtitle {
            color: #808080;
            font-size: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def get_model(model_name: str):
    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    try:
        return load_model(str(model_path))
    except Exception as exc:
        st.error(f"Could not load the selected model: {exc}")
        st.stop()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.title("🛡️ Network IDS")
    st.caption("Machine Learning Intrusion Detection")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "Analyze CSV"],
        help="View project results or analyze a CSV file.",
    )

    selected_model = st.selectbox(
        "Detection model",
        list(MODEL_PATHS.keys()),
    )

    st.divider()
    st.caption("Dataset: CICIDS2017")
    st.caption("Task: Multiclass classification")
    st.caption("Mode: Offline CSV analysis")


# --------------------------------------------------
# Dashboard page
# --------------------------------------------------
if page == "Dashboard":
    st.title("Network Security Dashboard")
    st.markdown(
        '<p class="subtitle">Machine-learning-based network traffic analysis '
        'using the CICIDS2017 dataset.</p>',
        unsafe_allow_html=True,
    )

    st.info(
        "This is an offline analysis dashboard. It does not capture live "
        "network traffic. The figures below summarize previously completed "
        "experiments; they are not live monitoring statistics."
    )

    st.subheader("Project overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Input features", "78")
    col2.metric("Traffic classes", "10")
    col3.metric("Training records", "1,299,329")
    col4.metric("Test records", "324,833")

    st.divider()
    st.subheader("Model evaluation")

    comparison = pd.DataFrame(
        {
            "Model": ["Decision Tree", "Random Forest"],
            "Accuracy (%)": [99.83, 99.69],
            "Macro F1 (%)": [96.17, 94.69],
            "Macro Recall (%)": [None, None],
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Rounded values from the existing random-split evaluation. "
        "Results may be optimistic because related traffic records can "
        "appear in both training and test sets."
    )

    st.divider()
    st.subheader("Existing evaluation reports")

    report_cols = st.columns(2)

    report_images = [
        (
            "Decision Tree — Confusion Matrix",
            REPORTS_DIR / "decision_tree_confusion_matrix.png",
        ),
        (
            "Decision Tree — Normalized Confusion Matrix",
            REPORTS_DIR / "decision_tree_confusion_matrix_normalized.png",
        ),
    ]

    for col, (title, image_path) in zip(report_cols, report_images):
        with col:
            st.markdown(f"**{title}**")
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning(f"Report image not found: {image_path.name}")

    st.subheader("Feature importance")

    importance_cols = st.columns(2)

    importance_images = [
        (
            "Decision Tree",
            REPORTS_DIR / "decision_tree_top20_features.png",
        ),
        (
            "Random Forest",
            REPORTS_DIR / "random_forest_top20_features.png",
        ),
    ]

    for col, (title, image_path) in zip(importance_cols, importance_images):
        with col:
            st.markdown(f"**{title}**")
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning(f"Report image not found: {image_path.name}")


# --------------------------------------------------
# CSV analysis page
# --------------------------------------------------
elif page == "Analyze CSV":
    st.title("Analyze Network Traffic")
    st.markdown(
        "Upload a CSV containing the model's 78 numeric input features. "
        "The selected model will classify each valid row."
    )

    model = get_model(selected_model)

    if not hasattr(model, "feature_names_in_"):
        st.error(
            "The selected model does not expose feature_names_in_. "
            "We need to verify its expected feature order before prediction."
        )
        st.stop()

    expected_features = list(model.feature_names_in_)

    st.caption(
        f"Selected model: {selected_model} · "
        f"Expected input features: {len(expected_features)}"
    )

    uploaded_file = st.file_uploader(
        "Choose a network traffic CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, low_memory=False)
        except Exception as exc:
            st.error(f"Could not read the CSV file: {exc}")
            st.stop()

        # Remove accidental whitespace around column names.
        data.columns = data.columns.astype(str).str.strip()

        if data.columns.duplicated().any():
            st.error(
                "The CSV contains duplicate column names after whitespace "
                "is removed. Please fix the column names and upload it again."
            )
            st.stop()

        st.subheader("Uploaded data")
        st.write(f"Rows: **{len(data):,}** · Columns: **{len(data.columns):,}**")
        st.dataframe(data.head(10), use_container_width=True)

        missing_features = [
            feature
            for feature in expected_features
            if feature not in data.columns
        ]

        if missing_features:
            st.error(
                f"The file is missing {len(missing_features)} required "
                "features. No predictions were made."
            )
            st.write("Missing feature names:")
            st.code("\n".join(missing_features))
            st.stop()

        # Keep only the exact training features and their original order.
        features = data.loc[:, expected_features].copy()

        # Convert values to numeric and exclude rows containing invalid values.
        for column in expected_features:
            features[column] = pd.to_numeric(
                features[column],
                errors="coerce",
            )

        features = features.replace([np.inf, -np.inf], np.nan)

        valid_mask = features.notna().all(axis=1)
        valid_features = features.loc[valid_mask]

        invalid_count = int((~valid_mask).sum())

        st.write(f"Rows ready for prediction: **{len(valid_features):,}**")
        if invalid_count:
            st.warning(
                f"{invalid_count:,} rows were excluded because one or more "
                "required features were missing, non-numeric, or infinite."
            )

        if len(valid_features) == 0:
            st.error("No valid rows remain for prediction.")
            st.stop()

        if st.button("Run intrusion detection", type="primary"):
            with st.spinner(f"Running {selected_model} predictions..."):
                try:
                    predictions = model.predict(valid_features)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.stop()

            results = data.loc[valid_mask].copy()
            results["Predicted_Class"] = predictions
            results["Detection_Status"] = np.where(
                results["Predicted_Class"].astype(str).str.strip().str.upper()
                == "BENIGN",
                "BENIGN",
                "ALERT",
            )

            # Evaluate binary attack detection when ground-truth labels exist.
            evaluation = None

            if "Label" in results.columns:
                actual_labels = (
                    results["Label"].astype(str).str.strip().str.upper()
                )
                labeled_mask = (
                    results["Label"].notna()
                    & actual_labels.ne("")
                    & actual_labels.ne("NAN")
                )

                if labeled_mask.any():
                    actual_attack = actual_labels[labeled_mask].ne("BENIGN")
                    predicted_attack = (
                        results.loc[labeled_mask, "Detection_Status"] == "ALERT"
                    )

                    tp = int((actual_attack & predicted_attack).sum())
                    fp = int((~actual_attack & predicted_attack).sum())
                    fn = int((actual_attack & ~predicted_attack).sum())
                    tn = int((~actual_attack & ~predicted_attack).sum())

                    precision = tp / (tp + fp) if tp + fp else 0.0
                    recall = tp / (tp + fn) if tp + fn else 0.0
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if precision + recall
                        else 0.0
                    )
                    false_positive_rate = (
                        fp / (fp + tn) if fp + tn else 0.0
                    )

                    evaluation = {
                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                        "TN": tn,
                        "Precision": precision,
                        "Recall": recall,
                        "F1-score": f1,
                        "False Positive Rate": false_positive_rate,
                    }


            alert_count = int(
                (results["Detection_Status"] == "ALERT").sum()
            )
            benign_count = len(results) - alert_count
            alert_rate = alert_count / len(results) * 100

            st.session_state["ids_results"] = results
            st.session_state["ids_summary"] = {
                "model": selected_model,
                "total": len(results),
                "alerts": alert_count,
                "benign": benign_count,
                "alert_rate": alert_rate,
                "invalid": invalid_count,
            }

    # Keep results visible after Streamlit reruns.
    if "ids_results" in st.session_state:
        summary = st.session_state["ids_summary"]

        if summary["model"] != selected_model:
            st.info(
                "The displayed results belong to "
                f"{summary['model']}. Run the analysis again to use "
                f"{selected_model}."
            )
        else:
            st.divider()
            st.subheader("Detection results")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Analyzed rows", f"{summary['total']:,}")
            c2.metric("Alerts", f"{summary['alerts']:,}")
            c3.metric("Benign", f"{summary['benign']:,}")
            c4.metric("Alert rate", f"{summary['alert_rate']:.2f}%")

            if evaluation is not None:
                st.divider()
                st.subheader("Binary Attack Detection Evaluation")
                st.caption(
                    "Ground-truth labels are grouped into BENIGN versus ATTACK. "
                    "Every predicted class other than BENIGN counts as an alert."
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Precision", f"{evaluation['Precision']:.2%}")
                m2.metric("Recall", f"{evaluation['Recall']:.2%}")
                m3.metric("F1-score", f"{evaluation['F1-score']:.2%}")
                m4.metric(
                    "False Positive Rate",
                    f"{evaluation['False Positive Rate']:.2%}",
                )

                st.markdown("**Confusion Matrix Counts**")
                cm1, cm2, cm3, cm4 = st.columns(4)
                cm1.metric("True Positives (TP)", evaluation["TP"])
                cm2.metric("False Positives (FP)", evaluation["FP"])
                cm3.metric("False Negatives (FN)", evaluation["FN"])
                cm4.metric("True Negatives (TN)", evaluation["TN"])


            results = st.session_state["ids_results"]

            st.markdown("**Predicted class distribution**")
            distribution = (
                results["Predicted_Class"]
                .astype(str)
                .value_counts()
                .rename_axis("Predicted class")
                .reset_index(name="Records")
            )
            st.bar_chart(
                distribution.set_index("Predicted class")["Records"]
            )

            status_filter = st.selectbox(
                "Filter results",
                ["All rows", "Alerts only", "BENIGN only"],
            )

            if status_filter == "Alerts only":
                visible_results = results[
                    results["Detection_Status"] == "ALERT"
                ]
            elif status_filter == "BENIGN only":
                visible_results = results[
                    results["Detection_Status"] == "BENIGN"
                ]
            else:
                visible_results = results

            st.dataframe(
                visible_results.head(1000),
                use_container_width=True,
                hide_index=True,
            )

            if len(visible_results) > 1000:
                st.caption(
                    "The table shows the first 1,000 matching rows. "
                    "The download includes all matching rows."
                )

            csv_bytes = visible_results.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                label="Download results as CSV",
                data=csv_bytes,
                file_name="ids_predictions.csv",
                mime="text/csv",
            )

            st.caption(
                "ALERT means the model predicted a class other than BENIGN. "
                "It is a model prediction, not proof that malicious activity "
                "occurred. Performance on traffic types outside the training "
                "classes is not guaranteed."
            )
