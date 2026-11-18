"""IrisVision AI — local Streamlit dashboard for Iris classification."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from irisvision.data import FEATURE_NAMES, SPECIES_NAMES, load_dataset
from irisvision.evaluation import evaluate_models
from irisvision.models import MODEL_ORDER
from irisvision.prediction import measurements_outside_training_range, predict_flower
from irisvision.visualizations import (
    comparison_metrics,
    confusion_heatmap,
    feature_distributions,
    feature_relationships,
    probabilities_chart,
    runtime_comparison,
    species_distribution,
)

st.set_page_config(page_title="IrisVision AI", page_icon="✿", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --iv-bg:#0b1220; --iv-panel:#111c2f; --iv-line:#24324a; --iv-blue:#60a5fa; --iv-mint:#34d399; --iv-muted:#91a3bd; }
.stApp { background: radial-gradient(ellipse at 15% -5%, #172b4a 0%, #0b1220 42%); }
[data-testid="stSidebar"] { background: #0a1020; border-right: 1px solid #1e2b40; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1450px; }
h1,h2,h3 { letter-spacing: -0.035em; }
.iv-eyebrow { color:#93c5fd; text-transform:uppercase; font-size:.73rem; font-weight:700; letter-spacing:.16em; margin-bottom:.35rem; }
.iv-subtitle { color:#a6b5ca; font-size:1.03rem; max-width:850px; line-height:1.6; }
.iv-card { background:linear-gradient(145deg,rgba(21,34,55,.96),rgba(15,25,42,.96)); border:1px solid #24334b; border-radius:16px; padding:19px 20px; min-height:122px; }
.iv-label { color:#9aacc4; font-size:.81rem; font-weight:600; letter-spacing:.02em; }
.iv-value { color:#f4f7fb; font-size:1.85rem; line-height:1.2; font-weight:720; margin-top:10px; }
.iv-note { color:#91a3bd; font-size:.8rem; margin-top:7px; }
.iv-banner { background:linear-gradient(100deg,rgba(37,99,235,.2),rgba(16,185,129,.09)); border:1px solid rgba(96,165,250,.3); border-radius:18px; padding:22px 24px; }
.iv-result { background:linear-gradient(120deg,rgba(37,99,235,.2),rgba(52,211,153,.08)); border:1px solid rgba(96,165,250,.42); border-radius:18px; padding:22px 24px; }
section[data-testid="stSidebar"] .stRadio label { padding-top:6px; padding-bottom:6px; }
.stButton button { border-radius:10px; font-weight:650; }
[data-testid="stMetric"] { background:#111c2f; border:1px solid #24334b; padding:14px 16px; border-radius:14px; }
[data-testid="stDataFrame"] { border:1px solid #24334b; border-radius:12px; overflow:hidden; }
hr { border-color:#24334b; }
small, .stCaption { color:#91a3bd !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training and evaluating the four classifiers…")
def get_evaluation():
    return evaluate_models()


def section_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="iv-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="iv-subtitle">{description}</div>', unsafe_allow_html=True)
    st.write("")


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="iv-card"><div class="iv-label">{label}</div>'
        f'<div class="iv-value">{value}</div><div class="iv-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def render_overview(frame: pd.DataFrame, active_model: str, metrics: pd.DataFrame) -> None:
    section_header("IRIS CLASSIFICATION WORKBENCH", "IrisVision AI", "A transparent, local-first machine-learning workbench for exploring the classic Iris dataset and comparing four trained classifiers.")
    st.markdown(f"""
    <div class="iv-banner"><div class="iv-eyebrow">Active model · {active_model}</div>
    <b>From measurements to a model-backed prediction.</b><br>
    <span style="color:#a6b5ca">Explore the original records, compare validation and holdout results, then test a flower measurement with a trained pipeline.</span></div>
    """, unsafe_allow_html=True)
    st.write("")
    a, b, c, d = st.columns(4)
    with a: metric_card("DATASET RECORDS", f"{len(frame)}", "Canonical Iris observations")
    with b: metric_card("INPUT FEATURES", f"{len(FEATURE_NAMES)}", "Measurements in centimeters")
    with c: metric_card("FLOWER SPECIES", f"{frame['species'].nunique()}", "Setosa · versicolor · virginica")
    active_row = metrics.loc[metrics["Model"] == active_model].iloc[0]
    with d: metric_card("HOLDOUT ACCURACY", f"{active_row['Test accuracy']:.1%}", "One stratified 30-record test split")
    st.write("")
    left, right = st.columns([1.12, .88], gap="large")
    with left:
        st.subheader("Species in the source dataset")
        st.plotly_chart(species_distribution(frame), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.subheader("Evaluation, at a glance")
        st.markdown("""
        **Selection:** stratified 5-fold cross-validation on the training portion only.  
        **Final check:** a stratified 80/20 holdout, reserved until tuning is complete.  
        **Reported scores:** macro precision, recall and F1 treat each species equally.  
        **Purpose:** a compact educational dataset; scores do not guarantee performance on other data.
        """)
        st.info("The original KNN baseline remains available. Standardization is fitted inside each training fold for KNN and SVM; the tree-based models use their unscaled inputs.")


def render_prediction(active_model: str, result) -> None:
    section_header("LIVE INFERENCE", "Interactive prediction", "Enter the four flower measurements in centimeters. The selected, cross-validated model—not a hard-coded rule—produces the prediction.")
    st.caption(f"Selected pipeline: **{active_model}**. The initial values are copied from the first real row of the included Iris dataset.")
    mode = st.radio("Measurement entry", ["Sliders", "Type values"], horizontal=True, key="measurement_mode")
    default = load_dataset().iloc[0]
    columns = st.columns(2, gap="large")
    values: list[float] = []
    for idx, feature in enumerate(FEATURE_NAMES):
        low = float(load_dataset()[feature].min())
        high = float(load_dataset()[feature].max())
        with columns[idx % 2]:
            label = feature.replace(" (cm)", "").title() + " (cm)"
            if mode == "Sliders":
                value = st.slider(label, min_value=low, max_value=high, value=float(default[feature]), step=0.1, key=f"slider_{idx}")
            else:
                value = st.number_input(label, min_value=0.0, value=float(default[feature]), step=0.1, format="%.2f", key=f"manual_{idx}")
            values.append(float(value))
    submitted = st.button("Predict flower species", type="primary", use_container_width=False)
    if submitted:
        try:
            outcome = predict_flower(result.models[active_model], values)
            st.session_state["last_prediction"] = {"model": active_model, "values": values, **outcome}
        except ValueError as exc:
            st.error(str(exc))
    saved = st.session_state.get("last_prediction")
    if saved:
        if saved["model"] != active_model:
            st.info("The active model changed. Run prediction again to refresh the result with this model.")
        else:
            outside = measurements_outside_training_range(saved["values"], result.X_train)
            if outside:
                st.warning("One or more measurements are outside the training-data range. The classifier can still return a class, but this is an extrapolation and should be interpreted cautiously.")
            st.write("")
            st.markdown(f"<div class='iv-result'><div class='iv-eyebrow'>Model output · {saved['model']}</div><div style='font-size:1.8rem;font-weight:750'>Predicted species: {saved['species'].title()}</div><div style='color:#a6b5ca'>The classifier evaluated the four entered measurements against patterns learned from the training records.</div></div>", unsafe_allow_html=True)
            probabilities = saved["probabilities"]
            if probabilities:
                st.subheader("Model probability estimates")
                st.plotly_chart(probabilities_chart(probabilities), use_container_width=True, config={"displayModeBar": False})
                st.caption("These are model-produced probability estimates, not calibrated guarantees or causal explanations. SVM probabilities use scikit-learn's probability estimation.")
            with st.expander("Prediction details"):
                st.write(pd.DataFrame({"Measurement": FEATURE_NAMES, "Value (cm)": saved["values"]}))
                st.write(f"Class selected by the trained {saved['model']} pipeline: **{saved['species']}**.")


def render_comparison(metrics: pd.DataFrame) -> None:
    section_header("BENCHMARKS", "Model comparison", "All four classifiers use the same stratified split and the same training-only cross-validation procedure. Final holdout scores are shown separately from cross-validation.")
    st.warning("Cross-validation macro F1 is used to select settings on training folds. The final test metrics come from the reserved holdout and should not be used to retune the models. Training/search time is hardware-dependent.")
    display = metrics[["Model", "CV macro F1", "Test accuracy", "Test precision (macro)", "Test recall (macro)", "Test F1 (macro)", "Training time (s)", "Prediction time (ms/sample)"]].copy()
    st.dataframe(display.style.format({col: "{:.3f}" for col in display.columns if col != "Model"}), use_container_width=True, hide_index=True)
    first, second = st.columns(2, gap="large")
    with first:
        st.subheader("Final holdout scores")
        st.plotly_chart(comparison_metrics(metrics), use_container_width=True, config={"displayModeBar": False})
    with second:
        st.subheader("Runtime snapshot")
        st.plotly_chart(runtime_comparison(metrics), use_container_width=True, config={"displayModeBar": False})
    with st.expander("Training-only selected settings"):
        for _, row in metrics.iterrows():
            st.markdown(f"**{row['Model']}** — `{row['Best parameters']}`")


def render_explorer(frame: pd.DataFrame) -> None:
    section_header("SOURCE DATA", "Dataset explorer", "Browse the real, included Iris records. Filter the species for the preview and explore the measured feature distributions and relationships.")
    selected = st.multiselect("Filter species", SPECIES_NAMES, default=SPECIES_NAMES, key="species_filter")
    filtered = frame.loc[frame["species"].isin(selected)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Filtered records", len(filtered))
    m2.metric("Species selected", len(selected))
    m3.metric("Missing values", int(filtered.isna().sum().sum()))
    st.subheader("Dataset preview")
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=300)
    if filtered.empty:
        st.info("Select at least one species to display charts.")
        return
    left, right = st.columns([.8, 1.2], gap="large")
    with left:
        st.subheader("Species distribution")
        st.plotly_chart(species_distribution(filtered), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.subheader("Feature distributions")
        st.plotly_chart(feature_distributions(filtered), use_container_width=True, config={"displayModeBar": False})
    st.subheader("Pairwise feature relationships")
    st.plotly_chart(feature_relationships(filtered), use_container_width=True, config={"displayModeBar": False})
    st.caption("Each point is a record from the bundled dataset; color denotes its recorded species.")


def render_evaluation(result, active_model: str) -> None:
    section_header("MODEL DIAGNOSTICS", "Model evaluation", "Inspect class-level metrics and errors on the same reserved test records. Confusion-matrix rows are actual species and columns are predicted species.")
    model = active_model
    row = result.metrics.loc[result.metrics["Model"] == model].iloc[0]
    cols = st.columns(4)
    for col, label, key in zip(cols, ["Accuracy", "Macro precision", "Macro recall", "Macro F1"], ["Test accuracy", "Test precision (macro)", "Test recall (macro)", "Test F1 (macro)"]):
        col.metric(label, f"{row[key]:.1%}")
    st.markdown(f"**Current model:** {model}. Test set size: {len(result.y_test)} records; the data was not used to select hyperparameters.")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Confusion matrix")
        st.plotly_chart(confusion_heatmap(result.confusion_matrices[model], SPECIES_NAMES), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.subheader("Classification report")
        report = result.reports[model]
        class_rows = pd.DataFrame(report).T.loc[SPECIES_NAMES, ["precision", "recall", "f1-score", "support"]]
        st.dataframe(class_rows.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:.0f}"}), use_container_width=True)
        st.caption("Macro averaging gives each species equal weight; support is the number of test records for that species.")
    details = result.X_test.copy()
    details.insert(0, "Actual species", result.y_test.to_numpy())
    details["Predicted species"] = result.predictions[model]
    details["Correct"] = details["Actual species"] == details["Predicted species"]
    st.subheader("Actual versus predicted test records")
    st.dataframe(details.reset_index(drop=True), use_container_width=True, hide_index=True)
    incorrect = int((~details["Correct"]).sum())
    st.info(f"On this reserved test split, **{incorrect} of {len(details)}** records were misclassified. This small benchmark is illustrative; a top score on this split does not guarantee generalization to other populations or measurement conditions.")


frame = load_dataset()
evaluation = get_evaluation()
with st.sidebar:
    st.markdown("## ✿ IrisVision AI")
    st.caption("Local ML workbench · v1.0.0")
    section = st.radio("WORKSPACE", ["Dashboard Overview", "Interactive Prediction", "Model Comparison", "Dataset Explorer", "Model Evaluation"], label_visibility="collapsed")
    st.divider()
    selected_model = st.selectbox("Active classifier", MODEL_ORDER, index=0)
    st.caption("Four pipelines · no external APIs")

if section == "Dashboard Overview":
    render_overview(frame, selected_model, evaluation.metrics)
elif section == "Interactive Prediction":
    render_prediction(selected_model, evaluation)
elif section == "Model Comparison":
    render_comparison(evaluation.metrics)
elif section == "Dataset Explorer":
    render_explorer(frame)
else:
    render_evaluation(evaluation, selected_model)

st.divider()
st.caption("IrisVision AI · Reproducible local analysis · Values and scores are specific to the included data and evaluation split.")
