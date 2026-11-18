"""Plotly visualizations for the IrisVision dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import FEATURE_NAMES

NAVY = "#0b1220"
BLUE = "#60a5fa"
TEAL = "#34d399"
PALETTE = {"setosa": "#60a5fa", "versicolor": "#34d399", "virginica": "#fbbf24"}


def species_distribution(frame: pd.DataFrame):
    counts = frame["species"].value_counts().reindex(["setosa", "versicolor", "virginica"]).rename_axis("Species").reset_index(name="Records")
    fig = px.bar(counts, x="Species", y="Records", color="Species", color_discrete_map=PALETTE)
    fig.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=22, b=10), yaxis_title="Real dataset records")
    return fig


def feature_distributions(frame: pd.DataFrame):
    long = frame.melt(id_vars="species", value_vars=FEATURE_NAMES, var_name="Measurement", value_name="Centimeters")
    fig = px.histogram(long, x="Centimeters", color="species", facet_col="Measurement", facet_col_wrap=2, barmode="overlay", opacity=0.72, color_discrete_map=PALETTE, category_orders={"species": ["setosa", "versicolor", "virginica"]})
    fig.update_layout(height=590, bargap=0.08, legend_title_text="Species", margin=dict(l=10, r=10, t=45, b=10))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def feature_relationships(frame: pd.DataFrame):
    fig = px.scatter_matrix(frame, dimensions=FEATURE_NAMES, color="species", color_discrete_map=PALETTE, opacity=0.75, category_orders={"species": ["setosa", "versicolor", "virginica"]})
    fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=5))
    fig.update_layout(height=690, margin=dict(l=12, r=12, t=28, b=12), legend_title_text="Species")
    return fig


def comparison_metrics(metrics: pd.DataFrame):
    metric_columns = ["Test accuracy", "Test precision (macro)", "Test recall (macro)", "Test F1 (macro)"]
    fig = px.bar(metrics, x="Model", y=metric_columns, barmode="group", color_discrete_sequence=[BLUE, TEAL, "#fbbf24"], labels={"value": "Score", "variable": "Metric"})
    fig.update_yaxes(range=[0, 1.05], title="Macro-averaged score")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=26, b=10), legend_title_text="Final test metric")
    return fig


def runtime_comparison(metrics: pd.DataFrame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=metrics["Model"], y=metrics["Training time (s)"], name="Training + CV search (s)", marker_color=BLUE), secondary_y=False)
    fig.add_trace(go.Scatter(x=metrics["Model"], y=metrics["Prediction time (ms/sample)"], name="Prediction (ms/sample)", mode="lines+markers", line=dict(color=TEAL, width=3), marker=dict(size=8)), secondary_y=True)
    fig.update_yaxes(title_text="Training + CV search time (s)", secondary_y=False)
    fig.update_yaxes(title_text="Prediction time (ms/sample)", secondary_y=True)
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=28, b=10), legend=dict(orientation="h", y=1.12))
    return fig


def confusion_heatmap(matrix, labels):
    fig = px.imshow(matrix, x=labels, y=labels, text_auto=True, color_continuous_scale=[[0, NAVY], [0.5, "#1d4ed8"], [1, "#93c5fd"]], aspect="auto", labels={"x": "Predicted species", "y": "Actual species", "color": "Records"})
    fig.update_xaxes(side="bottom")
    fig.update_layout(height=370, margin=dict(l=10, r=10, t=22, b=10), coloraxis_showscale=False)
    return fig


def probabilities_chart(probabilities: dict[str, float]):
    values = pd.DataFrame({"Species": list(probabilities), "Estimated probability": list(probabilities.values())})
    fig = px.bar(values, x="Estimated probability", y="Species", orientation="h", color="Species", color_discrete_map=PALETTE, text=values["Estimated probability"].map(lambda x: f"{x:.1%}"))
    fig.update_layout(height=230, xaxis=dict(range=[0, 1], tickformat=".0%"), showlegend=False, margin=dict(l=10, r=10, t=18, b=10))
    fig.update_traces(textposition="outside", cliponaxis=False)
    return fig
