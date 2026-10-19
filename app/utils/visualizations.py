"""
Senior Data Analyst Visualization Suite for Talent Match Intelligence.
Features stacked multi-line typography (<br>) and generous adaptive spacing
to ensure text and labels never collide with chart elements.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.config import BRAND_COLORS


def _wrap_label(text: str, max_words: int = 2) -> str:
    """Wraps text into stacked multi-line strings (<br>) to prevent crowding against charts."""
    words = str(text).replace("_", " ").split()
    if len(words) <= max_words:
        return " ".join(words)
    lines = []
    for i in range(0, len(words), max_words):
        lines.append(" ".join(words[i:i + max_words]))
    return "<br>".join(lines)


def _apply_pro_layout(
    fig: go.Figure,
    title: str = "",
    left_margin: int = 70,
    bottom_margin: int = 65,
    top_margin: int = 85,
    right_margin: int = 50
) -> go.Figure:
    """Applies modern corporate styling with generous breathing room between text and chart elements."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "y": 0.97,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 15, "family": "Plus Jakarta Sans, Inter, sans-serif"}
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=left_margin, r=right_margin, t=top_margin, b=bottom_margin),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0.05)",
            font=dict(size=11)
        ),
        font=dict(family="Plus Jakarta Sans, Inter, Segoe UI, sans-serif")
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.15)",
        title_standoff=16
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.15)",
        title_standoff=16
    )
    return fig


# ------------------------------------------------------------
# 1. STRATEGIC 9-BOX TALENT GRID (PERFORMANCE VS POTENTIAL)
# ------------------------------------------------------------
def plot_9box_grid(grid_df: pd.DataFrame, name_col: str = "fullname") -> go.Figure:
    """
    Senior Executive Tool: 9-Box Talent Review Grid.
    Stacked quadrant labels and generous margins ensure clear legibility.
    """
    if grid_df.empty or "performance_score" not in grid_df.columns or "potential_score" not in grid_df.columns:
        return go.Figure()

    sample_df = grid_df.sample(n=min(len(grid_df), 1200), random_state=42) if len(grid_df) > 1200 else grid_df

    fig = px.scatter(
        sample_df,
        x="performance_score",
        y="potential_score",
        color="quadrant",
        hover_name=name_col if name_col in sample_df.columns else None,
        hover_data={"performance_score": ":.2f", "potential_score": ":.1f", "quadrant": True},
        color_discrete_map={
            "🌟 Star Talent": "#10B981",
            "🚀 High Potential": "#06B6D4",
            "💎 High Performer": "#6366F1",
            "🎯 Core Performer": "#8B5CF6",
            "🛡️ Solid Professional": "#3B82F6",
            "💡 Rough Diamond": "#F59E0B",
            "⏳ Effective Operator": "#94A3B8",
            "🔄 Inconsistent Performer": "#F97316",
            "⚠️ Talent Risk": "#EF4444"
        },
        labels={
            "performance_score": "Performance Rating (Scale 1.0 - 5.0)",
            "potential_score": "Potential Capacity Index (0 - 100)"
        },
        title="Strategic 9-Box Talent Review Matrix"
    )

    fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color="white")))

    # Division Lines
    fig.add_vline(x=2.8, line=dict(color="rgba(148, 163, 184, 0.35)", dash="dash", width=1.5))
    fig.add_vline(x=3.8, line=dict(color="rgba(148, 163, 184, 0.35)", dash="dash", width=1.5))
    fig.add_hline(y=45, line=dict(color="rgba(148, 163, 184, 0.35)", dash="dash", width=1.5))
    fig.add_hline(y=65, line=dict(color="rgba(148, 163, 184, 0.35)", dash="dash", width=1.5))

    # Stacked Quadrant Labels with Subtitles
    annotations = [
        dict(x=4.4, y=86, text="<b>🌟 Star Talent</b><br><span style='font-size:10px;color:#10B981;'>High Perf • High Pot</span>", showarrow=False, align="center"),
        dict(x=3.3, y=86, text="<b>🚀 High Potential</b><br><span style='font-size:10px;color:#06B6D4;'>Med Perf • High Pot</span>", showarrow=False, align="center"),
        dict(x=1.9, y=86, text="<b>💡 Rough Diamond</b><br><span style='font-size:10px;color:#F59E0B;'>Low Perf • High Pot</span>", showarrow=False, align="center"),
        dict(x=4.4, y=55, text="<b>💎 High Performer</b><br><span style='font-size:10px;color:#6366F1;'>High Perf • Med Pot</span>", showarrow=False, align="center"),
        dict(x=3.3, y=55, text="<b>🎯 Core Backbone</b><br><span style='font-size:10px;color:#8B5CF6;'>Med Perf • Med Pot</span>", showarrow=False, align="center"),
        dict(x=1.9, y=55, text="<b>🔄 Inconsistent</b><br><span style='font-size:10px;color:#F97316;'>Low Perf • Med Pot</span>", showarrow=False, align="center"),
        dict(x=4.4, y=24, text="<b>🛡️ Solid Pro</b><br><span style='font-size:10px;color:#3B82F6;'>High Perf • Low Pot</span>", showarrow=False, align="center"),
        dict(x=3.3, y=24, text="<b>⏳ Effective Op</b><br><span style='font-size:10px;color:#94A3B8;'>Med Perf • Low Pot</span>", showarrow=False, align="center"),
        dict(x=1.9, y=24, text="<b>⚠️ Talent Risk</b><br><span style='font-size:10px;color:#EF4444;'>Low Perf • Low Pot</span>", showarrow=False, align="center"),
    ]
    for ann in annotations:
        fig.add_annotation(**ann)

    fig.update_xaxes(range=[0.8, 5.2])
    fig.update_yaxes(range=[0, 105])
    return _apply_pro_layout(fig, "Strategic 9-Box Talent Review Grid", left_margin=70, bottom_margin=70, top_margin=85)


# ------------------------------------------------------------
# 2. PREDICTIVE PERFORMANCE DRIVERS (LOLLIPOP CHART)
# ------------------------------------------------------------
def plot_predictive_drivers_lollipop(importance_df: pd.DataFrame, r2_score: float = 0.0) -> go.Figure:
    """
    Horizontal Lollipop chart of Random Forest feature importance.
    Labels are wrapped and given generous left margin to prevent crowding.
    """
    if importance_df.empty:
        return go.Figure()

    df_sorted = importance_df.sort_values("ImportancePct", ascending=True).copy()
    # Apply stacked multi-line label formatting
    df_sorted["StackedFeature"] = df_sorted["Feature"].apply(lambda f: _wrap_label(f, max_words=2))

    fig = go.Figure()

    # Stems
    for _, row in df_sorted.iterrows():
        fig.add_trace(go.Scatter(
            x=[0, row["ImportancePct"]],
            y=[row["StackedFeature"], row["StackedFeature"]],
            mode="lines",
            line=dict(color="rgba(99, 102, 241, 0.45)", width=3),
            showlegend=False,
            hoverinfo="skip"
        ))

    # Markers (Lollipop Heads)
    fig.add_trace(go.Scatter(
        x=df_sorted["ImportancePct"],
        y=df_sorted["StackedFeature"],
        mode="markers+text",
        marker=dict(size=14, color=BRAND_COLORS["primary"], line=dict(color="white", width=2)),
        text=df_sorted["ImportancePct"].apply(lambda x: f"  {x:.1f}%"),
        textposition="middle right",
        name="Importance",
        hoverinfo="x+y"
    ))

    fig.update_layout(
        xaxis_title="Statistical Explanatory Weight (% Feature Importance)",
        yaxis_title="Assessment Dimension"
    )
    return _apply_pro_layout(
        fig,
        f"Machine Learning: Predictive Performance Drivers (R² = {r2_score:.2f})",
        left_margin=150,
        right_margin=80,
        top_margin=85,
        bottom_margin=70
    )


# ------------------------------------------------------------
# 3. EXECUTIVE PERFORMANCE TIER DONUT CHART
# ------------------------------------------------------------
def plot_performance_tier_donut(df: pd.DataFrame, rating_col: str = "rating") -> go.Figure:
    """
    Executive Donut Chart with stacked tier descriptions.
    """
    if rating_col not in df.columns or df[rating_col].dropna().empty:
        return go.Figure()

    ratings = df[rating_col].dropna()

    def categorize_tier(val: float) -> str:
        if val >= 4.5:
            return "🌟 Tier 5<br>Outstanding (4.5-5.0)"
        elif val >= 3.5:
            return "💎 Tier 4<br>Exceeds Exp. (3.5-4.4)"
        elif val >= 2.5:
            return "🎯 Tier 3<br>Meets Exp. (2.5-3.4)"
        elif val >= 1.5:
            return "⚠️ Tier 2<br>Needs Impr. (1.5-2.4)"
        else:
            return "🚨 Tier 1<br>Unsatisfactory (< 1.5)"

    tier_series = ratings.apply(categorize_tier).value_counts().reset_index()
    tier_series.columns = ["Tier", "Headcount"]

    fig = px.pie(
        tier_series,
        values="Headcount",
        names="Tier",
        hole=0.55,
        color="Tier",
        color_discrete_map={
            "🌟 Tier 5<br>Outstanding (4.5-5.0)": "#10B981",
            "💎 Tier 4<br>Exceeds Exp. (3.5-4.4)": "#6366F1",
            "🎯 Tier 3<br>Meets Exp. (2.5-3.4)": "#3B82F6",
            "⚠️ Tier 2<br>Needs Impr. (1.5-2.4)": "#F59E0B",
            "🚨 Tier 1<br>Unsatisfactory (< 1.5)": "#EF4444"
        },
        title="Executive Performance Tier Breakdown"
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return _apply_pro_layout(fig, "Annual Performance Rating Distribution", left_margin=50, right_margin=50, top_margin=85, bottom_margin=60)


# ------------------------------------------------------------
# 4. CROSS-DEPARTMENT CAPABILITY VIOLIN & BOX PLOT
# ------------------------------------------------------------
def plot_department_competency_violin(
    df: pd.DataFrame,
    metric_col: str = "avg_competency_score",
    group_col: str = "department_id"
) -> go.Figure:
    """Violin distribution comparing capability spread across departments with stacked labels."""
    if metric_col not in df.columns or group_col not in df.columns:
        return go.Figure()

    clean_data = df.dropna(subset=[metric_col, group_col]).copy()
    clean_data[group_col] = "Dept " + clean_data[group_col].astype(str)

    top_depts = clean_data[group_col].value_counts().head(6).index.tolist()
    sub_df = clean_data[clean_data[group_col].isin(top_depts)]

    fig = px.violin(
        sub_df,
        x=group_col,
        y=metric_col,
        color=group_col,
        box=True,
        points="outliers",
        labels={
            metric_col: _wrap_label(metric_col.replace("_", " ").title()),
            group_col: "Organizational Unit"
        },
        title=f"{metric_col.replace('_', ' ').title()} Dispersion"
    )
    fig.update_layout(showlegend=False)
    return _apply_pro_layout(fig, f"{metric_col.replace('_', ' ').title()} Dispersion Across Key Departments", left_margin=70, bottom_margin=70, top_margin=85)


# ------------------------------------------------------------
# 5. COMPETENCY GAP WATERFALL CHART
# ------------------------------------------------------------
def plot_competency_gap_waterfall(
    candidate_name: str,
    categories: List[str],
    candidate_scores: List[float],
    benchmark_scores: List[float]
) -> go.Figure:
    """Waterfall chart with stacked category labels so words never hit the bars."""
    stacked_cats = [_wrap_label(c.replace("_", " ").title(), max_words=2) for c in categories]
    deltas = [round((c - b) * 100, 1) for c, b in zip(candidate_scores, benchmark_scores)]

    fig = go.Figure(go.Waterfall(
        name="Gap vs Benchmark",
        orientation="v",
        measure=["relative"] * len(stacked_cats),
        x=stacked_cats,
        text=[f"{'+' if d >= 0 else ''}{d:.1f}%" for d in deltas],
        y=deltas,
        textposition="outside",
        connector={"line": {"color": "rgba(148, 163, 184, 0.5)"}},
        decreasing={"marker": {"color": "#EF4444"}},
        increasing={"marker": {"color": "#10B981"}}
    ))

    fig.update_layout(
        yaxis_title="Competency Delta vs Cohort Mean (%)",
        xaxis_title="Assessment Dimension"
    )
    return _apply_pro_layout(fig, f"Competency Gap Waterfall: {candidate_name}", left_margin=80, bottom_margin=90, top_margin=85)


# ------------------------------------------------------------
# 6. SUCCESSION READINESS FUNNEL PYRAMID
# ------------------------------------------------------------
def plot_succession_pyramid(pyramid_df: pd.DataFrame) -> go.Figure:
    """Succession readiness funnel with stacked tier descriptions."""
    if pyramid_df.empty:
        return go.Figure()

    fig = px.funnel(
        pyramid_df,
        y="ReadinessTier",
        x="Headcount",
        color="ReadinessTier",
        color_discrete_sequence=["#10B981", "#6366F1", "#3B82F6", "#94A3B8"],
        title="Enterprise Succession Readiness Funnel"
    )
    fig.update_traces(textinfo="value+percent initial")
    fig.update_layout(showlegend=False)
    return _apply_pro_layout(fig, "Enterprise Succession Pipeline Pyramid", left_margin=80, bottom_margin=70, top_margin=85)


# ------------------------------------------------------------
# 7. MULTI-LAYER HEAD-TO-HEAD DUEL RADAR
# ------------------------------------------------------------
def plot_head_to_head_radar(
    cand1_name: str,
    cand1_scores: List[float],
    cand2_name: str,
    cand2_scores: List[float],
    bench_scores: List[float],
    categories: List[str]
) -> go.Figure:
    """Multi-layered Head-to-Head Duel Radar with stacked radial theta labels."""
    formatted_cats = [_wrap_label(c.replace("_", " ").title(), max_words=1) for c in categories]
    cats_closed = formatted_cats + [formatted_cats[0]]
    c1_closed = cand1_scores + [cand1_scores[0]]
    c2_closed = cand2_scores + [cand2_scores[0]]
    b_closed = bench_scores + [bench_scores[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=b_closed,
        theta=cats_closed,
        fill='toself',
        fillcolor='rgba(148, 163, 184, 0.1)',
        line=dict(color='#94A3B8', width=1.5, dash='dash'),
        name='Cohort Benchmark'
    ))

    fig.add_trace(go.Scatterpolar(
        r=c1_closed,
        theta=cats_closed,
        fill='toself',
        fillcolor='rgba(79, 70, 229, 0.25)',
        line=dict(color=BRAND_COLORS["primary"], width=2.5),
        name=f"Candidate A: {cand1_name}"
    ))

    fig.add_trace(go.Scatterpolar(
        r=c2_closed,
        theta=cats_closed,
        fill='toself',
        fillcolor='rgba(6, 182, 212, 0.25)',
        line=dict(color=BRAND_COLORS["secondary"], width=2.5),
        name=f"Candidate B: {cand2_name}"
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.05]), bgcolor="rgba(0,0,0,0)")
    )
    return _apply_pro_layout(fig, f"Head-to-Head Duel: {cand1_name} vs {cand2_name}", left_margin=60, bottom_margin=70, top_margin=85)


# ------------------------------------------------------------
# 8. 2D UNSUPERVISED ML PCA CLUSTER MAP
# ------------------------------------------------------------
def plot_pca_cluster_map(
    pca_df: pd.DataFrame,
    name_col: str = "fullname",
    rating_col: Optional[str] = "rating"
) -> go.Figure:
    """2D Unsupervised ML Projection (PCA + K-Means)."""
    if "pca_x" not in pca_df.columns or "pca_y" not in pca_df.columns:
        return go.Figure()

    sample_df = pca_df.sample(n=min(len(pca_df), 1500), random_state=42) if len(pca_df) > 1500 else pca_df

    hover_data = {name_col: True, "archetype": True, "pca_x": False, "pca_y": False}
    if rating_col and rating_col in sample_df.columns:
        hover_data[rating_col] = True

    fig = px.scatter(
        sample_df,
        x="pca_x",
        y="pca_y",
        color="archetype",
        hover_name=name_col if name_col in sample_df.columns else None,
        hover_data=hover_data,
        color_discrete_sequence=["#4F46E5", "#06B6D4", "#10B981", "#F59E0B"],
        labels={"pca_x": "PC 1 (Cognitive & Core Skills)", "pca_y": "PC 2 (Execution Endurance)"},
        title="Talent Archetype Landscape (2D PCA & K-Means Clusters)"
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75, line=dict(width=0.5, color="white")))
    return _apply_pro_layout(fig, "Talent Archetype Landscape (2D PCA + K-Means)", left_margin=70, bottom_margin=70, top_margin=85)


# ------------------------------------------------------------
# 9. RADAR & BAR HELPERS
# ------------------------------------------------------------
def plot_radar_comparison(
    candidate_name: str,
    categories: List[str],
    candidate_scores: List[float],
    benchmark_scores: List[float]
) -> go.Figure:
    """Single candidate vs benchmark radar chart with stacked labels."""
    formatted_cats = [_wrap_label(c.replace("_", " ").title(), max_words=1) for c in categories]
    cats_closed = formatted_cats + [formatted_cats[0]]
    cand_closed = candidate_scores + [candidate_scores[0]]
    bench_closed = benchmark_scores + [benchmark_scores[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=bench_closed,
        theta=cats_closed,
        fill='toself',
        fillcolor='rgba(148, 163, 184, 0.2)',
        line=dict(color='#94A3B8', width=2, dash='dot'),
        name='Cohort Benchmark'
    ))

    fig.add_trace(go.Scatterpolar(
        r=cand_closed,
        theta=cats_closed,
        fill='toself',
        fillcolor='rgba(79, 70, 229, 0.3)',
        line=dict(color=BRAND_COLORS["primary"], width=2.5),
        name=candidate_name
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.05]), bgcolor="rgba(0,0,0,0)")
    )
    return _apply_pro_layout(fig, f"Assessment Profile: {candidate_name}", left_margin=60, bottom_margin=70, top_margin=85)


def plot_top_candidates_bar(
    ranked_df: pd.DataFrame,
    top_n: int = 10,
    name_col: str = "fullname",
    score_col: str = "final_match_rate"
) -> go.Figure:
    """Horizontal bar chart for top N ranked candidates with gradient color."""
    if ranked_df.empty:
        return go.Figure()

    subset = ranked_df.head(top_n).sort_values(score_col, ascending=True).copy()
    display_names = subset[name_col].astype(str) if name_col in subset.columns else subset.index.astype(str)

    fig = px.bar(
        subset,
        x=score_col,
        y=display_names,
        orientation="h",
        text=subset[score_col].apply(lambda x: f"{x:.1f}%"),
        color=score_col,
        color_continuous_scale="Viridis",
        labels={score_col: "Match Rate (%)", "y": "Talent Name"},
        title=f"Top {len(subset)} Ranked Talents"
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    return _apply_pro_layout(fig, f"Top {len(subset)} Ranked Talents", left_margin=170, bottom_margin=70, top_margin=85)


def plot_rank_sensitivity(
    sensitivity_df: pd.DataFrame,
    top_n: int = 15,
    name_col: str = "fullname"
) -> go.Figure:
    """Rank sensitivity / drift diverging bar chart."""
    if "rank_delta" not in sensitivity_df.columns:
        return go.Figure()

    subset = sensitivity_df.head(top_n).copy()
    subset["color"] = subset["rank_delta"].apply(lambda x: "#10B981" if x > 0 else ("#EF4444" if x < 0 else "#94A3B8"))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=subset["rank_delta"],
        y=subset[name_col].astype(str),
        orientation="h",
        marker_color=subset["color"],
        text=subset["rank_delta"].apply(lambda x: f"+{x}" if x > 0 else str(x)),
        textposition="outside"
    ))

    fig.update_layout(xaxis_title="Leaderboard Position Drift (Delta Ranks)", yaxis_title="Candidate", bargap=0.25)
    return _apply_pro_layout(fig, "Leaderboard Sensitivity Drift (What-If Impact)", left_margin=170, bottom_margin=70, top_margin=85)


def plot_correlation_matrix(df: pd.DataFrame, columns: List[str]) -> go.Figure:
    """Correlation matrix heatmap with stacked multi-line labels."""
    valid_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 2:
        return go.Figure()

    corr = df[valid_cols].corr()
    stacked_labels = [_wrap_label(c.replace("_", " ").title(), max_words=2) for c in valid_cols]

    fig = px.imshow(
        corr,
        x=stacked_labels,
        y=stacked_labels,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Inter-Metric Correlation Matrix"
    )
    return _apply_pro_layout(fig, "Inter-Metric Correlation Matrix", left_margin=120, bottom_margin=100, top_margin=85)