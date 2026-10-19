"""
Talent Match Intelligence System — Senior Data Analyst Edition
Enterprise People Analytics & Predictive Decision Support Platform.
Analytical Frameworks:
- Strategic 9-Box Talent Review Grid (Performance vs Potential)
- Machine Learning Performance Driver Analysis (Random Forest Feature Importance)
- Executive Performance Tier Donut & Capability Violin Spread
- Succession Pipeline Readiness Funnel Pyramid
- Dynamic Weighted Success Formula with Waterfall Gap Diagnostics
- 2D Unsupervised ML (PCA + K-Means) Talent Landscape
- Multi-Layer Head-to-Head Candidate Duel Arena
"""

import sys
import json
import time
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np

# Core Utilities
from app.config import (
    BRAND_COLORS,
    OPENROUTER_MODEL,
    NOTEBOOKS_DIR,
    DATA_DIR
)
from app.utils.scoring import TalentScoringEngine, TalentMLClusterer
from app.utils.visualizations import (
    plot_9box_grid,
    plot_predictive_drivers_lollipop,
    plot_performance_tier_donut,
    plot_department_competency_violin,
    plot_competency_gap_waterfall,
    plot_succession_pyramid,
    plot_top_candidates_bar,
    plot_radar_comparison,
    plot_head_to_head_radar,
    plot_pca_cluster_map,
    plot_rank_sensitivity,
    plot_correlation_matrix
)
from app.utils.llm_generator import LLMGenerator
from app.utils.db_connector import DatabaseConnector
from app.utils.supabase_connector import SupabaseConnector

# ------------------------------------------------------------
# 1. PAGE SETUP & ADAPTIVE STYLING
# ------------------------------------------------------------
st.set_page_config(
    page_title="Talent Match Intelligence — Senior People Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Modern Glassmorphic Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.9) 50%, rgba(49, 46, 129, 0.85) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 24px 28px;
        color: #FFFFFF;
        margin-bottom: 20px;
        box-shadow: 0 12px 30px -10px rgba(49, 46, 129, 0.4);
    }
    .hero-banner h1 {
        font-size: 25px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-banner p {
        font-size: 13.5px;
        color: #C7D2FE;
        margin-top: 6px;
        margin-bottom: 12px;
        line-height: 1.5;
    }
    .hero-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 11.5px;
        font-weight: 500;
        color: #EEF2FF;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* Section Headings */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Executive Insight Callout Box */
    .insight-card {
        background: rgba(99, 102, 241, 0.08);
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 22px;
        font-size: 13.5px;
        line-height: 1.6;
    }

    /* Stacked Chart Headers with Breathing Room */
    .chart-container-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 6px;
        color: #F8FAFC;
    }
    .chart-container-subtitle {
        font-size: 12.5px;
        color: #94A3B8;
        margin-bottom: 18px;
        display: block;
        line-height: 1.4;
    }

    /* Spacious Container Padding */
    [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 20px 24px !important;
        margin-bottom: 22px !important;
        border-radius: 12px !important;
    }

    /* Separation between text widgets and Plotly charts */
    .stPlotlyChart {
        margin-top: 14px !important;
        margin-bottom: 16px !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# 2. SERVICE & DATA INITIALIZATION
# ------------------------------------------------------------
llm_service = LLMGenerator()
db_service = DatabaseConnector()
supabase_service = SupabaseConnector()

if "dataset" not in st.session_state:
    st.session_state.dataset = pd.DataFrame()

if "data_source_name" not in st.session_state:
    st.session_state.data_source_name = "None"


@st.cache_data(show_spinner=False)
def load_sample_dataset() -> pd.DataFrame:
    """Discovers and caches sample dataset for ultra-fast startup."""
    candidate_paths = [
        DATA_DIR / "combined_employee_data.csv",
        NOTEBOOKS_DIR / "combined_employee_data.csv",
        NOTEBOOKS_DIR / "data_contoh.csv",
    ]
    for p in candidate_paths:
        if p.exists():
            try:
                raw = pd.read_csv(p)
                return TalentScoringEngine.clean_talent_data(raw)
            except Exception:
                continue
    return pd.DataFrame()


# Auto-load on first launch
if st.session_state.dataset.empty:
    st.session_state.dataset = load_sample_dataset()
    if not st.session_state.dataset.empty:
        st.session_state.data_source_name = "Enterprise HR Dataset"


# ------------------------------------------------------------
# 3. SIDEBAR: DATA CONTROLLER
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Analytics Controller")
    st.caption("Talent Match Intelligence • Senior Edition")
    st.markdown("---")

    data_source = st.radio(
        "Data Ingestion Source",
        ["Default Demo Dataset", "Upload File (CSV/Excel)", "Supabase REST API"],
        index=0
    )

    if data_source == "Default Demo Dataset":
        if st.button("🔄 Reload Demo Dataset", use_container_width=True):
            st.session_state.dataset = load_sample_dataset()
            st.session_state.data_source_name = "Enterprise HR Dataset"
            st.success("✅ Dataset reloaded!")

    elif data_source == "Upload File (CSV/Excel)":
        uploaded_file = st.file_uploader(
            "Upload Local Dataset",
            type=["csv", "xlsx", "xls"],
            help="Support for clean employee assessment CSV or Excel."
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    raw_df = pd.read_csv(uploaded_file)
                else:
                    raw_df = pd.read_excel(uploaded_file)
                st.session_state.dataset = TalentScoringEngine.clean_talent_data(raw_df)
                st.session_state.data_source_name = uploaded_file.name
                st.success(f"✅ Loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    elif data_source == "Supabase REST API":
        table_name = st.text_input("Supabase Table", "employee_analysis")
        if st.button("⚡ Query Supabase Table", use_container_width=True):
            if not supabase_service.is_configured:
                st.error("SUPABASE_URL and SUPABASE_KEY are not set in `.env`.")
            else:
                with st.spinner("Fetching from Supabase..."):
                    try:
                        res = supabase_service.fetch_table(table_name)
                        if res and hasattr(res, "data") and res.data:
                            st.session_state.dataset = TalentScoringEngine.clean_talent_data(pd.DataFrame(res.data))
                            st.session_state.data_source_name = f"Supabase: {table_name}"
                            st.success(f"✅ Loaded {len(res.data)} records from Supabase!")
                        else:
                            st.warning("Empty result from Supabase.")
                    except Exception as e:
                        st.error(f"Supabase error: {e}")

    st.markdown("---")
    st.markdown("### 🤖 Generative AI")
    if llm_service.is_configured:
        st.markdown(f"🟢 **Online:** `{OPENROUTER_MODEL}`")
    else:
        st.markdown("🟡 **Offline:** Local Mode")
        st.caption("All Machine Learning & mathematical scoring run locally.")

    st.markdown("---")
    st.caption("Crafted by Senior People Analytics standards: 9-Box Grid, ML Driver Modeling, and Succession Planning.")


# ------------------------------------------------------------
# 4. HERO HEADER & SYSTEM TELEMETRY
# ------------------------------------------------------------
df = st.session_state.dataset

if df.empty:
    st.warning("⚠️ No dataset loaded. Please select or upload a dataset from the sidebar.")
    st.stop()

# Ensure clean data
df = TalentScoringEngine.clean_talent_data(df)

# Column mapping
id_col = next((c for c in ["employee_id", "id", "nip", "emp_id"] if c in df.columns), df.columns[0])
name_col = next((c for c in ["fullname", "name", "nama", "employee_name"] if c in df.columns), id_col)
rating_col = "rating" if "rating" in df.columns else None

total_records = len(df)
unique_talents = df[id_col].nunique() if id_col in df.columns else total_records
detected_metrics = TalentScoringEngine.identify_scoring_features(df)

st.markdown(
    f"""
    <div class="hero-banner">
        <h1>🎯 Talent Match Intelligence Platform</h1>
        <p>
            Enterprise Decision Support System combining <b>Strategic 9-Box Talent Reviews</b>, 
            <b>Machine Learning Predictive Driver Modeling</b>, and <b>Succession Pipeline Planning</b>.
        </p>
        <div class="hero-chips">
            <span class="chip">📂 Dataset: <b>{st.session_state.data_source_name}</b></span>
            <span class="chip">👥 Personnel: <b>{total_records:,}</b></span>
            <span class="chip">🧬 Active Assessment Dimensions: <b>{len(detected_metrics)}</b></span>
            <span class="chip">🔬 Model: <b>Random Forest + PCA K-Means</b></span>
            <span class="chip">⚡ Architecture: <b>Modular Enterprise Suite</b></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# 5. ENTERPRISE TABS (6 SENIOR DATA ANALYST VIEWS)
# ------------------------------------------------------------
tab_9box, tab_drivers, tab_succession, tab_archetypes, tab_duel, tab_dev = st.tabs([
    "🎯 9-Box Strategic Review",
    "🔬 Predictive Driver Analysis",
    "🏅 Succession & Match Lab",
    "🧬 Psychometric Archetypes",
    "⚔️ Head-to-Head Duel Arena",
    "💻 Data Architecture & API"
])


# ============================================================
# TAB 1: STRATEGIC 9-BOX TALENT REVIEW GRID
# ============================================================
with tab_9box:
    st.markdown('<div class="section-header">🎯 Strategic 9-Box Talent Review Matrix</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="insight-card">
            <b>Executive Context:</b> The 9-Box Matrix balances <b>Historical Performance</b> (Annual Review Ratings, X-Axis) 
            against <b>Growth Potential</b> (Composite Cognitive Capacity, Learning Agility & Pauli Execution Stamina, Y-Axis). 
            It is the standard framework used in executive talent reviews to identify leadership pipelines and retention priorities.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Computing 9-Box matrix dimensions..."):
        grid_df = TalentScoringEngine.compute_9box_matrix(df)

    if not grid_df.empty:
        # Quadrant Summary KPIs
        q_counts = grid_df["quadrant"].value_counts()
        stars_cnt = q_counts.get("🌟 Star Talent", 0)
        hipo_cnt = q_counts.get("🚀 High Potential", 0)
        hiperf_cnt = q_counts.get("💎 High Performer", 0)
        risk_cnt = q_counts.get("⚠️ Talent Risk", 0)

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            with st.container(border=True):
                st.metric("🌟 Star Talents", f"{stars_cnt:,}", f"{(stars_cnt/len(grid_df))*100:.1f}% (Succession Priority)")
        with q2:
            with st.container(border=True):
                st.metric("🚀 High Potentials", f"{hipo_cnt:,}", f"{(hipo_cnt/len(grid_df))*100:.1f}% (Fast-Track Track)")
        with q3:
            with st.container(border=True):
                st.metric("💎 High Performers", f"{hiperf_cnt:,}", f"{(hiperf_cnt/len(grid_df))*100:.1f}% (Core Functional Experts)")
        with q4:
            with st.container(border=True):
                st.metric("⚠️ Talent Risk / At Risk", f"{risk_cnt:,}", f"{(risk_cnt/len(grid_df))*100:.1f}% (Action Required)", delta_color="inverse")

        # Main 9-Box Chart
        with st.container(border=True):
            fig_9box = plot_9box_grid(grid_df, name_col=name_col)
            st.plotly_chart(fig_9box, use_container_width=True)

        # Quadrant Filter Explorer
        with st.container(border=True):
            st.markdown("#### 🔍 Filter Talents by 9-Box Quadrant")
            all_quadrants = sorted(grid_df["quadrant"].unique().tolist())
            selected_quad = st.selectbox("Select Talent Segment to Inspect Personnel:", all_quadrants, index=0)
            
            segment_df = grid_df[grid_df["quadrant"] == selected_quad]
            show_cols = [id_col, name_col, "performance_score", "potential_score"]
            if "avg_competency_score" in segment_df.columns:
                show_cols.append("avg_competency_score")
            if "iq" in segment_df.columns:
                show_cols.append("iq")
            if "department_id" in segment_df.columns:
                show_cols.append("department_id")

            st.dataframe(segment_df[show_cols].head(25), use_container_width=True)
    else:
        st.info("9-Box review matrix requires 'rating' and numerical assessment columns.")


# ============================================================
# TAB 2: MACHINE LEARNING PREDICTIVE DRIVER ANALYSIS
# ============================================================
with tab_drivers:
    st.markdown('<div class="section-header">🔬 Machine Learning: Predictive Performance Drivers</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="insight-card">
            <b>Senior Data Analyst Diagnostic:</b> Instead of relying on static correlations, we trained a 
            <b>Random Forest Regressor</b> directly on the employee assessment records to answer: 
            <i>"Which psychometric and competency attributes statistically explain annual performance ratings?"</i>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.spinner("Training Random Forest model to calculate feature importances..."):
        drivers_df, r2_score = TalentScoringEngine.compute_predictive_drivers(df, target_col="rating")

    d_col1, d_col2 = st.columns([3, 2])

    with d_col1:
        with st.container(border=True):
            if not drivers_df.empty:
                fig_lollipop = plot_predictive_drivers_lollipop(drivers_df, r2_score=r2_score)
                st.plotly_chart(fig_lollipop, use_container_width=True)
            else:
                st.info("Insufficient data for Random Forest training.")

    with d_col2:
        with st.container(border=True):
            # Executive Performance Tier Donut Chart (Solves the squished histogram problem permanently)
            fig_donut = plot_performance_tier_donut(df, rating_col="rating")
            st.plotly_chart(fig_donut, use_container_width=True)

    # Department Competency Violin Spread
    with st.container(border=True):
        st.markdown("#### 🎻 Cross-Department Capability Dispersion (Violin & Box Distribution)")
        st.caption("Analyzes whether competency standards are equitably distributed across organizational departments.")
        if "department_id" in df.columns:
            fig_violin = plot_department_competency_violin(df, metric_col="avg_competency_score", group_col="department_id")
            st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.info("Department ID column unavailable for cross-unit comparison.")


# ============================================================
# TAB 3: SUCCESSION PIPELINE & MATCH RANKING
# ============================================================
with tab_succession:
    st.markdown('<div class="section-header">🏅 Succession Readiness Pipeline & Dynamic Match Lab</div>', unsafe_allow_html=True)

    # Weight Tuning
    st.markdown("##### ⚙️ Dynamic Formula Calibration")
    weight_cols = st.columns(min(len(detected_metrics), 5))
    calibrated_weights: dict = {}

    default_weights = {
        "avg_competency_score": 0.35,
        "iq": 0.25,
        "years_of_service_months": 0.15,
        "pauli": 0.15,
        "faxtor": 0.10,
        "gtq": 0.15,
        "tiki": 0.10
    }

    for idx, feature in enumerate(detected_metrics[:5]):
        col = weight_cols[idx % len(weight_cols)]
        clean_name = feature.replace("_", " ").title()
        default_val = float(default_weights.get(feature, 0.20))
        with col:
            calibrated_weights[feature] = st.slider(
                clean_name,
                0.0, 1.0, default_val, 0.05,
                key=f"succession_w_{feature}"
            )

    ranked_df, scaled_df, active_features = TalentScoringEngine.evaluate_candidates(
        df,
        weights=calibrated_weights,
        selected_features=list(calibrated_weights.keys())
    )

    pyramid_df = TalentScoringEngine.compute_talent_pyramid(ranked_df)

    s_col1, s_col2 = st.columns([1, 1])

    with s_col1:
        with st.container(border=True):
            st.markdown("#### ⏳ Enterprise Succession Readiness Funnel")
            fig_pyramid = plot_succession_pyramid(pyramid_df)
            st.plotly_chart(fig_pyramid, use_container_width=True)

    with s_col2:
        with st.container(border=True):
            st.markdown("#### 🏆 Top Ranked Candidate Pool")
            fig_top_bar = plot_top_candidates_bar(ranked_df, top_n=10, name_col=name_col)
            st.plotly_chart(fig_top_bar, use_container_width=True)

    # Ranked Leaderboard Table
    with st.container(border=True):
        st.markdown(f"#### 📋 Enterprise Qualified Leaderboard ({len(ranked_df):,} Personnel)")
        display_cols = [id_col, name_col, "final_match_rate", "top_TGVs", "strengths", "gaps"]
        if rating_col and rating_col in ranked_df.columns:
            display_cols.insert(2, rating_col)

        st.dataframe(
            ranked_df[display_cols].head(20),
            use_container_width=True,
            column_config={
                "final_match_rate": st.column_config.ProgressColumn(
                    "Match Rate",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            }
        )

        csv_bytes = ranked_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Succession Leaderboard (CSV)", csv_bytes, "succession_leaderboard.csv", "text/csv")

    # Senior Data Analyst Diagnosis: Competency Gap Waterfall
    with st.container(border=True):
        st.markdown("#### 🌊 Candidate Diagnostic: Competency Gap Waterfall")
        st.caption("Visualizes exact point-by-point differentials between an individual talent and the cohort benchmark.")
        c_list = ranked_df[id_col].astype(str).tolist()
        if c_list and active_features:
            selected_cand = st.selectbox("Select Candidate to Diagnose:", c_list[:40])
            cand_row = ranked_df[ranked_df[id_col].astype(str) == selected_cand]
            if not cand_row.empty:
                idx = cand_row.index[0]
                c_name = str(cand_row[name_col].iloc[0])
                c_scores = [scaled_df.loc[idx, f] for f in active_features]
                b_scores = [scaled_df[f].mean() for f in active_features]

                w_col1, w_col2 = st.columns([3, 2])
                with w_col1:
                    fig_waterfall = plot_competency_gap_waterfall(c_name, active_features, c_scores, b_scores)
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                with w_col2:
                    fig_rad = plot_radar_comparison(c_name, active_features, c_scores, b_scores)
                    st.plotly_chart(fig_rad, use_container_width=True)


# ============================================================
# TAB 4: PSYCHOMETRIC ARCHETYPES (PCA + CLUSTERING)
# ============================================================
with tab_archetypes:
    st.markdown('<div class="section-header">🧬 Psychometric Archetypes & Workforce Landscape</div>', unsafe_allow_html=True)
    st.caption("Unsupervised Machine Learning (2D PCA projection + K-Means) discovering structural personality and behavioral clusters.")

    with st.spinner("Computing PCA 2D coordinates..."):
        pca_df, pca_model, kmeans_model = TalentMLClusterer.fit_talent_landscape(
            df,
            feature_cols=detected_metrics[:6],
            n_clusters=4
        )

    if "archetype" in pca_df.columns:
        counts = pca_df["archetype"].value_counts()
        arch_cols = st.columns(4)
        for i, (arch_name, count) in enumerate(counts.items()):
            with arch_cols[i % 4]:
                with st.container(border=True):
                    pct = (count / len(pca_df)) * 100
                    st.metric(arch_name, f"{count:,}", f"{pct:.1f}% share")

    with st.container(border=True):
        fig_pca = plot_pca_cluster_map(pca_df, name_col=name_col, rating_col=rating_col)
        st.plotly_chart(fig_pca, use_container_width=True)


# ============================================================
# TAB 5: HEAD-TO-HEAD CANDIDATE DUEL ARENA
# ============================================================
with tab_duel:
    st.markdown('<div class="section-header">⚔️ Head-to-Head Candidate Duel Arena</div>', unsafe_allow_html=True)
    st.caption("Directly pit two finalist candidates against each other. Compare multi-axis radar overlays and summon the AI Selection Arbitrator.")

    cand_keys = ranked_df[id_col].astype(str).tolist()
    if len(cand_keys) >= 2:
        d1, d2 = st.columns(2)
        with d1:
            cand1_id = st.selectbox("Select Candidate A (Blue):", cand_keys[:40], index=0)
        with d2:
            cand2_id = st.selectbox("Select Candidate B (Cyan):", cand_keys[:40], index=1)

        c1_row = ranked_df[ranked_df[id_col].astype(str) == cand1_id].iloc[0]
        c2_row = ranked_df[ranked_df[id_col].astype(str) == cand2_id].iloc[0]

        side1, side2 = st.columns(2)
        with side1:
            with st.container(border=True):
                st.markdown(f"### 🔵 {c1_row[name_col]}")
                st.markdown(f"**Match Rate:** `{c1_row['final_match_rate']}%` | **Rank:** `#{c1_row['rank']}`")
                st.markdown(f"**Top TGVs:** {c1_row.get('top_TGVs', 'N/A')}")
                st.markdown(f"**Strengths:** {c1_row.get('strengths', 'N/A')}")
                st.markdown(f"**Gaps:** {c1_row.get('gaps', 'N/A')}")

        with side2:
            with st.container(border=True):
                st.markdown(f"### 🔷 {c2_row[name_col]}")
                st.markdown(f"**Match Rate:** `{c2_row['final_match_rate']}%` | **Rank:** `#{c2_row['rank']}`")
                st.markdown(f"**Top TGVs:** {c2_row.get('top_TGVs', 'N/A')}")
                st.markdown(f"**Strengths:** {c2_row.get('strengths', 'N/A')}")
                st.markdown(f"**Gaps:** {c2_row.get('gaps', 'N/A')}")

        with st.container(border=True):
            c1_idx = ranked_df[ranked_df[id_col].astype(str) == cand1_id].index[0]
            c2_idx = ranked_df[ranked_df[id_col].astype(str) == cand2_id].index[0]
            c1_scores = [scaled_df.loc[c1_idx, f] for f in active_features]
            c2_scores = [scaled_df.loc[c2_idx, f] for f in active_features]
            bench_scores = [scaled_df[f].mean() for f in active_features]

            fig_duel = plot_head_to_head_radar(
                str(c1_row[name_col]), c1_scores,
                str(c2_row[name_col]), c2_scores,
                bench_scores, active_features
            )
            st.plotly_chart(fig_duel, use_container_width=True)

        st.markdown("#### ⚖️ AI Arbitrator Verdict")
        if st.button("🥊 Summon AI Arbitrator to Pick Winner", use_container_width=True):
            c1_summary = f"Match: {c1_row['final_match_rate']}%, Top: {c1_row.get('top_TGVs')}, Strengths: {c1_row.get('strengths')}, Gaps: {c1_row.get('gaps')}"
            c2_summary = f"Match: {c2_row['final_match_rate']}%, Top: {c2_row.get('top_TGVs')}, Strengths: {c2_row.get('strengths')}, Gaps: {c2_row.get('gaps')}"

            with st.spinner("AI Arbitrator is weighing psychometric tradeoffs..."):
                verdict = llm_service.generate_head_to_head_duel(
                    cand1_name=str(c1_row[name_col]),
                    cand1_metrics=c1_summary,
                    cand2_name=str(c2_row[name_col]),
                    cand2_metrics=c2_summary,
                    role_name="Senior / Lead Role"
                )
                with st.container(border=True):
                    st.markdown(verdict)


# ============================================================
# TAB 6: DATA ARCHITECTURE & API TELEMETRY
# ============================================================
with tab_dev:
    st.markdown('<div class="section-header">💻 Data Architecture & Microservice Telemetry</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)
    with t1:
        with st.container(border=True):
            mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("DataFrame Memory", f"{mem_mb:.2f} MB", f"{df.shape[1]} Dimensions")
    with t2:
        with st.container(border=True):
            st.metric("PostgreSQL Pooler", "Connected" if db_service.is_configured else "Offline", "SQLAlchemy Engine")
    with t3:
        with st.container(border=True):
            st.metric("Supabase Client", "Ready" if supabase_service.is_configured else "Offline", "REST API v1")

    st.markdown("#### 📡 Exportable REST API Payload (JSON)")
    top_candidates_api = ranked_df[[id_col, name_col, "final_match_rate", "top_TGVs"]].head(5).to_dict(orient="records")
    api_payload = {
        "status": "success",
        "timestamp": int(time.time()),
        "meta": {
            "total_evaluated": len(ranked_df),
            "algorithm": "RandomForest-MinMax-Composite-v3",
            "weights": calibrated_weights
        },
        "data": {
            "top_candidates": top_candidates_api
        }
    }
    st.code(json.dumps(api_payload, indent=2), language="json")
