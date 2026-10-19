# ============================================================
# 🎯 Talent Match Intelligence System (Ranked Talent Detail)
# ============================================================

import sys
from pathlib import Path

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.config import SUPABASE_URL, SUPABASE_KEY, BRAND_COLORS
from app.utils.scoring import TalentScoringEngine
from app.utils.llm_generator import LLMGenerator
from app.utils.supabase_connector import SupabaseConnector
from app.utils.visualizations import plot_radar_comparison

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="Ranked Talent Dashboard", layout="wide")
st.title("🎯 Ranked Talent Dashboard — Talent Match Intelligence")
st.markdown("Upload CSV atau ambil data dari Supabase. Hit `Run Analysis` untuk menghitung match rate, melihat ranked list, dan visualisasi.")

llm_service = LLMGenerator()
supabase_service = SupabaseConnector()

# ------------------------------
# SIDEBAR: data source & params
# ------------------------------
st.sidebar.header("Data Source & Analysis Settings")
data_source = st.sidebar.radio("Data source", ["Upload CSV", "Supabase", "Default Demo Dataset"])
uploaded = None

default_path = BASE_DIR / "notebooks" / "combined_employee_data.csv"

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload combined_employee_data.csv", type=["csv", "xlsx"])
elif data_source == "Supabase":
    st.sidebar.caption("Menggunakan kredensial Supabase dari environment variables (.env).")
else:
    st.sidebar.info("Menggunakan dataset demo bawaan proyek.")

st.sidebar.markdown("---")
st.sidebar.header("Success Formula Weights")
w_comp = st.sidebar.slider("Competency weight", 0.0, 1.0, 0.35, 0.05)
w_iq   = st.sidebar.slider("IQ weight", 0.0, 1.0, 0.25, 0.05)
w_exp  = st.sidebar.slider("Experience / Tenure weight", 0.0, 1.0, 0.15, 0.05)
w_lead = st.sidebar.slider("Leadership / Pauli weight", 0.0, 1.0, 0.15, 0.05)
w_other= st.sidebar.slider("Other numeric weight", 0.0, 1.0, 0.10, 0.05)

run_btn = st.sidebar.button("▶️ Run Analysis")

# ------------------------------
# Load data
# ------------------------------
df = pd.DataFrame()
load_error = None

if data_source == "Upload CSV" and uploaded is not None:
    try:
        df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    except Exception as e:
        load_error = f"Failed load: {e}"

elif data_source == "Supabase":
    if supabase_service.is_configured:
        try:
            resp = supabase_service.fetch_table("employee_analysis")
            if resp and resp.data:
                df = pd.DataFrame(resp.data)
            else:
                load_error = "No data returned from Supabase table."
        except Exception as e:
            load_error = f"Supabase connection failed: {e}"
    else:
        load_error = "SUPABASE_URL/SUPABASE_KEY not configured in .env"

elif data_source == "Default Demo Dataset":
    if default_path.exists():
        df = pd.read_csv(default_path)
    else:
        load_error = "Default dataset not found in notebooks/"

if load_error:
    st.sidebar.error(load_error)

if not df.empty:
    st.sidebar.success(f"Loaded {len(df):,} rows")

# ------------------------------
# Main analysis triggered by button or pre-loaded data
# ------------------------------
if run_btn or not df.empty:
    if df.empty:
        st.info("Pilih sumber data di sidebar untuk memulai.")
        st.stop()

    id_col = next((c for c in ["employee_id", "id", "emp_id", "nip"] if c in df.columns), df.columns[0])
    name_col = next((c for c in ["fullname", "name", "employee_name", "nama"] if c in df.columns), id_col)

    weights_dict = {
        "avg_competency_score": w_comp,
        "iq": w_iq,
        "years_of_service_months": w_exp,
        "pauli": w_lead,
        "faxtor": w_other
    }

    # Use robust ScoringEngine
    ranked, scaled, active_features = TalentScoringEngine.evaluate_candidates(
        df,
        weights=weights_dict
    )

    display_cols = [id_col, name_col, "final_match_rate", "top_TGVs", "strengths", "gaps"]
    display_cols = [c for c in display_cols if c in ranked.columns]
    ranked_display = ranked[display_cols].copy()

    st.markdown("## 🏅 Ranked Talent List")
    st.dataframe(ranked_display, use_container_width=True)

    # Download CSV
    csv = ranked_display.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Ranked CSV", csv, "ranked_talent_list.csv", "text/csv")

    # Visualizations
    st.markdown("## 📈 Visualizations")
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        fig_hist = px.histogram(
            ranked,
            x="final_match_rate",
            nbins=20,
            title="Final Match Rate Distribution",
            color_discrete_sequence=[BRAND_COLORS["primary"]]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_v2:
        top_n = st.slider("Top N candidates for comparison", 5, min(30, len(ranked)), 10)
        fig_bar = px.bar(
            ranked.head(top_n),
            x=name_col,
            y="final_match_rate",
            title=f"Top {top_n} Match Rates",
            color="final_match_rate",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Radar
    st.markdown("### 🕸 Benchmark vs Candidate (Radar)")
    if active_features:
        candidate_ids = ranked[id_col].astype(str).tolist()
        selected_id = st.selectbox("Select candidate to inspect:", candidate_ids[:40])
        cand_row = ranked[ranked[id_col].astype(str) == selected_id]

        if not cand_row.empty:
            idx = cand_row.index[0]
            cand_name = str(cand_row[name_col].iloc[0])
            c_scores = [scaled.loc[idx, f] for f in active_features]
            b_scores = [scaled[f].mean() for f in active_features]

            fig_radar = plot_radar_comparison(cand_name, active_features, c_scores, b_scores)
            st.plotly_chart(fig_radar, use_container_width=True)

    # AI Summary
    st.markdown("## 🧠 Summary Insights (AI-generated)")
    if st.button("🔎 Generate AI Insights for Top 5"):
        top5 = ranked.head(5)
        brief_lines = [
            f"{r.get(name_col, 'N/A')} (ID: {r[id_col]}) — Score: {r['final_match_rate']}%, Top: {r.get('top_TGVs', '')}, Strengths: {r.get('strengths', '')}"
            for _, r in top5.iterrows()
        ]
        top_summary = "\n".join(brief_lines)

        with st.spinner("🧠 Generating AI insights..."):
            insight = llm_service.generate_cohort_insights(top_summary)
            st.markdown(insight)

st.markdown("---")
st.caption("Tip: Gunakan `streamlit run app/app_dashboard.py` untuk pengalaman Enterprise Dashboard lengkap.")
