# ============================================================
# 🎯 Talent Match Intelligence System (Job Role Benchmark)
# ============================================================

import sys
from pathlib import Path

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.config import BRAND_COLORS
from app.utils.llm_generator import LLMGenerator

# ============================================================
# CONFIGURATIONS
# ============================================================

st.set_page_config(page_title="Talent Match Intelligence — Role Benchmark", layout="wide")
st.title("🎯 Talent Match Intelligence — Role Benchmarking")
st.markdown("---")

llm_service = LLMGenerator()

# ============================================================
# 🧰 SIDEBAR INPUT
# ============================================================

with st.sidebar:
    st.header("Job Vacancy Configuration")
    role_name = st.text_input("Role Name", "Data Analyst")
    job_level = st.selectbox("Job Level", ["Junior", "Middle", "Senior", "Lead"])
    role_purpose = st.text_area(
        "Role Purpose",
        "Analyze business data and create actionable insights to support strategic decisions.",
    )

    st.subheader("📂 Upload Employee Data")
    uploaded_file = st.file_uploader("Upload Excel/CSV file", type=["csv", "xlsx"])
    max_candidates = st.slider("Max Candidates to Analyze", 3, 20, 10)

# ============================================================
# 📥 LOAD DATA
# ============================================================

df = pd.DataFrame()

# Default fallback if available in notebooks/data
default_path = BASE_DIR / "notebooks" / "combined_employee_data.csv"

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
    st.success(f"✅ Data berhasil dimuat! Jumlah baris: {len(df)}")
elif default_path.exists():
    df = pd.read_csv(default_path)
    st.info(f"ℹ️ Memuat default sample dataset ({len(df):,} baris).")

if not df.empty:
    st.dataframe(df.head(), use_container_width=True)

# ============================================================
# 📊 VISUALIZATION & ANALYSIS
# ============================================================

if not df.empty:
    if "rating" not in df.columns:
        st.error("Kolom 'rating' tidak ditemukan! Pastikan data memiliki kolom 'rating'.")
    else:
        top_candidates = df[df["rating"] == df["rating"].max()]
        st.markdown("### 👥 Kandidat dengan Rating Tertinggi")
        st.dataframe(top_candidates, use_container_width=True)

        # ============================================================
        # VISUALIZATION SECTION
        # ============================================================
        st.markdown("## 📊 Dashboard Visualization")

        # --- Match-rate Distribution ---
        fig_dist = px.histogram(
            df,
            x="rating",
            nbins=5,
            title="Match Rate Distribution",
            color_discrete_sequence=[BRAND_COLORS["primary"]],
        )
        fig_dist.update_layout(bargap=0.2)
        st.plotly_chart(fig_dist, use_container_width=True)

        # --- Top Strengths & Gaps ---
        st.markdown("### 💪 Top Strengths & 🚧 Gaps Across TGVs")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            mean_scores = df[numeric_cols].mean().sort_values(ascending=False)
            fig_strength = px.bar(
                mean_scores.head(10),
                title="Top Strengths",
                orientation="h",
                color=mean_scores.head(10),
                color_continuous_scale="Blues",
            )
            fig_gap = px.bar(
                mean_scores.tail(10),
                title="Top Gaps",
                orientation="h",
                color=mean_scores.tail(10),
                color_continuous_scale="Reds",
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(fig_strength, use_container_width=True)
            with col_b:
                st.plotly_chart(fig_gap, use_container_width=True)

        # --- Benchmark vs Candidate Radar ---
        st.markdown("### 🕸 Benchmark vs Candidate Comparison")
        if len(numeric_cols) >= 3:
            avg_scores = df[numeric_cols].mean().to_dict()
            candidate = top_candidates.iloc[0][numeric_cols].to_dict()
            categories = list(avg_scores.keys())[:8]  # Limit to 8 for readability
            benchmark_vals = [avg_scores[k] for k in categories]
            candidate_vals = [candidate[k] for k in categories]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=benchmark_vals, theta=categories, fill="toself", name="Benchmark"))
            fig_radar.add_trace(go.Scatterpolar(r=candidate_vals, theta=categories, fill="toself", name="Candidate"))
            fig_radar.update_layout(title="Benchmark vs Candidate Radar", polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- Heatmap Comparison ---
        st.markdown("### 🌡 Heatmap Benchmark vs Candidates")
        if len(numeric_cols) > 0:
            fig_heat = px.imshow(
                df[numeric_cols[:10]].corr(),
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # ============================================================
        # ACTION BUTTONS
        # ============================================================
        col1, col2 = st.columns(2)

        # --- Generate Job Profile ---
        with col1:
            if st.button("📝 Generate Job Profile"):
                with st.spinner("Generating job profile via AI..."):
                    job_profile = llm_service.generate_job_profile(
                        role_name=role_name,
                        job_level=job_level,
                        role_purpose=role_purpose,
                        benchmark_summary=top_candidates.describe().to_string()
                    )
                st.markdown(job_profile)

        # --- Generate Job Profile + AI Recommendation ---
        with col2:
            if st.button("🚀 Generate Candidate Recommendation"):
                top_candidates_subset = top_candidates.head(max_candidates)
                name_col = next(
                    (c for c in ["fullname", "name", "nama", "employee_name"] if c in top_candidates_subset.columns),
                    top_candidates_subset.columns[0],
                )
                kandidat_lines = [
                    f"- {row.get(name_col, 'UNKNOWN')} | Rating: {row.get('rating', 'N/A')}"
                    for _, row in top_candidates_subset.iterrows()
                ]
                kandidat_list_text = "\n".join(kandidat_lines)

                with st.spinner("Analyzing candidates via AI..."):
                    rec_result = llm_service.generate_candidate_recommendations(
                        role_name=role_name,
                        job_level=job_level,
                        candidates_summary=kandidat_list_text
                    )
                st.markdown(rec_result)
else:
    st.info("⬆ Silakan upload file CSV atau Excel berisi data kandidat terlebih dahulu.")

st.markdown("---")
st.caption("Tip: Gunakan `streamlit run app/app_dashboard.py` untuk pengalaman Enterprise Dashboard lengkap.")
