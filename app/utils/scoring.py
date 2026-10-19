"""
Scoring, Talent Analytics & Machine Learning Engine.
Senior Data Analyst suite providing:
- Data Sanitization & Outlier Clamping
- 9-Box Talent Review Grid (Performance vs Potential)
- Predictive Performance Driver Analysis (Random Forest Feature Importance)
- Succession Readiness Pipeline Modeling
- Min-Max Scaling & Weighted Composite Match-Rate Engine
- Unsupervised ML (PCA + K-Means) Archetype Discovery
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from app.config import TALENT_METRIC_COLUMNS, EXCLUDED_SCORING_COLUMNS


class TalentScoringEngine:
    """Enterprise-grade scoring and analytics engine for Talent Match Intelligence."""

    @staticmethod
    def clean_talent_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Defensive Data Sanitization:
        - Clamps performance ratings to standard 1.0 - 5.0 range (eliminating 100% scale errors).
        - Sanitizes non-numeric strings in psychometric columns.
        """
        clean_df = df.copy()

        # Sanitize rating column
        if "rating" in clean_df.columns:
            # Handle possible percentage scale error (e.g. 100 -> 5.0, 80 -> 4.0)
            rating_numeric = pd.to_numeric(clean_df["rating"], errors="coerce")
            is_percentage = (rating_numeric > 5.0) & (rating_numeric <= 100.0)
            if is_percentage.any():
                clean_df.loc[is_percentage, "rating"] = (clean_df.loc[is_percentage, "rating"] / 20.0).round(1)

            # Clamp between 1.0 and 5.0
            clean_df["rating"] = pd.to_numeric(clean_df["rating"], errors="coerce")
            clean_df["rating"] = clean_df["rating"].clip(lower=1.0, upper=5.0)

        return clean_df

    @staticmethod
    def identify_scoring_features(df: pd.DataFrame) -> List[str]:
        """
        Identifies valid assessment and numerical performance features.
        Explicitly excludes administrative IDs, operational keys, and categorical codes.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Prioritize known talent assessment metrics
        matched_metrics = [c for c in TALENT_METRIC_COLUMNS if c in numeric_cols]

        # If not enough domain metrics matched, fall back to other numeric columns excluding administrative IDs
        if len(matched_metrics) < 3:
            remaining = [
                c for c in numeric_cols
                if c not in EXCLUDED_SCORING_COLUMNS and c not in matched_metrics
            ]
            matched_metrics.extend(remaining[:6])

        return matched_metrics

    @staticmethod
    def normalize_metrics(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Normalizes metrics between 0 and 1 using MinMaxScaler.
        Handles missing values via column median imputation.
        """
        if not feature_cols:
            return pd.DataFrame(index=df.index)

        feature_df = df[feature_cols].copy()
        for col in feature_cols:
            if feature_df[col].isnull().all():
                feature_df[col] = 0.0
            else:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())

        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        scaled_array = scaler.fit_transform(feature_df)
        return pd.DataFrame(scaled_array, columns=feature_cols, index=df.index)

    @staticmethod
    def compute_weighted_scores(
        scaled_df: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """Computes composite talent scores using normalized weights."""
        if scaled_df.empty:
            return pd.Series(dtype=float)

        total_weight = sum(weights.values())
        if total_weight <= 0:
            normalized_weights = {k: 1.0 / len(weights) for k in weights}
        else:
            normalized_weights = {k: v / total_weight for k, v in weights.items()}

        composite_score = pd.Series(0.0, index=scaled_df.index)
        for col, weight in normalized_weights.items():
            if col in scaled_df.columns:
                composite_score += scaled_df[col] * weight

        return composite_score

    @classmethod
    def evaluate_candidates(
        cls,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        selected_features: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        End-to-end evaluation: sanitizes data, applies weights, determines top TGVs,
        identifies strengths/gaps against peer benchmark, and sorts descending.
        """
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), []

        result_df = cls.clean_talent_data(df)
        features = selected_features or cls.identify_scoring_features(result_df)
        if not features:
            return result_df, pd.DataFrame(), []

        if not weights:
            weights = {f: 1.0 / len(features) for f in features}

        scaled_df = cls.normalize_metrics(result_df, features)
        final_score_raw = cls.compute_weighted_scores(scaled_df, weights)
        result_df["_final_score_raw"] = final_score_raw
        result_df["final_match_rate"] = (final_score_raw * 100).round(2)

        # Top TGVs
        def get_top_tgvs(idx: int) -> str:
            row = scaled_df.loc[idx]
            top_k = row.sort_values(ascending=False).head(3)
            return "; ".join([f"{col.replace('_', ' ').title()} ({row[col] * 100:.0f}%)" for col in top_k.index])

        result_df["top_TGVs"] = [get_top_tgvs(i) for i in scaled_df.index]

        # Strengths & Gaps
        benchmark_mean = scaled_df.mean()

        def compute_strengths_gaps(idx: int) -> Tuple[str, str]:
            row = scaled_df.loc[idx]
            diff = row - benchmark_mean
            strengths = diff[diff > 0.05].sort_values(ascending=False).index.tolist()
            gaps = diff[diff < -0.05].sort_values().index.tolist()
            s_str = ", ".join([s.replace('_', ' ').title() for s in strengths[:4]]) or "Cohort Baseline"
            g_str = ", ".join([g.replace('_', ' ').title() for g in gaps[:4]]) or "Balanced"
            return s_str, g_str

        sg = [compute_strengths_gaps(i) for i in scaled_df.index]
        result_df["strengths"], result_df["gaps"] = zip(*sg)

        ranked_df = result_df.sort_values("final_match_rate", ascending=False).reset_index(drop=True)
        ranked_df["rank"] = ranked_df.index + 1

        return ranked_df, scaled_df, features

    @classmethod
    def compute_9box_matrix(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Senior People Analytics: Strategic 9-Box Talent Review Grid.
        - X-Axis: Performance (Annual Rating, 1-5 scale)
        - Y-Axis: Potential Index (Composite cognitive agility, endurance, and competencies)
        Assigns each employee to one of the 9 executive quadrants.
        """
        grid_df = cls.clean_talent_data(df)
        if "rating" not in grid_df.columns:
            return pd.DataFrame()

        # Potential components: IQ, TIKI, GTQ, Pauli, Avg Competency
        potential_cols = [c for c in ["iq", "tiki", "gtq", "pauli", "avg_competency_score", "faxtor"] if c in grid_df.columns]
        if not potential_cols:
            potential_cols = [c for c in grid_df.select_dtypes(include=[np.number]).columns if c != "rating"][:4]

        scaled_pot = cls.normalize_metrics(grid_df, potential_cols)
        grid_df["potential_score"] = (scaled_pot.mean(axis=1) * 100).round(1)

        # Performance Score: Fill NaN ratings with median
        grid_df["performance_score"] = grid_df["rating"].fillna(grid_df["rating"].median()).round(2)

        # Categorize into 9 Quadrants
        # Performance tiers: Low (< 2.8), Medium (2.8 - 3.8), High (> 3.8)
        # Potential tiers: Low (< 45), Medium (45 - 65), High (> 65)
        def assign_quadrant(row) -> str:
            perf = row["performance_score"]
            pot = row["potential_score"]

            if perf >= 3.8:
                if pot >= 65:
                    return "🌟 Star Talent"
                elif pot >= 45:
                    return "💎 High Performer"
                else:
                    return "🛡️ Solid Professional"
            elif perf >= 2.8:
                if pot >= 65:
                    return "🚀 High Potential"
                elif pot >= 45:
                    return "🎯 Core Performer"
                else:
                    return "⏳ Effective Operator"
            else:
                if pot >= 65:
                    return "💡 Rough Diamond"
                elif pot >= 45:
                    return "🔄 Inconsistent Performer"
                else:
                    return "⚠️ Talent Risk"

        grid_df["quadrant"] = grid_df.apply(assign_quadrant, axis=1)
        return grid_df

    @classmethod
    def compute_predictive_drivers(cls, df: pd.DataFrame, target_col: str = "rating") -> Tuple[pd.DataFrame, float]:
        """
        Senior People Analytics: Machine Learning Performance Driver Analysis.
        Fits a Random Forest Regressor to discover the statistical importance of
        each psychometric and assessment attribute in explaining employee performance.
        Returns:
            Tuple[DataFrame(Feature, ImportancePct), R2_Score]
        """
        clean_df = cls.clean_talent_data(df)
        if target_col not in clean_df.columns:
            return pd.DataFrame(), 0.0

        # Filter rows where target is not null
        valid_df = clean_df.dropna(subset=[target_col]).copy()
        features = [c for c in TALENT_METRIC_COLUMNS if c in valid_df.columns and c != target_col]

        if len(features) < 2 or len(valid_df) < 20:
            return pd.DataFrame(), 0.0

        X = valid_df[features].fillna(valid_df[features].median())
        y = valid_df[target_col]

        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        model.fit(X, y)
        r2 = max(0.0, float(model.score(X, y)))

        importance_df = pd.DataFrame({
            "Feature": [f.replace("_", " ").title() for f in features],
            "RawFeature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        importance_df["ImportancePct"] = (importance_df["Importance"] * 100).round(1)
        return importance_df, r2

    @classmethod
    def compute_talent_pyramid(cls, ranked_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates succession readiness distribution across enterprise tiers.
        """
        if "final_match_rate" not in ranked_df.columns:
            return pd.DataFrame()

        tiers = [
            ("🏆 Tier 1: Ready Now (Match ≥ 80%)", len(ranked_df[ranked_df["final_match_rate"] >= 80])),
            ("🥈 Tier 2: Successor in 1-2 Yrs (Match 65-79%)", len(ranked_df[(ranked_df["final_match_rate"] >= 65) & (ranked_df["final_match_rate"] < 80)])),
            ("🥉 Tier 3: Core Pipeline (Match 50-64%)", len(ranked_df[(ranked_df["final_match_rate"] >= 50) & (ranked_df["final_match_rate"] < 65)])),
            ("🛠️ Tier 4: In Development (Match < 50%)", len(ranked_df[ranked_df["final_match_rate"] < 50]))
        ]

        pyramid_df = pd.DataFrame(tiers, columns=["ReadinessTier", "Headcount"])
        pyramid_df["Percentage"] = ((pyramid_df["Headcount"] / len(ranked_df)) * 100).round(1)
        return pyramid_df

    @classmethod
    def compute_rank_sensitivity(
        cls,
        df: pd.DataFrame,
        base_weights: Dict[str, float],
        tweaked_weights: Dict[str, float]
    ) -> pd.DataFrame:
        """Simulates sensitivity drift when weights are modified."""
        base_ranked, _, _ = cls.evaluate_candidates(df, weights=base_weights)
        tweaked_ranked, _, _ = cls.evaluate_candidates(df, weights=tweaked_weights)

        id_col = next((c for c in ["employee_id", "id", "nip"] if c in df.columns), df.columns[0])

        base_map = dict(zip(base_ranked[id_col], base_ranked["rank"]))
        tweaked_ranked["previous_rank"] = tweaked_ranked[id_col].map(base_map)
        tweaked_ranked["rank_delta"] = tweaked_ranked["previous_rank"] - tweaked_ranked["rank"]

        return tweaked_ranked


class TalentMLClusterer:
    """Unsupervised Machine Learning for Talent Archetyping using PCA + K-Means."""

    ARCHETYPE_LABELS = {
        0: "🧠 Analytical Specialist",
        1: "⚡ Agile Executor",
        2: "🌟 Strategic High-Performer",
        3: "🛡️ Consistent Contributor"
    }

    @classmethod
    def fit_talent_landscape(
        cls,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_clusters: int = 4
    ) -> Tuple[pd.DataFrame, PCA, KMeans]:
        """Fits PCA + KMeans and returns enriched DataFrame with 2D coordinates."""
        clean_df = TalentScoringEngine.clean_talent_data(df)
        if clean_df.empty or len(feature_cols) < 2:
            return clean_df.copy(), None, None

        scaled_df = TalentScoringEngine.normalize_metrics(clean_df, feature_cols)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_df)

        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(scaled_df)

        output_df = clean_df.copy()
        output_df["pca_x"] = pca_coords[:, 0]
        output_df["pca_y"] = pca_coords[:, 1]
        output_df["cluster_id"] = clusters
        output_df["archetype"] = [cls.ARCHETYPE_LABELS.get(c, f"Cluster {c}") for c in clusters]

        return output_df, pca, kmeans
