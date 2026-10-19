"""
LLM Integration Service for HR Talent Intelligence.
Supports OpenRouter API (OpenAI GPT-4o-mini, LLaMA, Anthropic Claude) with resilient error handling,
graceful fallbacks, and JSON response structuring.
"""

from typing import Dict, Any, Optional
import requests
import json
import logging
from app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL

logger = logging.getLogger(__name__)


class LLMGenerator:
    """Enterprise wrapper for AI-powered Talent Analysis."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.model = model or OPENROUTER_MODEL or "openai/gpt-4o-mini"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def is_configured(self) -> bool:
        """Returns True if a non-empty API key is present."""
        return bool(self.api_key and len(self.api_key.strip()) > 10)

    def _call_api(self, prompt: str, system_prompt: str, temperature: float = 0.7, max_tokens: int = 1500) -> str:
        """Internal helper to dispatch requests to OpenRouter."""
        if not self.is_configured:
            return (
                "⚠️ **OpenRouter API Key belum dikonfigurasi.**\n\n"
                "Silakan set `OPENROUTER_API_KEY` pada file `.env` atau melalui Streamlit Secrets untuk mengaktifkan AI Insights."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/talent-match-system",
            "X-Title": "Talent Match Intelligence"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=40)
            if response.status_code == 401:
                return "❌ **Autentikasi Gagal (401)**: API Key OpenRouter tidak valid atau expired. Mohon periksa kembali `.env` Anda."
            elif response.status_code == 429:
                return "⚠️ **Rate Limit (429)**: Kuota API tercapai atau terlalu banyak request. Coba beberapa saat lagi."

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            return "⏱️ **Request Timeout**: Layanan AI membutuhkan waktu terlalu lama untuk merespons. Silakan coba kembali."
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter request error: {e}")
            return f"❌ **Error Koneksi AI**: Terjadi kendala saat menghubungi provider AI ({e})."
        except Exception as e:
            logger.error(f"Unexpected error calling LLM: {e}")
            return f"❌ **Error**: {e}"

    def generate_job_profile(
        self,
        role_name: str,
        job_level: str,
        role_purpose: str,
        benchmark_summary: str = ""
    ) -> str:
        """Generates detailed job profile, core requirements, and KPIs."""
        system_prompt = (
            "You are a Senior Strategic HR & Talent Acquisition Director. "
            "Generate structured, highly actionable, professional Job Specifications."
        )
        prompt = f"""
Generate an executive-level Job Specification:

- **Target Role:** {role_name}
- **Seniority Level:** {job_level}
- **Primary Mission/Purpose:** {role_purpose}
- **Benchmark High-Performer Profile:**
{benchmark_summary or 'Standard high-performing corporate benchmark.'}

Please structure your response into clear Markdown sections:
1. 📌 **Executive Role Summary**
2. 🎯 **Key Accountability & Objectives (OKRs/KPIs)**
3. 🛠️ **Hard Skills & Technical Competencies**
4. 🧠 **Behavioral & Leadership Competencies**
5. 🎓 **Ideal Background & Experience Criteria**
"""
        return self._call_api(prompt, system_prompt)

    def generate_candidate_recommendations(
        self,
        role_name: str,
        job_level: str,
        candidates_summary: str
    ) -> str:
        """Synthesizes candidate scores and produces strategic talent match recommendations."""
        system_prompt = (
            "You are an Industrial & Organizational (I/O) Psychologist and Chief People Officer. "
            "Provide objective, evidence-based talent selection recommendations."
        )
        prompt = f"""
Role: {role_name} ({job_level})

Top Candidates Evaluated:
{candidates_summary}

Based on the quantitative talent match results:
1. 🏆 **Top Recommendation:** Identify the best candidate and defend the choice with quantitative evidence.
2. 💡 **Strengths & Core Value Proposition:** Highlight key differentiators.
3. 🚧 **Growth Areas & Risks:** What are potential developmental gaps to mitigate?
4. 🚀 **Onboarding & Coaching Plan:** Concrete 30-60-90 day recommendations for this hire.
"""
        return self._call_api(prompt, system_prompt)

    def generate_cohort_insights(self, top_candidates_summary: str) -> str:
        """Produces strategic HR insights across the high-performing cohort."""
        system_prompt = "You are an Enterprise People Analytics Specialist."
        prompt = f"""
Analyze the characteristics of our top high-performing talents:
{top_candidates_summary}

Provide a 4-pillar analysis:
1. **Performance Patterns:** What common traits do top performers share?
2. **Predictive Success Attributes:** Which assessments (IQ, Competency, Pauli, etc.) correlated highest with high ratings?
3. **Talent Pipeline Health:** Observed strengths and systemic talent gaps.
4. **Strategic People Recommendations:** Advice for HR leadership and talent development programs.
"""
        return self._call_api(prompt, system_prompt)

    def generate_head_to_head_duel(
        self,
        cand1_name: str,
        cand1_metrics: str,
        cand2_name: str,
        cand2_metrics: str,
        role_name: str = "Senior Analyst"
    ) -> str:
        """Compares two candidates directly in a head-to-head evaluation."""
        system_prompt = "You are an executive talent assessment board arbitrator."
        prompt = f"""
Arbitrate a direct Head-to-Head hiring decision for the role of '{role_name}':

### Candidate A: {cand1_name}
{cand1_metrics}

### Candidate B: {cand2_name}
{cand2_metrics}

Please deliver a decisive comparative assessment:
1. 🥊 **Direct Advantage Comparison:** Where does Candidate A dominate, and where does Candidate B outperform?
2. 🎯 **Contextual Fit:** Which candidate is better suited for (a) fast turnaround/high pressure vs (b) complex strategic architecture?
3. 🏁 **Final Hiring Verdict:** If there is only 1 headcount, who should be selected and why?
"""
        return self._call_api(prompt, system_prompt)