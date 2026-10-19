# 🎯 Sistem Intelijen Talenta (Talent Match Intelligence System)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?logo=plotly&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL%20%2F%20Supabase-4169E1?logo=postgresql&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**Platform People Analytics & Sistem Pendukung Keputusan Berbasis Data untuk Perencanaan Suksesi, Evaluasi 9-Box Grid, dan Pemodelan Prediktif Machine Learning.**

[Fitur Utama](#-fitur-analitika-enterprise) • [Metodologi](#-metodologi-analisis) • [Kamus Data](#-kamus-data-asesmen--psikometri) • [Arsitektur Proyek](#-arsitektur-proyek) • [Panduan Instalasi](#-panduan-instalasi--eksekusi)

</div>

---

## 🌟 Ringkasan Eksekutif

**Sistem Intelijen Talenta** (*Talent Match Intelligence System*) adalah platform analitika sumber daya manusia tingkat lanjut (*Enterprise People Analytics*) yang dirancang untuk memecahkan tantangan krusial dalam manajemen modal manusia strategis: **mengidentifikasi, memetakan, dan merencanakan suksesi talenta berkinerja tinggi secara objektif dan berbasis data (*evidence-based*).**

Platform ini mengintegrasikan pemodelan statistik, algoritma *Machine Learning* (Supervised & Unsupervised), serta kecerdasan buatan generatif (*Generative AI*) untuk mentransformasi ribuan baris data asesmen psikometri dan penilaian kinerja tahunan menjadi keputusan promosi dan rekrutmen yang presisi.

```mermaid
flowchart TD
    A["Data Mentah HR<br>(Excel / CSV / Supabase)"] --> B["Sanitasi Data & Penanganan Outlier"]
    B --> C["Domain Metric Filtering<br>(Pemisahan ID vs Skor Asesmen)"]
    C --> D["Normalisasi Fitur Min-Max"]
    
    D --> E["Engine 9-Box Talent Review<br>(Performance vs Potential)"]
    D --> F["Model Prediktif Random Forest<br>(Analisis Bobot Kinerja)"]
    D --> G["Klasterisasi 2D PCA + K-Means<br>(Pola Arketipe Perilaku)"]
    D --> H["Simulasi Bobot Suksesi Dinamis<br>(What-If Sensitivity)"]
    
    E --> I["Dashboard Interaktif Streamlit & Plotly"]
    F --> I
    G --> I
    H --> I
    
    I --> J["AI Selection Arbitrator & Profiler<br>(Spesifikasi Jabatan & Rekomendasi)"]
```

---

## 🚀 Fitur Analitika Enterprise

| Modul Analitika | Metodologi & Nilai Bisnis |
|---|---|
| **🎯 Matriks 9-Box Talent Review** | Mengadaptasi standar global McKinsey & General Electric untuk memetakan karyawan ke dalam 9 kuadran (*Star Talent, High Potential, Core Backbone, hingga Talent Risk*) berdasarkan evaluasi Kinerja (*Performance*) dan Kapasitas Tumbuh (*Potential*). |
| **🔬 Machine Learning Predictive Drivers** | Melatih model **Random Forest Regressor** untuk mengungkap faktor psikometri dan kompetensi apa yang secara statistik paling menentukan rating performa tinggi karyawan, dilengkapi visualisasi *Lollipop Chart*. |
| **⏳ Piramida Kesiapan Suksesi** | Visualisasi corong (*Funnel Pyramid*) yang mengkategorikan kesiapan talenta menjadi 4 tingkat suksesi: *Tier 1: Ready Now ($\ge 80\%$), Tier 2: 1-2 Years, Tier 3: Core Pipeline, dan Tier 4: In Development*. |
| **🌊 Diagnostik Gap Kompetensi Waterfall** | Grafik *Waterfall* interaktif yang memperlihatkan selisih persentase positif/negatif ($\pm \Delta\%$) kompetensi kandidat terhadap rata-rata *benchmark* perusahaan. |
| **🧬 Arketipe Talenta 2D PCA + K-Means** | Mereduksi multi-dimensi skor psikometri menggunakan *Principal Component Analysis* (PCA) dan mengelompokkannya secara otomatis ke dalam 4 kuadran arketipe bakat. |
| **⚔️ Arena Duel Kandidat Head-to-Head** | Perbandingan langsung dua kandidat finalis dengan grafik radar multi-layer serta bantuan **AI Selection Arbitrator** untuk memberikan pertimbangan perekrutan yang objektif. |
| **💻 Developer Console & API Sandbox** | Fitur inspeksi teknis: profiling pemakaian memori RAM, *schema viewer*, dan generator *payload* REST API (format JSON) siap integrasi ke backend microservice. |

---

## 📊 Kamus Data Asesmen & Psikometri

Dataset studi kasus mencakup metrik psikometri, kognitif, dan indikator performa:

| Kolom Metrik | Deskripsi Analitikal | Skala / Rentang |
|---|---|---|
| `rating` | Nilai evaluasi kinerja tahunan karyawan | $1.0 - 5.0$ |
| `avg_competency_score` | Rata-rata evaluasi kompetensi perilaku & fungsional 360° | $1.0 - 5.0$ |
| `iq` | Tingkat inteligensi umum (*Cognitive Capacity*) | Skala IQ ($80 - 140+$) |
| `gtq` | *General Talent Quotient* (Daya tangkap & logika umum) | Skor Poin |
| `tiki` | Tes Inteligensi Kolektif Indonesia (Pemahaman analogi & numerik) | Stanine Level $1 - 10$ |
| `pauli` | Ketahanan kerja, kecepatan stabil, dan konsistensi mental | Skor Ketahanan |
| `faxtor` | Kecepatan pemecahan masalah dan ketelitian kognitif | Skor Poin |
| `years_of_service_months` | Masa kerja karyawan di organisasi | Total Bulan |
| `disc` / `disc_word` | Profil gaya kerja (Dominance, Influence, Steadiness, Conscientiousness) | Kategorikal |
| `mbti` | Tipe kepribadian Myers-Briggs | 16 Tipe (misal: `INTJ`, `ESFP`) |

---

## 🧮 Metodologi Analisis

### 1. Normalisasi Fitur (Min-Max Scaling)
Untuk membandingkan indikator dengan skala berbeda (misalnya IQ skala 80-140 vs Kompetensi skala 1-5), data dinormalisasi ke interval $[0, 1]$:

$$\tilde{x}_{ij} = \frac{x_{ij} - \min(X_j)}{\max(X_j) - \min(X_j)}$$

### 2. Formula Kesesuaian Tertimbang (*Weighted Composite Match Score*)
Skor akhir kesesuaian talenta ($S_i$) dihitung dengan normalisasi bobot dinamis:

$$S_i = \sum_{j=1}^{M} w_j \cdot \tilde{x}_{ij} \times 100\% \quad \text{dengan} \quad \sum_{j=1}^{M} w_j = 1$$

### 3. Pemodelan Faktor Penentu (Random Forest Feature Importance)
Kepentingan relatif fitur dihitung menggunakan algoritma *Random Forest* untuk mengukur penurunan impuritas (*Mean Decrease in Impurity*) terhadap variabel target `rating`.

---

## 🏗️ Arsitektur Proyek

```text
sistem-intelijen-talenta/
├── app/
│   ├── app_dashboard.py       # 🚀 Master Enterprise Streamlit Dashboard
│   ├── main.py                # Standard entrypoint runner
│   ├── job_role.py            # Modul benchmark profil posisi
│   ├── job_detail.py          # Modul peringkat talenta detail
│   ├── config.py              # Konfigurasi terpusat & manajemen secrets
│   └── utils/
│       ├── scoring.py         # 🧮 Engine matematika 9-Box, ML Random Forest & PCA
│       ├── visualizations.py  # 📊 Visualisasi Plotly korporat (Lollipop, Radar, Donut, Waterfall)
│       ├── llm_generator.py   # 🤖 Layanan integrasi AI OpenRouter
│       ├── db_connector.py    # 🗄️ Konektor PostgreSQL SQLAlchemy dengan connection pooling
│       └── supabase_connector.py # 🔌 Client wrapper Supabase REST API
├── notebooks/
│   ├── combined_employee_data.csv # Dataset bersih (10.000+ data karyawan)
│   ├── dataAnalis.ipynb       # Notebook Exploratory Data Analysis (EDA)
│   └── Study Case DA.xlsx     # Spreadsheet sumber studi kasus
├── sql/
│   ├── create_tables.sql      # DDL skema tabel database
│   ├── success_formula.sql    # Query agregasi faktor sukses
│   └── test_queries.sql       # Query verifikasi data
├── .env.example               # Template environment variables aman
├── .gitignore                 # Proteksi secrets, cache, dan virtual environments
├── requirements.txt           # Daftar dependensi ter-pinning
└── README.md                  # Dokumentasi teknis proyek
```

---

## ⚡ Panduan Instalasi & Eksekusi

### 1. Kloning Repository & Setup Lingkungan

```powershell
# Kloning repository
git clone https://github.com/fallss/sistem-intelijen-talenta.git

# Masuk ke direktori
cd sistem-intelijen-talenta

# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Instal seluruh dependensi
pip install -r requirements.txt
```

### 2. Konfigurasi Environment Variables (Opsional)

Salin template konfigurasi `.env.example` ke `.env`:

```powershell
cp .env.example .env
```

Isi kunci API dan kredensial database (opsional; jika tidak diisi, sistem otomatis berjalan dalam *Local Offline Mode* dengan dataset demo):

```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
DATABASE_URL=postgresql://user:pass@host:6543/postgres
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

### 3. Menjalankan Dashboard

```powershell
python -m streamlit run app/app_dashboard.py
```

Dashboard akan otomatis terbuka pada browser Anda di tautan `http://localhost:8501`.

---

## 🛡️ Standar Kualitas & Keamanan Data

- ✅ **Proteksi Kredensial:** Tidak ada kunci API atau kata sandi yang di-hardcode dalam repositori. Seluruh konfigurasi dilindungi `.gitignore`.
- ✅ **Defensive Data Handling:** Sanitasi otomatis terhadap data anomali dan penanganan *missing values* menggunakan imputasi median.
- ✅ **Arsitektur Modular (Clean Code):** Pemisahan tegas antara antarmuka pengguna (*presentation layer*), logika analitik (*business logic*), dan akses data (*data layer*).

---

<div align="center">
Dikembangkan untuk Portofolio Profesional <b>Senior People Analytics & Data Engineering</b>.
</div>
