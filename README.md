# Universal Multimodal & Signal Intelligence Engine ⚡

An enterprise-grade, end-to-end architecture and Python implementation for ingesting any physical signal or document, storing context across long horizons, reasoning through anomalies using deterministic DSP verification, and synthesizing findings into clean executive takeaways, dynamic visual graphs, and grounded Q&A pairs.

---

## 🏗️ 5-Stage System Architecture

```
[ Raw Multimodal Inputs ]
  ├── Documents & Scans (PDF, DOCX, TXT)
  ├── Visuals & Diagrams (PNG, JPEG, TIFF)
  ├── Time-Series & IoT (Sensors, Logs, Telemetry)
  ├── Audio & Acoustics (Voice, Machinery Noise)
  ├── Biomedical Signals (ECG, EEG, PPG)
  └── Tabular & Structured (CSV, Parquet, SQL)
            │
            ▼
[ Stage 1: Ingestion & Feature Normalization (DSP Sandbox) ]
  ├── Document Layout & OCR Engine (Docling / Unstructured)
  ├── Signal DSP Engine (FFT, Wavelet, Bandpass, Z-score, HRV, PSD)
  └── Tabular Profiler (Pandas / DuckDB)
            │
            ▼
[ Stage 2: Dual-Store Memory Engine ]
  ├── Ephemeral Working Memory (Session context & recent window)
  ├── Semantic Vector Store (Dense embeddings + BM25 sparse search)
  └── Metric & Time-Series Store (DuckDB / SQLite-vec)
            │
            ▼
[ Stage 3: The Reasoning ("Thinking") Engine ]
  ├── Deterministic DSP Sandbox (Guarantees zero arithmetic hallucination)
  ├── Chain-of-Thought (CoT) Multimodal Reasoning Model
  └── Strict Schema Enforcement (Pydantic validation)
            │
            ▼
[ Stage 4: Output Synthesis & Simplification ]
  ├── Executive Plain-Language Takeaways (3-6 scannable bullets with bold lead-ins)
  ├── Automated Visual & Chart Engine (Plotly declarative charts)
  └── Self-Assessment Q&A Generator (4-8 core grounded concepts & answers)
```

---

## 📊 Universal Signal & Ingestion Matrix

| Signal / Modality Type | Common Formats | Dedicated Extraction Tools | Extracted Features & Representations |
| :--- | :--- | :--- | :--- |
| **Documents & Scans** | PDF, DOCX, TXT, scanned receipts | Docling, Unstructured, PyMuPDF | Hierarchical headings, tables, layout bounding boxes, and OCR text. |
| **IoT & Industrial Sensors** | Accelerometer, vibration, temperature | NumPy, SciPy, Pandas | Rolling statistics, FFT harmonics, anomaly z-scores, ISO 10816 severity. |
| **Biomedical Signals** | ECG, EEG, PPG, EMG (EDF, CSV) | SciPy (signal), BioSPPy | Bandpass-filtered baselines, QRS peak detection, HRV (SDNN/RMSSD), rhythm diagnosis. |
| **RF & Electromagnetic** | I/Q raw data, radar telemetry | NumPy, SciPy | Power Spectral Density (PSD), waterfall matrix, SNR, PAPR, carrier frequency offset. |
| **Acoustic & Audio** | WAV, MP3, FLAC, ultrasound | SciPy, Librosa, Torchaudio | RMS decibels (dBFS), Mel-spectrograms, zero-crossing rate, spectral centroid. |
| **Tabular & Event Logs** | CSV, JSON, Parquet, Syslog | DuckDB, Pandas | Summary metrics, distribution shifts, error rate spikes, and outlier timestamps. |

---

## 🚀 Quickstart & Setup

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests

```bash
python -m unittest tests/test_pipeline.py
```

### 3. Launch the Interactive Web Dashboard

```bash
streamlit run app.py
```

---

## 🛡️ Deployment Across Environments

### Option A: Air-Gapped / Edge / Embedded Tier (100% Data Sovereignty)
- **Engine**: Built-in `DeterministicLocalEngine` or `OllamaClient` (DeepSeek-R1 / Qwen2.5-VL).
- **Storage**: In-memory / local DuckDB instance with zero network egress.
- **Benefits**: Compliant with defense, healthcare, and industrial air-gapped isolation regulations.

### Option B: Cloud / Managed Enterprise Tier
- **Engine**: `GeminiClient` utilizing `gemini-2.5-flash` or `gemini-2.5-pro` via the `google-genai` SDK.
- **Structured Outputs**: Native Pydantic schema enforcement (`response_schema=EngineOutput`).
- **Benefits**: Handles multi-million-token contexts with frontier-grade reasoning.

---

## 📂 Repository Structure

```
├── universal_signal_ai_engine.md   # Complete technical architecture specification
├── app.py                          # Streamlit interactive web dashboard
├── requirements.txt                # Production dependencies
├── README.md                       # Documentation & quickstart guide
├── universal_engine/               # Core Python engine package
│   ├── __init__.py                 # Exports public APIs
│   ├── models.py                   # Pydantic schemas (EngineOutput, ChartSpec, QAPair)
│   ├── dsp.py                      # UniversalSignalProcessor (DSP code sandbox)
│   ├── memory.py                   # UniversalMemoryStore (DuckDB + Hybrid BM25 Vector)
│   ├── llm.py                      # Multi-tier LLM clients (Gemini, Ollama, Deterministic)
│   ├── engine.py                   # UniversalIntelligenceEngine orchestrator
│   └── generators.py               # Multi-modality synthetic signal generators
└── tests/
    └── test_pipeline.py            # Unit & integration test suite
```

