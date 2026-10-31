"""
Universal Signal Preprocessor (DSP Code Sandbox)
Deterministic feature extraction across Acoustic, IoT, Biomedical, RF, and Tabular inputs.
"""

from __future__ import annotations
import math
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from .models import ModalityType, SignalFeatureSummary


class UniversalSignalProcessor:
    """Deterministic feature extractor for all physical signals and structured inputs."""

    @staticmethod
    def process_time_series(
        signal_array: Union[np.ndarray, List[float]],
        sampling_rate_hz: float = 1000.0,
    ) -> Dict[str, Any]:
        """Computes fundamental time-domain and frequency-domain summaries."""
        arr = np.asarray(signal_array, dtype=np.float64)
        if len(arr) == 0:
            return {"error": "Empty signal array"}

        n_samples = len(arr)
        duration_sec = float(n_samples / sampling_rate_hz)

        # Time-domain metrics
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        max_val = float(np.max(arr))
        min_val = float(np.min(arr))
        peak_to_peak = float(max_val - min_val)
        rms_val = float(np.sqrt(np.mean(arr**2)))
        crest_factor = float(max(abs(max_val), abs(min_val)) / (rms_val + 1e-9))

        # Kurtosis and Skewness
        variance = std_val**2 + 1e-12
        skewness = float(np.mean((arr - mean_val) ** 3) / (variance**1.5))
        kurtosis = float(np.mean((arr - mean_val) ** 4) / (variance**2) - 3.0)

        # Frequency-domain (Fast Fourier Transform)
        fft_vals = np.abs(np.fft.rfft(arr))
        freqs = np.fft.rfftfreq(n_samples, 1.0 / sampling_rate_hz)
        peak_idx = int(np.argmax(fft_vals))
        peak_freq = float(freqs[peak_idx])

        # Top 3 spectral peak frequencies
        sorted_indices = np.argsort(fft_vals)[::-1]
        top_peaks = []
        for idx in sorted_indices:
            f = float(freqs[idx])
            mag = float(fft_vals[idx])
            if all(abs(f - p["frequency_hz"]) > (sampling_rate_hz / 50.0) for p in top_peaks):
                top_peaks.append({"frequency_hz": round(f, 2), "magnitude": round(mag, 2)})
            if len(top_peaks) >= 3:
                break

        # Detect statistical anomalies (> 3 standard deviations)
        z_scores = np.abs((arr - mean_val) / (std_val + 1e-8))
        anomaly_mask = z_scores > 3.0
        anomaly_count = int(np.sum(anomaly_mask))
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_timestamps = [
            round(float(idx / sampling_rate_hz), 4) for idx in anomaly_indices[:15]
        ]

        # Downsampled waveform preview for compact context
        step = max(1, n_samples // 128)
        preview = [round(float(x), 3) for x in arr[::step][:128]]

        return {
            "sample_count": n_samples,
            "sampling_rate_hz": sampling_rate_hz,
            "duration_seconds": round(duration_sec, 4),
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "range": [round(min_val, 4), round(max_val, 4)],
            "peak_to_peak": round(peak_to_peak, 4),
            "rms": round(rms_val, 4),
            "crest_factor": round(crest_factor, 3),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurtosis, 3),
            "dominant_frequency_hz": round(peak_freq, 2),
            "top_spectral_peaks": top_peaks,
            "anomalies_detected": anomaly_count,
            "anomaly_sample_timestamps_sec": anomaly_timestamps,
            "downsampled_preview": preview,
        }

    @classmethod
    def process_iot_vibration(
        cls,
        signal_array: Union[np.ndarray, List[float]],
        sampling_rate_hz: float = 2000.0,
    ) -> Dict[str, Any]:
        """Industrial machinery vibration diagnostics & ISO 10816 severity estimation."""
        base = cls.process_time_series(signal_array, sampling_rate_hz)
        arr = np.asarray(signal_array, dtype=np.float64)

        # Vibration severity classification (ISO 10816 standards proxy in mm/s or g RMS)
        rms = base.get("rms", 0.0)
        if rms < 1.12:
            severity = "Normal (Class A: Newly commissioned machine)"
        elif rms < 2.8:
            severity = "Acceptable (Class B: Unrestricted long-term operation)"
        elif rms < 7.1:
            severity = "Alert (Class C: Unsatisfactory for continuous operation)"
        else:
            severity = "Critical Alarm (Class D: Immediate damage risk, trip recommended)"

        # High-frequency transient spikes (indicating bearing or gear mesh defect)
        envelope = np.abs(sp_signal.hilbert(arr - np.mean(arr)))
        env_mean = float(np.mean(envelope))
        env_max = float(np.max(envelope))
        shock_ratio = round(float(env_max / (env_mean + 1e-8)), 2)

        base.update({
            "modality_subtype": "industrial_vibration",
            "iso_severity_class": severity,
            "shock_pulse_ratio": shock_ratio,
            "bearing_wear_flag": bool(shock_ratio > 4.5 or base.get("kurtosis", 0.0) > 3.5),
        })
        return base

    @classmethod
    def process_biomedical_ecg(
        cls,
        signal_array: Union[np.ndarray, List[float]],
        sampling_rate_hz: float = 250.0,
    ) -> Dict[str, Any]:
        """Biomedical ECG signal extraction: QRS peak detection, Heart Rate, and HRV metrics."""
        arr = np.asarray(signal_array, dtype=np.float64)
        n_samples = len(arr)
        if n_samples < 50:
            return {"error": "Signal too short for biomedical analysis"}

        # 1. Bandpass filter 0.5 - 40 Hz to suppress baseline wander & high frequency noise
        nyquist = 0.5 * sampling_rate_hz
        low = max(0.01, 0.5 / nyquist)
        high = min(0.99, 40.0 / nyquist)
        b, a = sp_signal.butter(2, [low, high], btype="bandpass")
        filtered_ecg = sp_signal.filtfilt(b, a, arr)

        # 2. R-peak detection via peak prominence and refractory distance (min 250 ms = 240 bpm)
        min_distance = int(sampling_rate_hz * 0.25)
        std_val = float(np.std(filtered_ecg))
        prominence_thresh = max(0.2, std_val * 1.2)
        peaks, properties = sp_signal.find_peaks(
            filtered_ecg,
            distance=min_distance,
            prominence=prominence_thresh,
        )

        r_peak_times = [round(float(p / sampling_rate_hz), 3) for p in peaks]
        rr_intervals_sec = np.diff(peaks) / sampling_rate_hz if len(peaks) > 1 else np.array([])

        if len(rr_intervals_sec) > 0:
            mean_rr = float(np.mean(rr_intervals_sec))
            heart_rate_bpm = float(60.0 / mean_rr) if mean_rr > 0 else 0.0

            # Heart Rate Variability (HRV) metrics
            sdnn_ms = float(np.std(rr_intervals_sec) * 1000.0)
            rmssd_ms = float(np.sqrt(np.mean(np.diff(rr_intervals_sec) ** 2)) * 1000.0) if len(rr_intervals_sec) > 1 else 0.0
            pnn50_ratio = float(np.mean(np.abs(np.diff(rr_intervals_sec) * 1000.0) > 50.0) * 100.0) if len(rr_intervals_sec) > 1 else 0.0

            # Rhythm diagnosis
            if heart_rate_bpm < 60:
                clinical_rhythm = "Sinus Bradycardia (< 60 BPM)"
            elif heart_rate_bpm > 100:
                clinical_rhythm = "Sinus Tachycardia (> 100 BPM)"
            elif sdnn_ms > 120 or (len(rr_intervals_sec) > 2 and np.max(rr_intervals_sec) / np.min(rr_intervals_sec) > 1.6):
                clinical_rhythm = "Irregular Rhythm / Ectopic Beat (PVC/PAC detected)"
            else:
                clinical_rhythm = "Normal Sinus Rhythm (NSR)"
        else:
            heart_rate_bpm = 0.0
            sdnn_ms = 0.0
            rmssd_ms = 0.0
            pnn50_ratio = 0.0
            clinical_rhythm = "Insufficient R-peaks detected"

        time_series_base = cls.process_time_series(filtered_ecg, sampling_rate_hz)
        time_series_base.update({
            "modality_subtype": "biomedical_ecg",
            "qrs_peaks_detected": len(peaks),
            "r_peak_timestamps_sec": r_peak_times[:12],
            "average_heart_rate_bpm": round(heart_rate_bpm, 1),
            "hrv_sdnn_ms": round(sdnn_ms, 2),
            "hrv_rmssd_ms": round(rmssd_ms, 2),
            "hrv_pnn50_percent": round(pnn50_ratio, 1),
            "rhythm_assessment": clinical_rhythm,
        })
        return time_series_base

    @classmethod
    def process_rf_iq(
        cls,
        i_signal: Union[np.ndarray, List[float]],
        q_signal: Optional[Union[np.ndarray, List[float]]] = None,
        sampling_rate_hz: float = 1e6,
    ) -> Dict[str, Any]:
        """Radio Frequency (RF) & Electromagnetic signal processing: I/Q envelope, PSD, SNR."""
        i_arr = np.asarray(i_signal, dtype=np.float64)
        if q_signal is None or len(q_signal) == 0:
            q_arr = np.zeros_like(i_arr)
        else:
            q_arr = np.asarray(q_signal, dtype=np.float64)

        complex_iq = i_arr + 1j * q_arr
        n_samples = len(complex_iq)

        # Signal power and magnitude envelope
        power_envelope = np.abs(complex_iq) ** 2
        mean_power = float(np.mean(power_envelope))
        peak_power = float(np.max(power_envelope))
        papr_db = float(10 * np.log10(max(1e-12, peak_power) / (mean_power + 1e-12)))

        # Power Spectral Density (PSD)
        fft_complex = np.fft.fftshift(np.fft.fft(complex_iq))
        psd = (np.abs(fft_complex) ** 2) / (n_samples * sampling_rate_hz + 1e-12)
        psd_db = 10 * np.log10(psd + 1e-15)
        freqs = np.fft.fftshift(np.fft.fftfreq(n_samples, 1.0 / sampling_rate_hz))

        peak_idx = int(np.argmax(psd))
        carrier_offset_hz = float(freqs[peak_idx])

        # Estimated SNR: peak PSD power vs median noise floor
        noise_floor_db = float(np.median(psd_db))
        peak_snr_db = float(np.max(psd_db) - noise_floor_db)

        # Waterfall representation summary (8 time slices x 16 frequency bins)
        n_slices = 8
        slice_size = max(16, n_samples // n_slices)
        waterfall_preview = []
        for s in range(n_slices):
            chunk = complex_iq[s * slice_size : (s + 1) * slice_size]
            if len(chunk) > 0:
                chunk_fft = np.abs(np.fft.fft(chunk))
                step = max(1, len(chunk_fft) // 8)
                waterfall_preview.append([round(float(v), 2) for v in chunk_fft[::step][:8]])

        return {
            "modality_subtype": "rf_electromagnetic",
            "sample_count": n_samples,
            "sampling_rate_hz": sampling_rate_hz,
            "duration_seconds": round(n_samples / sampling_rate_hz, 6),
            "carrier_frequency_offset_hz": round(carrier_offset_hz, 2),
            "estimated_snr_db": round(peak_snr_db, 2),
            "noise_floor_db": round(noise_floor_db, 2),
            "papr_db": round(papr_db, 2),
            "mean_power": round(mean_power, 4),
            "waterfall_matrix_summary": waterfall_preview,
            "downsampled_envelope": [
                round(float(x), 3) for x in power_envelope[:: max(1, n_samples // 64)][:64]
            ],
        }

    @classmethod
    def process_acoustic_audio(
        cls,
        signal_array: Union[np.ndarray, List[float]],
        sampling_rate_hz: float = 16000.0,
    ) -> Dict[str, Any]:
        """Acoustic waveform processing: RMS decibels, spectral centroid, zero crossings."""
        arr = np.asarray(signal_array, dtype=np.float64)
        base = cls.process_time_series(arr, sampling_rate_hz)

        # Zero-crossing rate
        zero_crossings = int(np.sum(np.diff(np.sign(arr) != 0)))
        zcr_rate = float(zero_crossings / (len(arr) + 1e-8))

        # Spectral Centroid
        fft_vals = np.abs(np.fft.rfft(arr))
        freqs = np.fft.rfftfreq(len(arr), 1.0 / sampling_rate_hz)
        sum_fft = np.sum(fft_vals) + 1e-12
        spectral_centroid = float(np.sum(freqs * fft_vals) / sum_fft)

        # Decibel Full Scale (dBFS)
        rms = base.get("rms", 1e-6)
        dbfs = float(20 * np.log10(max(1e-6, rms)))

        base.update({
            "modality_subtype": "acoustic_audio",
            "zero_crossing_rate": round(zcr_rate, 4),
            "spectral_centroid_hz": round(spectral_centroid, 2),
            "peak_dbfs": round(dbfs, 2),
        })
        return base

    @classmethod
    def process_tabular_data(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Tabular and log profiler: summary statistics, outlier rates, correlations, shifts."""
        if df.empty:
            return {"error": "Empty dataframe provided"}

        row_count, col_count = df.shape
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Missing values
        missing_map = df.isnull().sum().to_dict()
        total_missing = sum(missing_map.values())
        missing_percentage = round((total_missing / (row_count * col_count + 1e-9)) * 100, 2)

        # Numerical statistics and Outliers (IQR method)
        stats_map: Dict[str, Dict[str, Any]] = {}
        outlier_counts: Dict[str, int] = {}
        for col in numerical_cols:
            series = df[col].dropna()
            if len(series) > 0:
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = int(((series < lower_bound) | (series > upper_bound)).sum())
                outlier_counts[col] = outliers

                stats_map[col] = {
                    "mean": round(float(series.mean()), 3),
                    "std": round(float(series.std()), 3),
                    "min": round(float(series.min()), 3),
                    "max": round(float(series.max()), 3),
                    "median": round(float(series.median()), 3),
                    "outliers_iqr": outliers,
                }

        # Significant correlations
        correlations: List[Dict[str, Any]] = []
        if len(numerical_cols) >= 2:
            corr_matrix = df[numerical_cols].corr()
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    c1, c2 = numerical_cols[i], numerical_cols[j]
                    val = corr_matrix.loc[c1, c2]
                    if not np.isnan(val) and abs(val) >= 0.6:
                        correlations.append({
                            "col_a": c1,
                            "col_b": c2,
                            "pearson_r": round(float(val), 3),
                            "relationship": "Strong Positive" if val > 0 else "Strong Inverse",
                        })

        sample_preview = df.head(5).to_dict(orient="records")

        return {
            "modality_subtype": "tabular_logs",
            "rows": row_count,
            "columns": col_count,
            "column_names": list(df.columns),
            "numerical_columns": numerical_cols,
            "categorical_columns": categorical_cols,
            "missing_percentage": missing_percentage,
            "column_statistics": stats_map,
            "outlier_counts": outlier_counts,
            "key_correlations": correlations[:5],
            "preview_rows": sample_preview,
        }

    @classmethod
    def process_document_text(cls, text: str) -> Dict[str, Any]:
        """Document text profiler: structural hierarchy, word counts, and section parsing."""
        lines = text.strip().splitlines()
        headings = [line.strip() for line in lines if line.startswith("#") or (line.isupper() and len(line) < 60 and len(line) > 3)]
        words = text.split()
        char_count = len(text)
        word_count = len(words)
        estimated_tokens = int(word_count * 1.33)

        return {
            "modality_subtype": "documents_scans",
            "character_count": char_count,
            "word_count": word_count,
            "estimated_tokens": estimated_tokens,
            "detected_headings": headings[:10],
            "line_count": len(lines),
            "text_sample": text[:800],
        }

    @classmethod
    def extract_features(
        cls,
        input_data: Any,
        modality: ModalityType = ModalityType.IOT_SENSOR,
        sampling_rate_hz: float = 1000.0,
        **kwargs,
    ) -> SignalFeatureSummary:
        """Universal dispatcher returning a standardized SignalFeatureSummary."""
        if modality == ModalityType.BIOMEDICAL:
            feats = cls.process_biomedical_ecg(input_data, sampling_rate_hz=sampling_rate_hz)
            preview = feats.get("downsampled_preview", [])
        elif modality == ModalityType.IOT_SENSOR:
            feats = cls.process_iot_vibration(input_data, sampling_rate_hz=sampling_rate_hz)
            preview = feats.get("downsampled_preview", [])
        elif modality == ModalityType.RF_ELECTROMAGNETIC:
            q_sig = kwargs.get("q_signal", None)
            feats = cls.process_rf_iq(input_data, q_sig, sampling_rate_hz=sampling_rate_hz)
            preview = feats.get("downsampled_envelope", [])
        elif modality == ModalityType.ACOUSTIC:
            feats = cls.process_acoustic_audio(input_data, sampling_rate_hz=sampling_rate_hz)
            preview = feats.get("downsampled_preview", [])
        elif modality == ModalityType.TABULAR_LOGS:
            if isinstance(input_data, pd.DataFrame):
                df = input_data
            else:
                df = pd.DataFrame(input_data)
            feats = cls.process_tabular_data(df)
            preview = []
        elif modality in (ModalityType.DOCUMENTS_SCANS, ModalityType.IMAGES):
            feats = cls.process_document_text(str(input_data))
            preview = []
        else:
            feats = cls.process_time_series(input_data, sampling_rate_hz=sampling_rate_hz)
            preview = feats.get("downsampled_preview", [])

        return SignalFeatureSummary(
            modality=modality,
            sampling_rate_hz=sampling_rate_hz,
            duration_seconds=feats.get("duration_seconds", 0.0),
            sample_count=feats.get("sample_count", feats.get("rows", 0)),
            time_domain={
                k: v for k, v in feats.items() if isinstance(v, (int, float)) and k in [
                    "mean", "std", "rms", "crest_factor", "skewness", "kurtosis",
                    "average_heart_rate_bpm", "hrv_sdnn_ms", "hrv_rmssd_ms",
                    "carrier_frequency_offset_hz", "estimated_snr_db", "peak_dbfs"
                ]
            },
            frequency_domain={
                "dominant_frequency_hz": feats.get("dominant_frequency_hz"),
                "top_spectral_peaks": feats.get("top_spectral_peaks", []),
            },
            anomalies={
                "count": feats.get("anomalies_detected", feats.get("outlier_counts", 0)),
                "timestamps": feats.get("anomaly_sample_timestamps_sec", []),
                "flag": feats.get("bearing_wear_flag", False),
            },
            preview_samples=preview,
            raw_feature_map=feats,
        )

