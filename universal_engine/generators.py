"""
Synthetic Multi-Modality Signal & Data Generators
Allows immediate out-of-the-box testing across Industrial IoT, Biomedical ECG, RF I/Q,
Acoustics, Tabular Telemetry, and Technical Documents.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class SignalDataGenerator:
    """Generates synthetic signals with known ground-truth anomalies."""

    @staticmethod
    def generate_iot_vibration(
        duration_sec: float = 2.0,
        sampling_rate_hz: float = 2000.0,
        shaft_freq_hz: float = 30.0,
        bearing_fault_freq_hz: float = 14.2,
        add_anomalies: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates industrial accelerometer signal:
        - Shaft rotation tone (30 Hz)
        - Bearing wear defect tone (14.2 Hz)
        - Gaussian baseline noise
        - High-energy transient shock pulses (anomalies)
        """
        n_samples = int(duration_sec * sampling_rate_hz)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Baseline rotational and defect harmonics
        shaft_wave = 1.2 * np.sin(2 * np.pi * shaft_freq_hz * t)
        bearing_wave = 0.8 * np.sin(2 * np.pi * bearing_fault_freq_hz * t)
        harmonic_2x = 0.3 * np.sin(2 * np.pi * (2 * shaft_freq_hz) * t)
        noise = 0.25 * np.random.normal(0, 1, n_samples)

        signal = shaft_wave + bearing_wave + harmonic_2x + noise

        # Inject transient shock pulse anomalies
        if add_anomalies:
            shock_indices = [
                int(n_samples * 0.25),
                int(n_samples * 0.52),
                int(n_samples * 0.78),
            ]
            for idx in shock_indices:
                width = int(sampling_rate_hz * 0.015)
                start = max(0, idx - width // 2)
                end = min(n_samples, idx + width // 2)
                decay = np.exp(-np.linspace(0, 4, end - start))
                signal[start:end] += 6.5 * decay * np.sin(2 * np.pi * 320 * t[start:end])

        return t, signal

    @staticmethod
    def generate_biomedical_ecg(
        duration_sec: float = 6.0,
        sampling_rate_hz: float = 250.0,
        heart_rate_bpm: float = 75.0,
        inject_arrhythmia: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesizes a realistic Lead-II ECG waveform:
        - P-wave, QRS-complex, T-wave
        - Baseline wander (respiratory modulation)
        - Premature Ventricular Contraction (PVC) ectopic beats
        """
        n_samples = int(duration_sec * sampling_rate_hz)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        ecg = np.zeros(n_samples)

        rr_interval = 60.0 / heart_rate_bpm
        beat_times = []
        curr = 0.2
        beat_idx = 0
        while curr < duration_sec - 0.2:
            # Inject premature ectopic beat at beat #4
            if inject_arrhythmia and beat_idx == 3:
                curr += rr_interval * 0.65  # premature beat
                beat_times.append((curr, True))
                curr += rr_interval * 1.35  # compensatory pause
            else:
                curr += rr_interval
                beat_times.append((curr, False))
            beat_idx += 1

        for b_time, is_pvc in beat_times:
            # Gaussian components for P, Q, R, S, T waves
            dt = t - b_time
            if not is_pvc:
                # Normal P-QRS-T
                p_wave = 0.18 * np.exp(-((dt + 0.16) ** 2) / (2 * 0.02**2))
                q_wave = -0.15 * np.exp(-((dt + 0.04) ** 2) / (2 * 0.01**2))
                r_wave = 1.45 * np.exp(-(dt**2) / (2 * 0.015**2))
                s_wave = -0.35 * np.exp(-((dt - 0.04) ** 2) / (2 * 0.01**2))
                t_wave = 0.32 * np.exp(-((dt - 0.22) ** 2) / (2 * 0.04**2))
                ecg += p_wave + q_wave + r_wave + s_wave + t_wave
            else:
                # Wide, distorted ectopic PVC complex
                r_pvc = 1.9 * np.exp(-(dt**2) / (2 * 0.045**2))
                s_pvc = -0.8 * np.exp(-((dt - 0.08) ** 2) / (2 * 0.03**2))
                t_pvc = -0.45 * np.exp(-((dt - 0.25) ** 2) / (2 * 0.06**2))
                ecg += r_pvc + s_pvc + t_pvc

        # Baseline wander + subtle 50 Hz power hum
        baseline_wander = 0.15 * np.sin(2 * np.pi * 0.33 * t)
        noise = 0.04 * np.random.normal(0, 1, n_samples)
        ecg += baseline_wander + noise

        return t, ecg

    @staticmethod
    def generate_rf_iq(
        duration_sec: float = 0.002,
        sampling_rate_hz: float = 1e6,
        carrier_offset_hz: float = 45000.0,
        snr_db: float = 22.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates complex baseband I/Q radar pulse / QPSK signal:
        - Carrier frequency offset
        - Additive White Gaussian Noise calibrated to target SNR
        """
        n_samples = int(duration_sec * sampling_rate_hz)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)

        # Pulsed chirp / modulated signal
        phase = 2 * np.pi * carrier_offset_hz * t + 0.5 * np.pi * (t / duration_sec) ** 2
        i_clean = np.cos(phase)
        q_clean = np.sin(phase)

        # Noise power calibration
        sig_power = 1.0
        noise_power = sig_power / (10 ** (snr_db / 10.0))
        noise_std = np.sqrt(noise_power / 2.0)

        i_noisy = i_clean + np.random.normal(0, noise_std, n_samples)
        q_noisy = q_clean + np.random.normal(0, noise_std, n_samples)

        return t, i_noisy, q_noisy

    @staticmethod
    def generate_tabular_telemetry(num_records: int = 250) -> pd.DataFrame:
        """Generates structured IoT pump telemetry dataframe with pressure and temperature spikes."""
        np.random.seed(42)
        timestamps = pd.date_range("2026-10-31 08:00:00", periods=num_records, freq="10s")
        motor_temp = 68.0 + np.random.normal(0, 2.5, num_records)
        vibration_rms = 1.4 + np.random.exponential(0.3, num_records)
        line_pressure_psi = 120.0 + 5.0 * np.sin(np.linspace(0, 10, num_records)) + np.random.normal(0, 1.2, num_records)
        flow_rate_lpm = 45.0 - 0.15 * (vibration_rms - 1.4) + np.random.normal(0, 0.8, num_records)
        error_code = ["ERR_00" if v < 2.8 else "ERR_WARN_VIB" for v in vibration_rms]

        # Inject sudden overheating & pressure drop anomaly at 70%..76% of records
        start_anom = int(num_records * 0.70)
        end_anom = min(num_records, int(num_records * 0.76))
        motor_temp[start_anom:end_anom] += 26.5
        vibration_rms[start_anom:end_anom] += 3.8
        line_pressure_psi[start_anom:end_anom] -= 32.0
        for i in range(start_anom, end_anom):
            error_code[i] = "ERR_CRIT_OVERHEAT"

        df = pd.DataFrame({
            "timestamp": timestamps,
            "motor_temp_c": np.round(motor_temp, 2),
            "vibration_rms_g": np.round(vibration_rms, 3),
            "line_pressure_psi": np.round(line_pressure_psi, 2),
            "flow_rate_lpm": np.round(flow_rate_lpm, 2),
            "status_code": error_code,
        })
        return df

    @staticmethod
    def sample_technical_document() -> str:
        """Returns sample technical maintenance manual / operating procedure."""
        return """# PUMP STATION 4 - CENTRIFUGAL BOOSTER UNIT #3
## Standard Operating Specifications & Thresholds
- **Model**: Titan-Hydro V9 High-Pressure Booster
- **Rated Operating Speed**: 1,800 RPM (30.0 Hz primary shaft rotation)
- **Bearing Type**: SKF Explorer 6312 Deep Groove Ball Bearing
- **Ball Pass Frequency Outer Race (BPFO)**: 14.2 Hz defect signature
- **Vibration Alarm Limit (ISO 10816)**: Class C Alert threshold = 2.80 mm/s RMS; Class D Critical Trip = 7.10 mm/s RMS
- **Thermal Threshold**: Maximum continuous bearing casing temperature 85.0°C
- **Hydraulic Operating Baseline**: 120 ± 8 PSI delivery pressure at 45.0 LPM nominal flow.

### Operational Incident Protocols
In the event of anomalous vibration exceeding 3.0 standard deviations or peak harmonics around 14.2 Hz:
1. Immediately throttle downstream regulator valve to relieve hydraulic back-pressure.
2. Conduct acoustic stethoscope inspection on the drive-end bearing bracket.
3. Check grease port for metallic particulate debris or thermal discoloration.
"""
