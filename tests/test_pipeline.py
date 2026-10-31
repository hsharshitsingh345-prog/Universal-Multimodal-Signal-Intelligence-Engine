"""
Automated Unit and Pipeline Tests for Universal Signal AI Engine.
"""

import unittest
import numpy as np
import pandas as pd

from universal_engine.models import ModalityType, EngineOutput
from universal_engine.dsp import UniversalSignalProcessor
from universal_engine.memory import UniversalMemoryStore
from universal_engine.generators import SignalDataGenerator
from universal_engine.llm import DeterministicLocalEngine
from universal_engine.engine import UniversalIntelligenceEngine


class TestUniversalSignalEngine(unittest.TestCase):

    def test_iot_vibration_processing(self):
        t, signal = SignalDataGenerator.generate_iot_vibration(
            duration_sec=1.0,
            sampling_rate_hz=1000.0,
            shaft_freq_hz=30.0,
            bearing_fault_freq_hz=14.2,
        )
        res = UniversalSignalProcessor.process_iot_vibration(signal, sampling_rate_hz=1000.0)
        self.assertIn("dominant_frequency_hz", res)
        self.assertIn("rms", res)
        self.assertIn("iso_severity_class", res)
        self.assertGreater(res["rms"], 0.0)
        self.assertGreaterEqual(res["anomalies_detected"], 0)

    def test_biomedical_ecg_processing(self):
        t, ecg = SignalDataGenerator.generate_biomedical_ecg(
            duration_sec=4.0,
            sampling_rate_hz=250.0,
            heart_rate_bpm=75.0,
        )
        res = UniversalSignalProcessor.process_biomedical_ecg(ecg, sampling_rate_hz=250.0)
        self.assertIn("average_heart_rate_bpm", res)
        self.assertIn("hrv_sdnn_ms", res)
        self.assertIn("qrs_peaks_detected", res)
        self.assertGreater(res["qrs_peaks_detected"], 1)
        self.assertGreater(res["average_heart_rate_bpm"], 40.0)

    def test_rf_iq_processing(self):
        t, i_sig, q_sig = SignalDataGenerator.generate_rf_iq(
            duration_sec=0.001,
            sampling_rate_hz=1e6,
            carrier_offset_hz=25000.0,
            snr_db=20.0,
        )
        res = UniversalSignalProcessor.process_rf_iq(i_sig, q_sig, sampling_rate_hz=1e6)
        self.assertIn("carrier_frequency_offset_hz", res)
        self.assertIn("estimated_snr_db", res)
        self.assertIn("papr_db", res)
        self.assertGreater(res["estimated_snr_db"], 10.0)

    def test_tabular_profiling(self):
        df = SignalDataGenerator.generate_tabular_telemetry(num_records=100)
        res = UniversalSignalProcessor.process_tabular_data(df)
        self.assertEqual(res["rows"], 100)
        self.assertIn("column_statistics", res)
        self.assertIn("motor_temp_c", res["column_statistics"])

    def test_duckdb_memory_and_hybrid_search(self):
        mem = UniversalMemoryStore(db_path=":memory:")

        # Test metric storage
        mem.store_signal_run(
            session_id="sess_01",
            signal_id="sig_test_vib",
            modality="iot_vibration",
            feature_summary={"rms": 3.42, "dominant_frequency_hz": 14.2, "anomalies_detected": 4},
            sampling_rate_hz=1000.0,
        )

        df = mem.query_metrics_sql("SELECT signal_id, rms, dominant_frequency_hz FROM signal_metrics")
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["rms"], 3.42, places=2)

        # Test hybrid search
        mem.add_document_chunk("doc1", "Centrifugal booster pump operating at 1800 RPM with BPFO bearing fault")
        mem.add_document_chunk("doc2", "Patient electrocardiogram showing sinus tachycardia above 110 BPM")

        results = mem.hybrid_retrieve("pump bearing fault", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "doc1")

    def test_end_to_end_intelligence_engine(self):
        engine = UniversalIntelligenceEngine(model_client=DeterministicLocalEngine())
        t, signal = SignalDataGenerator.generate_iot_vibration()
        doc_text = SignalDataGenerator.sample_technical_document()

        output = engine.process_and_analyze(
            raw_input=signal,
            modality=ModalityType.IOT_SENSOR,
            sampling_rate_hz=2000.0,
            document_text=doc_text,
        )

        self.assertIsInstance(output, EngineOutput)
        self.assertGreaterEqual(len(output.executive_takeaways), 3)
        self.assertGreaterEqual(len(output.qa_pairs), 4)
        self.assertGreaterEqual(len(output.charts), 1)
        self.assertTrue(output.metadata.get("verified_arithmetic"))

    def test_tabular_end_to_end(self):
        engine = UniversalIntelligenceEngine(model_client=DeterministicLocalEngine())
        df = SignalDataGenerator.generate_tabular_telemetry(num_records=100)
        output = engine.process_and_analyze(
            raw_input=df,
            modality=ModalityType.TABULAR_LOGS,
        )
        self.assertIsInstance(output, EngineOutput)
        self.assertGreaterEqual(len(output.executive_takeaways), 3)
        self.assertGreaterEqual(len(output.qa_pairs), 4)


if __name__ == "__main__":
    unittest.main()
