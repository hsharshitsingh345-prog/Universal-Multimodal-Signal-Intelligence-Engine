"""
Universal Dual-Store Memory Engine
- Short-Term Conversational & Anomaly Working Buffer
- Deterministic Time-Series Metric Store (DuckDB)
- Semantic Long-Term Memory with Dense Vector + Sparse BM25 Hybrid Retrieval
"""

from __future__ import annotations
import math
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import duckdb
import numpy as np
import pandas as pd


class BM25Index:
    """Lightweight sparse BM25 keyword indexer for zero-dependency hybrid search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_term_freqs: List[Counter] = []
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.df: Counter = Counter()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())

    def add_document(self, doc_id: str, text: str):
        tokens = self.tokenize(text)
        tf = Counter(tokens)
        self.doc_ids.append(doc_id)
        self.doc_texts.append(text)
        self.doc_term_freqs.append(tf)
        self.doc_lens.append(len(tokens))

        for term in set(tokens):
            self.df[term] += 1

        self.corpus_size = len(self.doc_ids)
        self.avg_doc_len = sum(self.doc_lens) / max(1, self.corpus_size)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        query_tokens = self.tokenize(query)
        scores = []
        for i, tf in enumerate(self.doc_term_freqs):
            score = 0.0
            doc_len = self.doc_lens[i]
            for term in query_tokens:
                if term in tf:
                    doc_freq = self.df.get(term, 0)
                    idf = math.log((self.corpus_size - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
                    term_f = tf[term]
                    denom = term_f + self.k1 * (1 - self.b + self.b * (doc_len / max(1e-9, self.avg_doc_len)))
                    score += idf * (term_f * (self.k1 + 1)) / max(1e-9, denom)
            scores.append((self.doc_ids[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class UniversalMemoryStore:
    """Combines DuckDB for SQL metric analytics, working session buffers, and hybrid semantic retrieval."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.con = duckdb.connect(database=db_path)
        self._init_duckdb_schema()

        # Ephemeral Working Memory Buffer
        self.active_session_dialogue: List[Dict[str, Any]] = []
        self.reviewed_anomalies: List[Dict[str, Any]] = []

        # Semantic Hybrid Store
        self.bm25_index = BM25Index()
        self.semantic_chunks: Dict[str, Dict[str, Any]] = {}
        self.dense_vectors: Dict[str, np.ndarray] = {}

    def _init_duckdb_schema(self):
        """Initializes tables for time-series streams and summarized signal metrics."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS signal_metrics (
                session_id VARCHAR,
                signal_id VARCHAR,
                modality VARCHAR,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sampling_rate_hz DOUBLE,
                duration_seconds DOUBLE,
                sample_count BIGINT,
                rms DOUBLE,
                mean_val DOUBLE,
                std_val DOUBLE,
                dominant_frequency_hz DOUBLE,
                anomalies_detected BIGINT,
                raw_summary_json VARCHAR
            );
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS signal_stream_samples (
                signal_id VARCHAR,
                sample_index BIGINT,
                time_sec DOUBLE,
                amplitude DOUBLE
            );
        """)

    def store_signal_run(
        self,
        session_id: str,
        signal_id: str,
        modality: str,
        feature_summary: Dict[str, Any],
        raw_samples: Optional[np.ndarray] = None,
        sampling_rate_hz: float = 1000.0,
    ):
        """Stores deterministic DSP features and optionally downsampled time-series into DuckDB."""
        t_dom = feature_summary.get("time_domain", {})
        f_dom = feature_summary.get("frequency_domain", {})
        anom = feature_summary.get("anomalies", {})

        rms = float(t_dom.get("rms", feature_summary.get("rms", 0.0)))
        mean_val = float(t_dom.get("mean", feature_summary.get("mean", 0.0)))
        std_val = float(t_dom.get("std", feature_summary.get("std", 0.0)))
        dom_freq = float(f_dom.get("dominant_frequency_hz", feature_summary.get("dominant_frequency_hz", 0.0) or 0.0))
        anom_count = int(anom.get("count", feature_summary.get("anomalies_detected", 0)))
        duration = float(feature_summary.get("duration_seconds", 0.0))
        sample_count = int(feature_summary.get("sample_count", 0))

        summary_json = json.dumps(feature_summary, default=str)

        self.con.execute("""
            INSERT INTO signal_metrics VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            signal_id,
            modality,
            sampling_rate_hz,
            duration,
            sample_count,
            rms,
            mean_val,
            std_val,
            dom_freq,
            anom_count,
            summary_json,
        ))

        # Store stream samples if provided
        if raw_samples is not None and len(raw_samples) > 0:
            step = max(1, len(raw_samples) // 200)
            rows = []
            for i, idx in enumerate(range(0, len(raw_samples), step)):
                t = float(idx / sampling_rate_hz)
                amp = float(raw_samples[idx])
                rows.append((signal_id, int(idx), t, amp))
            
            # Fast DuckDB appender / bulk insert
            temp_df = pd.DataFrame(rows, columns=["signal_id", "sample_index", "time_sec", "amplitude"])
            self.con.register("temp_samples_df", temp_df)
            self.con.execute("INSERT INTO signal_stream_samples SELECT * FROM temp_samples_df")
            self.con.unregister("temp_samples_df")

    def query_metrics_sql(self, sql_query: str) -> pd.DataFrame:
        """Executes arbitrary deterministic SQL queries over ingested metrics and time-series."""
        return self.con.execute(sql_query).df()

    def add_document_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        """Indexes text chunks for hybrid semantic vector + BM25 keyword retrieval."""
        self.semantic_chunks[chunk_id] = {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata or {},
        }
        self.bm25_index.add_document(chunk_id, text)

        # Dense embedding: if provided use it, otherwise compute a normalized hashed pseudo-embedding
        if embedding is not None:
            norm = np.linalg.norm(embedding)
            self.dense_vectors[chunk_id] = embedding / max(1e-9, norm)
        else:
            # Deterministic character n-gram projection for offline environments
            vec = np.zeros(64, dtype=np.float32)
            for token in self.bm25_index.tokenize(text):
                h = hash(token) % 64
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            self.dense_vectors[chunk_id] = vec / max(1e-9, norm)

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 3,
        alpha: float = 0.5,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining Dense Vector Cosine Similarity and Sparse BM25 Keyword Search.
        Score = alpha * dense_score + (1 - alpha) * bm25_normalized_score.
        """
        if not self.semantic_chunks:
            return []

        # 1. Sparse BM25 retrieval
        bm25_results = dict(self.bm25_index.search(query, top_k=len(self.semantic_chunks)))
        max_bm25 = max(bm25_results.values()) if bm25_results and max(bm25_results.values()) > 0 else 1.0

        # 2. Dense vector similarity
        if query_embedding is None:
            q_vec = np.zeros(64, dtype=np.float32)
            for token in self.bm25_index.tokenize(query):
                h = hash(token) % 64
                q_vec[h] += 1.0
            q_norm = np.linalg.norm(q_vec)
            query_embedding = q_vec / max(1e-9, q_norm)
        else:
            q_norm = np.linalg.norm(query_embedding)
            query_embedding = query_embedding / max(1e-9, q_norm)

        hybrid_ranked = []
        for chunk_id, chunk_data in self.semantic_chunks.items():
            bm25_raw = bm25_results.get(chunk_id, 0.0)
            bm25_norm = bm25_raw / max_bm25

            d_vec = self.dense_vectors.get(chunk_id)
            if d_vec is not None and len(d_vec) == len(query_embedding):
                dense_sim = float(np.dot(query_embedding, d_vec))
            else:
                dense_sim = 0.0

            combined_score = alpha * dense_sim + (1.0 - alpha) * bm25_norm
            hybrid_ranked.append({
                "chunk_id": chunk_id,
                "text": chunk_data["text"],
                "metadata": chunk_data["metadata"],
                "score": round(combined_score, 4),
                "dense_score": round(dense_sim, 4),
                "bm25_score": round(bm25_norm, 4),
            })

        hybrid_ranked.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_ranked[:top_k]

    # --- Working Buffer Management ---
    def append_dialogue(self, role: str, content: str):
        self.active_session_dialogue.append({
            "timestamp": time.time(),
            "role": role,
            "content": content,
        })

    def log_reviewed_anomaly(self, signal_id: str, anomaly_type: str, notes: str):
        self.reviewed_anomalies.append({
            "timestamp": time.time(),
            "signal_id": signal_id,
            "anomaly_type": anomaly_type,
            "notes": notes,
        })

    def get_working_context_summary(self) -> Dict[str, Any]:
        return {
            "dialogue_turns": len(self.active_session_dialogue),
            "recent_messages": self.active_session_dialogue[-4:],
            "reviewed_anomalies_count": len(self.reviewed_anomalies),
            "indexed_documents_count": len(self.semantic_chunks),
        }
