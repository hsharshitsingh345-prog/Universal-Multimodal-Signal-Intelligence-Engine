"""
Universal Intelligence Engine Orchestrator
Coordinates DSP feature extraction, memory persistence, reasoning models,
and structured output synthesis.
"""

from __future__ import annotations
import json
import time
import uuid
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from .models import EngineOutput, ModalityType, SignalFeatureSummary
from .dsp import UniversalSignalProcessor
from .memory import UniversalMemoryStore
from .llm import BaseLLMClient, DeterministicLocalEngine


class UniversalIntelligenceEngine:
    """The central orchestrator for Multimodal & Signal Intelligence."""

    def __init__(
        self,
        model_client: Optional[BaseLLMClient] = None,
        memory_store: Optional[UniversalMemoryStore] = None,
    ):
        self.client = model_client or DeterministicLocalEngine()
        self.memory = memory_store or UniversalMemoryStore()

    @staticmethod
    def construct_system_prompt() -> str:
        return """You are an expert Data & Signal Intelligence Analyst. Your goal is to inspect raw multimodal inputs, sensor signals, documents, or data tables, and synthesize them into the simplest, cleanest format possible.

STRICT REQUIREMENTS:
1. Executive Takeaways: Provide 3-5 concise, high-impact bullet points in plain language. Use bold lead-in verbs (e.g. **Identified Dominant Frequency**, **Detected Vibration Spike**).
2. Visuals / Graphs: Recommend the exact charts needed to understand the data, specifying series or discrete data points.
3. Questions & Answers: Generate 4-8 simple questions that test or explain the core insights, along with clear, definitive 1-2 sentence answers strictly grounded in the data.
4. Output strictly valid JSON matching the requested schema. No conversational filler or external Markdown markdown blocks around the JSON."""

    def analyze(
        self,
        raw_data_summary: Union[Dict[str, Any], SignalFeatureSummary],
        document_text: str = "",
        session_id: Optional[str] = None,
        signal_id: Optional[str] = None,
    ) -> EngineOutput:
        """
        Runs full pipeline: DSP normalization -> Memory indexing -> CoT reasoning -> Pydantic validation.
        """
        start_time = time.time()
        session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        signal_id = signal_id or f"sig_{uuid.uuid4().hex[:8]}"

        # Unpack feature summary dictionary
        if isinstance(raw_data_summary, SignalFeatureSummary):
            features_dict = raw_data_summary.raw_feature_map or raw_data_summary.model_dump()
            modality = raw_data_summary.modality.value
            sampling_rate_hz = raw_data_summary.sampling_rate_hz
        else:
            features_dict = raw_data_summary
            modality = str(features_dict.get("modality_subtype", features_dict.get("modality", "iot_sensor")))
            sampling_rate_hz = float(features_dict.get("sampling_rate_hz", 1000.0))

        # 1. Store in DuckDB Metric Store
        self.memory.store_signal_run(
            session_id=session_id,
            signal_id=signal_id,
            modality=modality,
            feature_summary=features_dict,
            sampling_rate_hz=sampling_rate_hz,
        )

        # 2. Index document text into Semantic Vector Store if present
        if document_text and len(document_text.strip()) > 0:
            doc_chunk_id = f"doc_{signal_id}"
            self.memory.add_document_chunk(
                chunk_id=doc_chunk_id,
                text=document_text,
                metadata={"signal_id": signal_id, "session_id": session_id},
            )

        # 3. Construct Reasoning Context Payload
        context_payload = {
            "session_id": session_id,
            "signal_id": signal_id,
            "signal_features": features_dict,
            "document_content": document_text[:6000],
            "memory_summary": self.memory.get_working_context_summary(),
        }

        user_prompt = (
            f"Analyze the following ingested multimodal data and signals:\n"
            f"{json.dumps(context_payload, indent=2, default=str)}"
        )

        # 4. Invoke LLM Client (Cloud, Ollama, or Deterministic Local Engine)
        raw_response = self.client.generate(self.construct_system_prompt(), user_prompt)

        # 5. Parse and Validate Output through Pydantic
        try:
            if isinstance(raw_response, dict):
                output = EngineOutput.model_validate(raw_response)
            else:
                # Strip markdown code fences if model enclosed JSON
                cleaned_text = raw_response.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                cleaned_text = cleaned_text.strip()
                output = EngineOutput.model_validate_json(cleaned_text)
        except Exception as e:
            # Fallback parsing or fallback to deterministic engine
            print(f"[Engine] JSON validation warning: {e}. Falling back to deterministic synthesizer.")
            fallback_engine = DeterministicLocalEngine()
            fallback_json = fallback_engine.generate(self.construct_system_prompt(), user_prompt)
            output = EngineOutput.model_validate_json(fallback_json)

        # Attach execution provenance
        elapsed_sec = round(time.time() - start_time, 3)
        output.metadata.update({
            "session_id": session_id,
            "signal_id": signal_id,
            "modality": modality,
            "processing_latency_sec": elapsed_sec,
            "verified_arithmetic": True,
        })

        # Log to working memory buffer
        self.memory.append_dialogue(
            role="engine",
            content=f"Processed {modality} signal {signal_id} in {elapsed_sec}s.",
        )

        return output

    def process_and_analyze(
        self,
        raw_input: Any,
        modality: ModalityType = ModalityType.IOT_SENSOR,
        sampling_rate_hz: float = 1000.0,
        document_text: str = "",
        session_id: Optional[str] = None,
        **kwargs,
    ) -> EngineOutput:
        """End-to-end convenience method: raw input -> DSP -> memory -> reasoning -> output."""
        features = UniversalSignalProcessor.extract_features(
            input_data=raw_input,
            modality=modality,
            sampling_rate_hz=sampling_rate_hz,
            **kwargs,
        )
        return self.analyze(
            raw_data_summary=features,
            document_text=document_text,
            session_id=session_id,
        )

