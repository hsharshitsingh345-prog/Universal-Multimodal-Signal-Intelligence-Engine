from __future__ import annotations
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, model_validator


class ModalityType(str, Enum):
    ACOUSTIC = "acoustic"
    IOT_SENSOR = "iot_sensor"
    BIOMEDICAL = "biomedical"
    RF_ELECTROMAGNETIC = "rf_electromagnetic"
    TABULAR_LOGS = "tabular_logs"
    DOCUMENTS_SCANS = "documents_scans"
    IMAGES = "images"


class ChartPoint(BaseModel):
    label: str = Field(..., description="X-axis label, category name, or timestamp")
    value: float = Field(..., description="Y-axis numerical value")


class ChartSpec(BaseModel):
    title: str = Field(..., description="Clear title for the graph")
    chart_type: str = Field(
        ...,
        description="Type of chart: 'line', 'bar', 'scatter', 'spectrogram', 'histogram', or 'pie'",
    )
    x_label: str = Field(default="Time / Frequency / Category", description="X-axis label")
    y_label: str = Field(default="Amplitude / Power / Count", description="Y-axis label")
    series: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Named data series arrays for multi-line or multi-metric plotting",
    )
    data: List[ChartPoint] = Field(
        default_factory=list,
        description="Discrete points list for bar/pie/scatter plotting",
    )

    @model_validator(mode="before")
    @classmethod
    def reconcile_axis_labels(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "x_axis_label" in values and "x_label" not in values:
                values["x_label"] = values["x_axis_label"]
            if "y_axis_label" in values and "y_label" not in values:
                values["y_label"] = values["y_axis_label"]
        return values


class QAPair(BaseModel):
    question: str = Field(..., description="Simple, direct question about the input data")
    answer: str = Field(..., description="1-2 sentence plain-language factual answer")

    @model_validator(mode="before")
    @classmethod
    def reconcile_answer_field(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "simple_answer" in values and "answer" not in values:
                values["answer"] = values["simple_answer"]
            elif "answer" in values and "simple_answer" not in values:
                values["simple_answer"] = values["answer"]
        return values


class EngineOutput(BaseModel):
    executive_takeaways: List[str] = Field(
        ...,
        description="3-6 scannable bullet points with bold lead-ins and active verbs",
    )
    charts: List[ChartSpec] = Field(
        default_factory=list,
        description="Declarative visualization specifications ready for dynamic rendering",
    )
    qa_pairs: List[QAPair] = Field(
        ...,
        description="4-8 foundational Q&A pairs grounded strictly in the source data",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operational metadata, provenance, DSP summaries, and audit trail",
    )

    @model_validator(mode="before")
    @classmethod
    def reconcile_field_aliases(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "key_points" in values and "executive_takeaways" not in values:
                values["executive_takeaways"] = values["key_points"]
            if "questions_and_answers" in values and "qa_pairs" not in values:
                values["qa_pairs"] = values["questions_and_answers"]
            if "visualizations" in values and "charts" not in values:
                values["charts"] = values["visualizations"]
            if "summary_points" in values and "executive_takeaways" not in values:
                values["executive_takeaways"] = values["summary_points"]
        return values


class SignalFeatureSummary(BaseModel):
    modality: ModalityType
    sampling_rate_hz: float = 1.0
    duration_seconds: float = 0.0
    sample_count: int = 0
    time_domain: Dict[str, float] = Field(default_factory=dict)
    frequency_domain: Dict[str, Any] = Field(default_factory=dict)
    anomalies: Dict[str, Any] = Field(default_factory=dict)
    preview_samples: List[float] = Field(default_factory=list)
    raw_feature_map: Dict[str, Any] = Field(default_factory=dict)

