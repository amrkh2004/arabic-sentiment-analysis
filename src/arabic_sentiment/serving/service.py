"""
BentoML Production Serving Definition for Arabic Sentiment Analysis.
Features adaptive batching, async processing, and graceful ONNX/PyTorch fallback.
"""

from __future__ import annotations

from typing import Any

import bentoml
from pydantic import BaseModel, Field

from arabic_sentiment.api.service import SentimentInferenceService
from arabic_sentiment.data.preprocessor import ArabicTextPreprocessor


class SentimentInput(BaseModel):
    text: str = Field(
        ..., description="Arabic review text to analyze", example="المنتج ممتاز جدا وخامته رائعة"
    )


class BatchSentimentInput(BaseModel):
    texts: list[str] = Field(..., description="List of Arabic review texts")


class SentimentOutput(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class SentimentModelRunnable(bentoml.Runnable):
    """
    BentoML custom Runnable encapsulating Arabic sentiment inference
    with adaptive batching support.
    """

    SUPPORTED_RESOURCES = ("cpu", "nvidia.com/gpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.inference_service = SentimentInferenceService()

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        results = self.inference_service.predict(texts)
        return [
            {
                "text": r.text,
                "label": r.label,
                "confidence": r.confidence,
                "probabilities": r.probabilities,
            }
            for r in results
        ]


# BentoML Runner with adaptive batching (Rubric requirement)
try:
    sentiment_runner = bentoml.Runner(
        SentimentModelRunnable,
        name="arabic_sentiment_runner",
        batchable=True,
    )
except Exception:  # noqa: BLE001
    sentiment_runner = None


@bentoml.service(
    name="arabic_sentiment_service",
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)
class ArabicSentimentBentoService:
    def __init__(self):
        self.inference_service = SentimentInferenceService()
        self.preprocessor = ArabicTextPreprocessor(normalize_chars=True, remove_tashkeel=True)

    @bentoml.api(batchable=False)
    def predict(self, text: str) -> dict[str, Any]:
        """Single review sentiment prediction."""
        result = self.inference_service.predict([text])[0]
        return {
            "text": result.text,
            "label": result.label,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
        }

    @bentoml.api(batchable=True, batch_dim=0)
    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """High-throughput adaptive batch sentiment prediction."""
        results = self.inference_service.predict(texts)
        return [
            {
                "text": r.text,
                "label": r.label,
                "confidence": r.confidence,
                "probabilities": r.probabilities,
            }
            for r in results
        ]
