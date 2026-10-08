from pydantic import BaseModel, Field, field_validator, model_validator


class PredictRequest(BaseModel):
    """Input payload for sentiment prediction."""

    text: str | None = Field(
        default=None,
        description="Single Arabic review text",
        examples=["المنتج ممتاز جدا وخاماته عالية الجودة والتوصيل كان سريع"],
    )
    texts: list[str] | None = Field(
        default=None,
        description="Batch of Arabic review texts",
        examples=[
            [
                "المنتج ممتاز والتوصيل سريع جدا",
                "جودة سيئة ولن أشتري مرة أخرى",
                "المنتج عادي ومقبول بالنسبة لسعره",
            ]
        ],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("Review text cannot be empty or whitespace only.")
        return v

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            if len(v) == 0:
                raise ValueError("Batch texts list cannot be empty.")
            for t in v:
                if not t or not t.strip():
                    raise ValueError("All reviews in the batch must be non-empty strings.")
        return v

    @model_validator(mode="after")
    def validate_at_least_one_field(self) -> "PredictRequest":
        if self.text is None and self.texts is None:
            raise ValueError(
                "Either 'text' or 'texts' field must be provided in the request payload."
            )
        return self


class SentimentPrediction(BaseModel):
    """Individual review sentiment prediction result."""

    text: str = Field(..., description="Original input text")
    label: str = Field(..., description="Predicted sentiment class (positive, neutral, negative)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score")
    probabilities: dict[str, float] = Field(
        ..., description="Probability distribution across all 3 classes"
    )
    model_version: str = Field(
        default="student-int8-onnx-v1", description="Serving model version identifier"
    )


class PredictResponse(BaseModel):
    """Output payload from sentiment prediction endpoint."""

    label: str = Field(
        ..., description="Primary predicted sentiment class (positive, neutral, negative)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated confidence score")
    model_version: str = Field(
        default="student-int8-onnx-v1", description="Serving model version identifier"
    )
    results: list[SentimentPrediction] = Field(
        default_factory=list, description="List of predictions for all input texts"
    )
    latency_ms: float = Field(default=0.0, ge=0.0, description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Service health status payload."""

    status: str = Field(default="healthy", description="Service status")
    model_loaded: bool = Field(..., description="Whether inference model is loaded and ready")
    model_type: str = Field(default="ONNX_INT8", description="Inference engine backend")
    app_version: str = Field(default="0.1.0", description="Application package version")
    supported_labels: list[str] = Field(
        default=["negative", "neutral", "positive"], description="Sentiment classes"
    )
