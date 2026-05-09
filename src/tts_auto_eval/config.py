"""설정 파싱 및 유효성 검증."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """TTS 모델 설정."""

    type: str = "custom"
    module: str | None = None
    params: dict = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """평가 설정."""

    language: str = "ko"
    metrics: list[str] = Field(default_factory=lambda: ["utmos", "wer"])
    whisper_model: str = "large-v3"
    device: str = "cpu"
    reference_audio: str | None = None
    model_cache_dir: str = ".cache"


class DatasetConfig(BaseModel):
    """데이터셋 설정."""

    path: str = "./datasets/ko/"
    categories: list[str] | None = None
    max_samples: int | None = None


class ReportConfig(BaseModel):
    """리포트 설정."""

    formats: list[str] = Field(default_factory=lambda: ["html", "csv"])
    output_dir: str = "./results/"
    compare_with: str | None = None
    include_audio: bool = True


class Config(BaseModel):
    """전체 설정."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


def load_config(path: str | Path) -> Config:
    """YAML 파일에서 설정 로드."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**(data or {}))


class ABTestConfig(BaseModel):
    """A/B 테스트 설정."""

    model_a: ModelConfig
    model_b: ModelConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


def load_ab_config(path: str | Path) -> "ABTestConfig":
    """A/B 테스트 설정 로드."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ABTestConfig(**(data or {}))
