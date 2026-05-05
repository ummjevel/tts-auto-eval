"""메인 평가 파이프라인."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tts_auto_eval.audio import load_waveform, save_waveform
from tts_auto_eval.config import Config
from tts_auto_eval.datasets.loader import TestDataset, load_datasets_from_dir
from tts_auto_eval.metrics.base import BaseMetric, MetricResult
from tts_auto_eval.models.base import BaseTTSModel

logger = logging.getLogger(__name__)


def _create_metrics(config: Config) -> list[BaseMetric]:
    """설정에 따라 평가 지표 인스턴스 생성."""
    metrics = []
    device = config.evaluation.device

    for metric_name in config.evaluation.metrics:
        if metric_name == "utmos":
            from tts_auto_eval.metrics.quality import UTMOSMetric

            metrics.append(UTMOSMetric(device=device))
        elif metric_name == "wer":
            from tts_auto_eval.metrics.intelligibility import WERMetric

            metrics.append(
                WERMetric(
                    whisper_model=config.evaluation.whisper_model,
                    language=config.evaluation.language,
                    device=device,
                )
            )
        else:
            logger.warning(f"알 수 없는 지표: {metric_name} (건너뜀)")

    return metrics


def _load_model(config: Config) -> BaseTTSModel | None:
    """설정에 따라 TTS 모델 로드."""
    if config.model.module is None:
        return None

    import importlib

    module_path, class_name = config.model.module.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class(**config.model.params)


def run_evaluation(
    config: Config,
    model: BaseTTSModel | None = None,
    audio_dir: Path | None = None,
) -> dict:
    """평가 파이프라인 실행.

    두 가지 모드:
    1. model이 주어지면: 텍스트 → 음성 생성 → 평가
    2. audio_dir이 주어지면: 기존 음성 파일로 평가 (eval-only)
    """
    output_dir = Path(config.report.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir = output_dir / "audio"

    # 데이터셋 로드
    datasets = load_datasets_from_dir(
        config.dataset.path,
        categories=config.dataset.categories,
        max_samples=config.dataset.max_samples,
    )

    if not datasets:
        raise ValueError(f"데이터셋을 찾을 수 없습니다: {config.dataset.path}")

    # 모델 로드 (설정에서)
    if model is None and config.model.module:
        model = _load_model(config)

    # 지표 생성
    metrics = _create_metrics(config)
    if not metrics:
        raise ValueError("평가할 지표가 없습니다")

    logger.info(f"평가 지표: {[m.name for m in metrics]}")

    # 전체 결과 저장
    all_results: dict[str, list[MetricResult]] = {m.name: [] for m in metrics}
    sample_details = []

    for dataset in datasets:
        logger.info(f"데이터셋 평가 중: {dataset.language}/{dataset.category} ({len(dataset.items)}개)")

        for item in tqdm(dataset.items, desc=f"{dataset.category}"):
            audio = None
            sr = None

            # 음성 생성 또는 로드
            if model is not None:
                audio, sr = model.synthesize(item.text)
                audio_output_dir.mkdir(parents=True, exist_ok=True)
                save_waveform(audio, sr, audio_output_dir / f"{item.id}.wav")
            elif audio_dir is not None:
                audio_path = audio_dir / f"{item.id}.wav"
                if not audio_path.exists():
                    logger.warning(f"음성 파일 없음: {audio_path} (건너뜀)")
                    continue
                audio, sr = load_waveform(audio_path)
            else:
                raise ValueError("모델 또는 audio_dir 중 하나는 필요합니다")

            # 각 지표로 평가
            sample_result = {
                "id": item.id,
                "text": item.text,
                "category": dataset.category,
                "tags": item.tags,
            }

            for metric in metrics:
                result = metric.evaluate_sample(
                    audio=audio,
                    sr=sr,
                    text=item.text,
                )
                result.sample_id = item.id
                all_results[metric.name].append(result)
                sample_result[metric.name] = result.score
                sample_result[f"{metric.name}_details"] = result.details

            sample_details.append(sample_result)

    # 집계
    summary = {}
    for metric in metrics:
        results = all_results[metric.name]
        if results:
            summary[metric.name] = metric.aggregate(results)

    # 결과 구조
    eval_result = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model.name if model else "eval-only",
            "language": config.evaluation.language,
            "metrics": config.evaluation.metrics,
            "total_samples": len(sample_details),
        },
        "summary": summary,
        "samples": sample_details,
    }

    # JSON으로 원시 결과 저장
    result_path = output_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"결과 저장: {result_path}")

    return eval_result
