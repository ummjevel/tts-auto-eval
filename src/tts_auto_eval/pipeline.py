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
    from tts_auto_eval.metrics.registry import create_metric, discover_plugins

    # 플러그인 탐색
    discover_plugins()

    # 메트릭별 공통 kwargs 매핑
    metrics = []

    for metric_name in config.evaluation.metrics:
        try:
            kwargs = _build_metric_kwargs(config, metric_name)
            metrics.append(create_metric(metric_name, **kwargs))
        except KeyError:
            logger.warning(f"알 수 없는 지표: {metric_name} (건너뜀)")
        except Exception as e:
            logger.warning(f"지표 생성 실패: {metric_name} — {e}")

    return metrics


def _build_metric_kwargs(config: Config, metric_name: str) -> dict:
    """메트릭별 생성자 kwargs를 config에서 추출."""
    device = config.evaluation.device
    cache_dir = config.evaluation.model_cache_dir
    language = config.evaluation.language
    whisper_backend = config.evaluation.whisper_backend

    # Metrics that need whisper/ASR
    if metric_name in ("wer", "itn", "phoneme"):
        return {
            "whisper_model": config.evaluation.whisper_model,
            "language": language,
            "device": device,
            "cache_dir": cache_dir,
            "whisper_backend": whisper_backend,
        }
    # Metrics that need device + cache
    elif metric_name in ("utmos", "speaker_similarity"):
        return {"device": device, "cache_dir": cache_dir}
    # Metrics with no special args
    else:
        return {}


_BUILTIN_MODELS = {
    "f5-tts": ("tts_auto_eval.models.f5_tts", "F5TTSModel"),
    "coqui": ("tts_auto_eval.models.coqui", "CoquiTTSModel"),
    "cosyvoice": ("tts_auto_eval.models.cosyvoice", "CosyVoiceModel"),
    "openai": ("tts_auto_eval.models.openai_tts", "OpenAITTSModel"),
}


def _load_model(config: Config) -> BaseTTSModel | None:
    """설정에 따라 TTS 모델 로드.

    1. type이 빌트인 모델명이면 해당 어댑터 사용
    2. type이 'custom'이면 module 경로에서 동적 로드
    3. module이 없으면 None 반환
    """
    import importlib

    model_type = config.model.type
    params = config.model.params

    # 빌트인 모델
    if model_type in _BUILTIN_MODELS:
        module_path, class_name = _BUILTIN_MODELS[model_type]
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class(**params)

    # 커스텀 모델
    if config.model.module:
        module_path, class_name = config.model.module.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class(**params)

    return None


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

    # 참조 음성 로드 (화자 유사도, PESQ, STOI 등에 필요)
    reference_audio = None
    reference_sr = None
    if config.evaluation.reference_audio:
        ref_path = Path(config.evaluation.reference_audio)
        if ref_path.exists():
            reference_audio, reference_sr = load_waveform(ref_path)
            logger.info(f"참조 음성 로드: {ref_path}")
        else:
            logger.warning(f"참조 음성 파일 없음: {ref_path}")

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
                # ITN 지표는 expected_spoken과 tags를 추가로 전달
                if metric.name == "itn":
                    from tts_auto_eval.metrics.itn import ITNMetric

                    result = metric.evaluate_sample(
                        audio=audio,
                        sr=sr,
                        text=item.text,
                        expected_spoken=item.expected_spoken,
                        tags=item.tags,
                    )
                else:
                    result = metric.evaluate_sample(
                        audio=audio,
                        sr=sr,
                        text=item.text,
                        reference_audio=reference_audio,
                        reference_sr=reference_sr,
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
