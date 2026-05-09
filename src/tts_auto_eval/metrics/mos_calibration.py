"""MOS 주관 평가 연동 — 인간 MOS ↔ UTMOS 예측 상관분석."""

import json
import logging
from pathlib import Path

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class MOSCalibrationMetric(BaseMetric):
    """MOS 캘리브레이션 지표.

    인간 MOS 점수와 UTMOS 예측값 간의 상관도를 분석.
    human_scores: {sample_id: float} 매핑 (JSON/CSV 파일에서 로드)
    """

    def __init__(self, human_scores_path: str | None = None, device: str = "cpu", cache_dir: str = ".cache"):
        self._device = device
        self._cache_dir = cache_dir
        self._human_scores: dict[str, float] = {}
        self._utmos_model = None
        if human_scores_path:
            self._load_human_scores(human_scores_path)

    @property
    def name(self) -> str:
        return "mos_calibration"

    def _load_human_scores(self, path: str) -> None:
        """인간 MOS 점수 로드 (JSON: {sample_id: score})."""
        path = Path(path)
        if path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                self._human_scores = json.load(f)
        elif path.suffix == ".csv":
            import csv

            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._human_scores[row["id"]] = float(row["mos"])
        logger.info(f"Human MOS scores loaded: {len(self._human_scores)} samples")

    def _predict_utmos(self, audio: np.ndarray, sr: int) -> float:
        """UTMOS 예측."""
        if self._utmos_model is None:
            try:
                import torch

                self._utmos_model = torch.hub.load(
                    "tarepan/SpeechMOS:v1.2.0",
                    "utmos22_strong",
                    trust_repo=True,
                )
                logger.info("UTMOS model loaded for MOS calibration")
            except Exception as e:
                logger.warning(f"UTMOS load failed: {e}")
                return 0.0

        import torch
        import librosa

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = self._utmos_model(waveform, sr)
        return float(score.item())

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
        **kwargs,
    ) -> MetricResult:
        sample_id = kwargs.get("sample_id", "")
        predicted_mos = self._predict_utmos(audio, sr)
        human_mos = self._human_scores.get(sample_id)

        return MetricResult(
            sample_id=sample_id,
            metric_name=self.name,
            score=predicted_mos,
            details={
                "predicted_mos": predicted_mos,
                "human_mos": human_mos,
                "has_human_score": human_mos is not None,
            },
        )

    def aggregate(self, results: list[MetricResult]) -> dict:
        """Pearson/Spearman 상관도 계산."""
        paired = [
            (r.details["predicted_mos"], r.details["human_mos"])
            for r in results
            if r.details.get("human_mos") is not None
        ]

        base = {
            "mean_predicted": float(np.mean([r.score for r in results])) if results else 0.0,
            "total_samples": len(results),
            "paired_samples": len(paired),
            "count": len(results),
        }

        if len(paired) >= 3:
            try:
                from scipy.stats import pearsonr, spearmanr

                predicted = [p[0] for p in paired]
                human = [p[1] for p in paired]
                base["pearson_r"] = float(pearsonr(predicted, human)[0])
                base["spearman_r"] = float(spearmanr(predicted, human)[0])
            except ImportError:
                logger.warning("scipy not installed, skipping correlation")

        return base
