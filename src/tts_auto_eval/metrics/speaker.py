"""화자 유사도 평가 - Speaker Cosine Similarity (ECAPA-TDNN)."""

import logging

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class SpeakerSimilarityMetric(BaseMetric):
    """화자 유사도 (SIM-o).

    ECAPA-TDNN 임베딩의 코사인 유사도로 측정.
    참조 음성(reference_audio)이 필요.
    """

    def __init__(self, device: str = "cpu", cache_dir: str = ".cache"):
        self._device = device
        self._cache_dir = cache_dir
        self._model = None

    @property
    def name(self) -> str:
        return "speaker_similarity"

    def _load_model(self):
        if self._model is not None:
            return

        from pathlib import Path
        from speechbrain.inference.speaker import EncoderClassifier

        savedir = Path(self._cache_dir) / "speechbrain" / "spkrec-ecapa-voxceleb"
        savedir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ECAPA-TDNN 모델 로딩 중... (cache: {savedir})")
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(savedir),
            run_opts={"device": self._device},
        )
        logger.info("ECAPA-TDNN 모델 로딩 완료")

    def _get_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import torch
        import librosa

        self._load_model()

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        embedding = self._model.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
    ) -> MetricResult:
        if reference_audio is None:
            return MetricResult(
                sample_id="",
                metric_name=self.name,
                score=0.0,
                details={"error": "참조 음성 없음"},
            )

        emb_synth = self._get_embedding(audio, sr)
        emb_ref = self._get_embedding(reference_audio, reference_sr or sr)

        # 코사인 유사도
        cos_sim = float(
            np.dot(emb_synth, emb_ref)
            / (np.linalg.norm(emb_synth) * np.linalg.norm(emb_ref) + 1e-8)
        )

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=cos_sim,
            details={"cosine_similarity": cos_sim},
        )
