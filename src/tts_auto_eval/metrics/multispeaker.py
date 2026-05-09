"""다화자 대화 TTS 평가 — cpSIM, 화자 일관성, 화자 전환 자연스러움."""

import logging

import numpy as np

from tts_auto_eval.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class MultiSpeakerMetric(BaseMetric):
    """다화자 TTS 평가.

    - within_speaker_sim: 동일 화자 세그먼트 간 임베딩 유사도 (높을수록 좋음)
    - cross_speaker_sim (cpSIM): 서로 다른 화자 간 유사도 (낮을수록 좋음 = 구별 잘 됨)
    """

    def __init__(self, device: str = "cpu", cache_dir: str = ".cache"):
        self._device = device
        self._cache_dir = cache_dir
        self._speaker_model = None

    @property
    def name(self) -> str:
        return "multispeaker"

    def _load_speaker_model(self):
        if self._speaker_model is not None:
            return
        try:
            from speechbrain.inference.speaker import EncoderClassifier

            self._speaker_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(f"{self._cache_dir}/speechbrain_spkrec"),
                run_opts={"device": self._device},
            )
            logger.info("Speaker model (ECAPA-TDNN) loaded for multi-speaker eval")
        except ImportError:
            logger.warning("speechbrain not installed")
            self._speaker_model = None

    def _get_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        """음성에서 화자 임베딩 추출."""
        self._load_speaker_model()
        if self._speaker_model is None:
            return None
        import torch
        import librosa

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        embedding = self._speaker_model.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def evaluate_sample(
        self,
        audio: np.ndarray,
        sr: int,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
        **kwargs,
    ) -> MetricResult:
        speakers = kwargs.get("speakers")
        if not speakers or len(speakers) < 2:
            return MetricResult(
                sample_id="",
                metric_name=self.name,
                score=0.0,
                details={"error": "speakers config required (at least 2 speakers)"},
            )

        # Extract embeddings per speaker segment
        speaker_embeddings: dict[str, list[np.ndarray]] = {}
        for seg in speakers:
            speaker_id = seg.get("speaker_id", "unknown")
            start_s = seg.get("start", 0.0)
            end_s = seg.get("end", len(audio) / sr)
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < sr * 0.5:  # Skip segments < 0.5s
                continue

            emb = self._get_embedding(segment_audio, sr)
            if emb is not None:
                if speaker_id not in speaker_embeddings:
                    speaker_embeddings[speaker_id] = []
                speaker_embeddings[speaker_id].append(emb)

        if len(speaker_embeddings) < 2:
            return MetricResult(
                sample_id="",
                metric_name=self.name,
                score=0.0,
                details={"error": "not enough speaker segments"},
            )

        # Within-speaker similarity (should be high)
        within_sims = []
        for spk_id, embs in speaker_embeddings.items():
            if len(embs) >= 2:
                for i in range(len(embs)):
                    for j in range(i + 1, len(embs)):
                        within_sims.append(self._cosine_similarity(embs[i], embs[j]))

        # Cross-speaker similarity (should be low)
        cross_sims = []
        speaker_ids = list(speaker_embeddings.keys())
        for i in range(len(speaker_ids)):
            for j in range(i + 1, len(speaker_ids)):
                for emb_a in speaker_embeddings[speaker_ids[i]]:
                    for emb_b in speaker_embeddings[speaker_ids[j]]:
                        cross_sims.append(self._cosine_similarity(emb_a, emb_b))

        within_sim = float(np.mean(within_sims)) if within_sims else 1.0
        cross_sim = float(np.mean(cross_sims)) if cross_sims else 0.0

        # Score: high within-speaker + low cross-speaker
        score = (within_sim + (1.0 - cross_sim)) / 2.0

        return MetricResult(
            sample_id="",
            metric_name=self.name,
            score=score,
            details={
                "within_speaker_sim": within_sim,
                "cross_speaker_sim": cross_sim,
                "num_speakers": len(speaker_embeddings),
                "num_segments": sum(len(v) for v in speaker_embeddings.values()),
            },
        )
