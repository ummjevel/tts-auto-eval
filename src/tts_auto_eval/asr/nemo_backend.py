"""NVIDIA NeMo ASR 백엔드 — Conformer/Canary 모델."""

import logging
from pathlib import Path

import numpy as np

from tts_auto_eval.asr import BaseASR

logger = logging.getLogger(__name__)


class NeMoASR(BaseASR):
    """NVIDIA NeMo ASR."""

    # 모델명 매핑
    _MODEL_MAP = {
        "canary": "nvidia/canary-1b",
        "conformer": "nvidia/stt_en_conformer_ctc_large",
        "conformer-ko": "nvidia/stt_ko_conformer_ctc_large",
    }

    def __init__(self, model_name: str = "conformer", device: str = "cpu", cache_dir: str = ".cache"):
        self._model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr

        pretrained_name = self._MODEL_MAP.get(self._model_name, self._model_name)

        logger.info(f"NeMo ASR {pretrained_name} 모델 로딩 중...")
        self._model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=pretrained_name,
        )
        if self._device.startswith("cuda"):
            self._model = self._model.cuda()
        logger.info("NeMo ASR 모델 로딩 완료")

    def transcribe(self, audio: np.ndarray, sr: int, language: str) -> str:
        import librosa
        import tempfile
        import soundfile as sf

        self._load_model()

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # NeMo expects file path, so write temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 16000)
            result = self._model.transcribe([f.name])

        if result and len(result) > 0:
            # result can be list of strings or list of Hypothesis
            text = result[0] if isinstance(result[0], str) else result[0].text
            return text.strip()
        return ""
