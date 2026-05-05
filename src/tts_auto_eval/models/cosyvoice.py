"""CosyVoice 백엔드 어댑터.

설치: git clone + pip install -e .
참고: https://github.com/FunAudioLLM/CosyVoice
"""

import logging
from pathlib import Path

import numpy as np

from tts_auto_eval.models.base import BaseTTSModel

logger = logging.getLogger(__name__)


class CosyVoiceModel(BaseTTSModel):
    """CosyVoice 모델 어댑터.

    Alibaba의 zero-shot TTS. 참조 음성으로 음성 복제 가능.
    """

    def __init__(
        self,
        model_dir: str = "pretrained_models/CosyVoice-300M",
        ref_audio: str | None = None,
        ref_text: str = "",
        mode: str = "zero_shot",
    ):
        """
        Args:
            model_dir: CosyVoice 모델 디렉토리
            ref_audio: 참조 음성 경로 (zero_shot 모드)
            ref_text: 참조 음성의 텍스트
            mode: 'zero_shot', 'cross_lingual', 'sft' 중 선택
        """
        self._model_dir = model_dir
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._mode = mode
        self._model = None

    @property
    def name(self) -> str:
        model_short = Path(self._model_dir).name
        return f"cosyvoice-{model_short}"

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice

            logger.info(f"CosyVoice 로딩 중 ({self._model_dir})...")
            self._model = CosyVoice(self._model_dir)
            logger.info("CosyVoice 로딩 완료")
        except ImportError:
            raise ImportError(
                "CosyVoice가 설치되지 않았습니다. "
                "https://github.com/FunAudioLLM/CosyVoice 참고하여 설치하세요."
            )

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import torchaudio

        self._load_model()

        if self._mode == "zero_shot" and self._ref_audio:
            output = self._model.inference_zero_shot(
                tts_text=text,
                prompt_text=self._ref_text,
                prompt_speech_16k=torchaudio.load(self._ref_audio)[0],
            )
        elif self._mode == "sft":
            output = self._model.inference_sft(text)
        else:
            output = self._model.inference_zero_shot(
                tts_text=text,
                prompt_text=self._ref_text,
                prompt_speech_16k=torchaudio.load(self._ref_audio)[0],
            )

        # CosyVoice는 generator를 반환하는 경우가 있음
        audio_list = []
        for chunk in output:
            audio_list.append(chunk["tts_speech"].numpy().squeeze())

        audio = np.concatenate(audio_list) if len(audio_list) > 1 else audio_list[0]
        return audio, 22050
