"""OpenAI TTS API 백엔드 어댑터.

설치: pip install openai
참고: https://platform.openai.com/docs/guides/text-to-speech
"""

import io
import logging
import os

import numpy as np

from tts_auto_eval.models.base import BaseTTSModel

logger = logging.getLogger(__name__)


class OpenAITTSModel(BaseTTSModel):
    """OpenAI TTS API 어댑터.

    상용 API. 환경변수 OPENAI_API_KEY 필요.
    """

    def __init__(
        self,
        model: str = "tts-1-hd",
        voice: str = "alloy",
        speed: float = 1.0,
        api_key: str | None = None,
    ):
        """
        Args:
            model: 'tts-1' (빠름) 또는 'tts-1-hd' (고품질) 또는 'gpt-4o-mini-tts'
            voice: 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'
            speed: 0.25 ~ 4.0
            api_key: OpenAI API 키 (None이면 환경변수 사용)
        """
        self._model = model
        self._voice = voice
        self._speed = speed
        self._api_key = api_key
        self._client = None

    @property
    def name(self) -> str:
        return f"openai-{self._model}-{self._voice}"

    def _load_client(self):
        if self._client is not None:
            return

        try:
            from openai import OpenAI

            key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY 환경변수를 설정하거나 api_key 파라미터를 전달하세요."
                )
            self._client = OpenAI(api_key=key)
            logger.info(f"OpenAI TTS 클라이언트 초기화 (model={self._model}, voice={self._voice})")
        except ImportError:
            raise ImportError(
                "openai 패키지가 설치되지 않았습니다. `pip install openai`로 설치하세요."
            )

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        self._load_client()

        response = self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            speed=self._speed,
            response_format="wav",
        )

        audio_bytes = response.content
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # 스테레오 → 모노
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio, sr
