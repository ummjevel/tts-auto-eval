"""ASR 백엔드 추상화.

openai-whisper와 faster-whisper를 동일 인터페이스로 사용.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseASR(ABC):
    """ASR 백엔드 기본 클래스."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int, language: str) -> str:
        """오디오를 텍스트로 변환."""
        ...


def create_asr(
    backend: str = "openai-whisper",
    model_name: str = "large-v3",
    device: str = "cpu",
    cache_dir: str = ".cache",
) -> BaseASR:
    """팩토리: ASR 백엔드 생성."""
    if backend == "faster-whisper":
        from tts_auto_eval.asr.faster_whisper_backend import FasterWhisperASR

        return FasterWhisperASR(model_name=model_name, device=device, cache_dir=cache_dir)
    elif backend == "paraformer":
        from tts_auto_eval.asr.paraformer_backend import ParaformerASR

        return ParaformerASR(model_name=model_name, device=device, cache_dir=cache_dir)
    elif backend == "nemo":
        from tts_auto_eval.asr.nemo_backend import NeMoASR

        return NeMoASR(model_name=model_name, device=device, cache_dir=cache_dir)
    else:
        from tts_auto_eval.asr.whisper_backend import OpenAIWhisperASR

        return OpenAIWhisperASR(model_name=model_name, device=device, cache_dir=cache_dir)
