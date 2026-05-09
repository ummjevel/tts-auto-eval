"""TTS 모델 어댑터 기본 클래스."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from tts_auto_eval.audio import save_waveform


class BaseTTSModel(ABC):
    """TTS 모델 어댑터 기본 클래스.

    모든 TTS 백엔드는 이 클래스를 상속하여 synthesize()를 구현한다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """모델 이름 (리포트에 표시)."""
        ...

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """텍스트를 음성으로 변환.

        Returns:
            (audio_array, sample_rate)
        """
        ...

    def synthesize_to_file(self, text: str, path: str | Path) -> Path:
        """음성을 파일로 저장."""
        audio, sr = self.synthesize(text)
        return save_waveform(audio, sr, path)

    def synthesize_stream(self, text: str):
        """스트리밍 음성 합성 (청크 단위).

        Yields:
            (audio_chunk, sample_rate) 튜플

        기본 구현: 지원하지 않음.
        """
        raise NotImplementedError(f"{self.name}은 스트리밍을 지원하지 않습니다")
