"""오디오 로딩 및 전처리 유틸리티.

ZipVoice의 load_waveform 패턴을 참고하여 구현.
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_waveform(
    path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    """오디오 파일을 로드하고 전처리.

    Args:
        path: 오디오 파일 경로
        target_sr: 목표 샘플레이트 (기본 16kHz)
        mono: 모노로 변환 여부
        max_seconds: 최대 길이 (초). None이면 전체 로드

    Returns:
        (audio_array, sample_rate)
    """
    audio, sr = sf.read(str(path), dtype="float32")

    # 스테레오 → 모노
    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 리샘플링
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 최대 길이 truncation
    if max_seconds is not None:
        max_samples = int(max_seconds * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

    return audio, sr


def save_waveform(audio: np.ndarray, sr: int, path: str | Path) -> Path:
    """오디오 배열을 파일로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path


def get_duration(path: str | Path) -> float:
    """오디오 파일의 길이(초)를 반환."""
    info = sf.info(str(path))
    return info.duration
