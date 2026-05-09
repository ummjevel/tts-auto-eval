"""ASR 추상화 레이어 단위 테스트."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_auto_eval.asr import BaseASR, create_asr
from tts_auto_eval.asr.whisper_backend import OpenAIWhisperASR
from tts_auto_eval.asr.faster_whisper_backend import FasterWhisperASR


# ---------------------------------------------------------------------------
# create_asr factory tests
# ---------------------------------------------------------------------------


def test_create_asr_default_returns_openai():
    """create_asr() 기본값은 OpenAIWhisperASR을 반환."""
    result = create_asr()
    assert isinstance(result, OpenAIWhisperASR)


def test_create_asr_faster_whisper():
    """create_asr(backend='faster-whisper')는 FasterWhisperASR을 반환."""
    result = create_asr(backend="faster-whisper")
    assert isinstance(result, FasterWhisperASR)


def test_create_asr_unknown_backend_returns_openai():
    """알 수 없는 backend는 else 분기 → OpenAIWhisperASR으로 폴백."""
    result = create_asr(backend="unknown")
    assert isinstance(result, OpenAIWhisperASR)


# ---------------------------------------------------------------------------
# OpenAIWhisperASR tests
# ---------------------------------------------------------------------------


def test_openai_whisper_asr_lazy_load():
    """모델은 생성 시 로딩하지 않고 transcribe() 호출 시 lazy load."""
    asr = OpenAIWhisperASR(model_name="base", device="cpu", cache_dir="/tmp/test")
    assert asr._model is None


def test_openai_whisper_asr_transcribe():
    """OpenAIWhisperASR.transcribe()가 올바르게 텍스트를 반환하는지 검증."""
    mock_whisper = MagicMock()
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    mock_model.transcribe.return_value = {"text": "안녕하세요"}

    mock_librosa = MagicMock()

    with patch.dict(sys.modules, {"whisper": mock_whisper, "librosa": mock_librosa}):
        asr = OpenAIWhisperASR(model_name="base", device="cpu", cache_dir="/tmp/test")
        audio = np.zeros(16000, dtype=np.float32)
        result = asr.transcribe(audio, 16000, "ko")

    assert result == "안녕하세요"
    mock_whisper.load_model.assert_called_once()
    mock_model.transcribe.assert_called_once()


def test_openai_whisper_asr_resamples_non_16k():
    """16kHz가 아닌 오디오는 리샘플링."""
    mock_whisper = MagicMock()
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    mock_model.transcribe.return_value = {"text": "test"}

    mock_librosa = MagicMock()
    mock_librosa.resample.return_value = np.zeros(16000, dtype=np.float32)

    with patch.dict(sys.modules, {"whisper": mock_whisper, "librosa": mock_librosa}):
        asr = OpenAIWhisperASR(model_name="base", device="cpu", cache_dir="/tmp/test")
        audio = np.zeros(44100, dtype=np.float32)
        asr.transcribe(audio, 44100, "ko")

    mock_librosa.resample.assert_called_once()


# ---------------------------------------------------------------------------
# FasterWhisperASR tests
# ---------------------------------------------------------------------------


def test_faster_whisper_asr_lazy_load():
    """FasterWhisperASR도 lazy load 패턴 사용."""
    asr = FasterWhisperASR(model_name="base", device="cpu", cache_dir="/tmp/test")
    assert asr._model is None


def test_faster_whisper_asr_transcribe():
    """FasterWhisperASR.transcribe()가 세그먼트를 조인하여 텍스트 반환."""
    mock_model = MagicMock()

    seg1 = MagicMock()
    seg1.text = "안녕"
    seg2 = MagicMock()
    seg2.text = "하세요"
    mock_model.transcribe.return_value = (iter([seg1, seg2]), MagicMock())

    mock_librosa = MagicMock()

    with patch.dict(sys.modules, {"librosa": mock_librosa}):
        asr = FasterWhisperASR(model_name="base", device="cpu", cache_dir="/tmp/test")
        # Inject mock model directly to bypass _load_model's import of faster_whisper
        asr._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = asr.transcribe(audio, 16000, "ko")

    assert result == "안녕 하세요"
    mock_model.transcribe.assert_called_once()


def test_faster_whisper_asr_load_model():
    """_load_model이 faster_whisper.WhisperModel을 사용하여 모델 생성."""
    mock_faster_whisper = MagicMock()
    mock_whisper_model_cls = MagicMock()
    mock_faster_whisper.WhisperModel = mock_whisper_model_cls

    with patch.dict(sys.modules, {"faster_whisper": mock_faster_whisper}):
        asr = FasterWhisperASR(model_name="base", device="cpu", cache_dir="/tmp/test")
        asr._load_model()

    mock_whisper_model_cls.assert_called_once()
    assert asr._model is mock_whisper_model_cls.return_value


# ---------------------------------------------------------------------------
# WERMetric / ITNMetric integration with ASR backend
# ---------------------------------------------------------------------------


@patch("tts_auto_eval.asr.create_asr")
def test_wer_metric_uses_asr_backend(mock_create_asr):
    """WERMetric이 whisper_backend='faster-whisper'로 create_asr를 호출하는지 검증."""
    from tts_auto_eval.metrics.intelligibility import WERMetric

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = "테스트 문장"
    mock_create_asr.return_value = mock_asr

    metric = WERMetric(
        whisper_model="base",
        language="ko",
        device="cpu",
        cache_dir="/tmp/test",
        whisper_backend="faster-whisper",
    )

    audio = np.zeros(16000, dtype=np.float32)
    metric._transcribe(audio, 16000)

    mock_create_asr.assert_called_once_with(
        backend="faster-whisper",
        model_name="base",
        device="cpu",
        cache_dir="/tmp/test",
    )
    mock_asr.transcribe.assert_called_once_with(audio, 16000, "ko")


@patch("tts_auto_eval.asr.create_asr")
def test_itn_metric_uses_asr_backend(mock_create_asr):
    """ITNMetric이 whisper_backend='faster-whisper'로 create_asr를 호출하는지 검증."""
    from tts_auto_eval.metrics.itn import ITNMetric

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = "테스트 문장"
    mock_create_asr.return_value = mock_asr

    metric = ITNMetric(
        whisper_model="base",
        language="ko",
        device="cpu",
        cache_dir="/tmp/test",
        whisper_backend="faster-whisper",
    )

    audio = np.zeros(16000, dtype=np.float32)
    metric._transcribe(audio, 16000)

    mock_create_asr.assert_called_once_with(
        backend="faster-whisper",
        model_name="base",
        device="cpu",
        cache_dir="/tmp/test",
    )
    mock_asr.transcribe.assert_called_once_with(audio, 16000, "ko")
