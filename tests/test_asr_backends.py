"""Paraformer / NeMo ASR 백엔드 단위 테스트."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_auto_eval.asr import create_asr
from tts_auto_eval.asr.paraformer_backend import ParaformerASR
from tts_auto_eval.asr.nemo_backend import NeMoASR


# ---------------------------------------------------------------------------
# create_asr factory tests
# ---------------------------------------------------------------------------


def test_create_asr_paraformer():
    """create_asr(backend='paraformer')는 ParaformerASR을 반환."""
    result = create_asr(backend="paraformer")
    assert isinstance(result, ParaformerASR)


def test_create_asr_nemo():
    """create_asr(backend='nemo')는 NeMoASR을 반환."""
    result = create_asr(backend="nemo")
    assert isinstance(result, NeMoASR)


# ---------------------------------------------------------------------------
# ParaformerASR tests
# ---------------------------------------------------------------------------


def test_paraformer_lazy_load():
    """ParaformerASR 생성 시 모델은 None."""
    asr = ParaformerASR(model_name="paraformer-zh", device="cpu", cache_dir="/tmp/test")
    assert asr._model is None


def test_paraformer_transcribe():
    """ParaformerASR.transcribe()가 올바르게 텍스트를 반환하는지 검증."""
    mock_funasr = MagicMock()
    mock_model = MagicMock()
    mock_funasr.AutoModel.return_value = mock_model
    mock_model.generate.return_value = [{"text": "你好世界"}]

    mock_librosa = MagicMock()

    with patch.dict(sys.modules, {"funasr": mock_funasr, "librosa": mock_librosa}):
        asr = ParaformerASR(model_name="paraformer-zh", device="cpu", cache_dir="/tmp/test")
        audio = np.zeros(16000, dtype=np.float32)
        result = asr.transcribe(audio, 16000, "zh")

    assert result == "你好世界"
    mock_model.generate.assert_called_once()


# ---------------------------------------------------------------------------
# NeMoASR tests
# ---------------------------------------------------------------------------


def test_nemo_lazy_load():
    """NeMoASR 생성 시 모델은 None."""
    asr = NeMoASR(model_name="conformer", device="cpu", cache_dir="/tmp/test")
    assert asr._model is None


def test_nemo_transcribe():
    """NeMoASR.transcribe()가 올바르게 텍스트를 반환하는지 검증."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ["hello world"]

    mock_librosa = MagicMock()
    mock_sf = MagicMock()

    with patch.dict(sys.modules, {"librosa": mock_librosa, "soundfile": mock_sf}):
        asr = NeMoASR(model_name="conformer", device="cpu", cache_dir="/tmp/test")
        # Inject mock model to bypass _load_model's import
        asr._model = mock_model

        audio = np.zeros(16000, dtype=np.float32)
        result = asr.transcribe(audio, 16000, "en")

    assert result == "hello world"
    mock_model.transcribe.assert_called_once()
