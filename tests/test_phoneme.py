"""PhonemeMetric 테스트."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tts_auto_eval.metrics.phoneme import PhonemeMetric


def test_phoneme_metric_name():
    """name 프로퍼티 확인."""
    metric = PhonemeMetric()
    assert metric.name == "phoneme"


def test_phoneme_evaluate_sample():
    """ASR + G2P 모킹 후 PER 계산 확인."""
    metric = PhonemeMetric(language="ko")

    # Mock ASR
    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = "인녕하세요"
    metric._asr = mock_asr

    # Mock G2P: 다른 음소를 반환하여 PER > 0
    mock_g2p = MagicMock()
    mock_g2p.side_effect = lambda text: f"phonemes_of_{text}"
    metric._g2p = mock_g2p

    audio = np.zeros(16000, dtype=np.float32)
    with patch("jiwer.wer", return_value=0.3):
        result = metric.evaluate_sample(audio=audio, sr=16000, text="안녕하세요")

    assert result.metric_name == "phoneme"
    assert result.details["per"] == pytest.approx(0.3)
    assert result.details["accuracy"] == pytest.approx(0.7)


def test_phoneme_perfect_match():
    """동일 음소 -> accuracy=1.0, per=0.0."""
    metric = PhonemeMetric(language="ko")

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = "안녕하세요"
    metric._asr = mock_asr

    mock_g2p = MagicMock()
    mock_g2p.side_effect = lambda text: "same_phonemes"
    metric._g2p = mock_g2p

    audio = np.zeros(16000, dtype=np.float32)
    with patch("jiwer.wer", return_value=0.0):
        result = metric.evaluate_sample(audio=audio, sr=16000, text="안녕하세요")

    assert result.score == pytest.approx(1.0)
    assert result.details["per"] == pytest.approx(0.0)


def test_phoneme_lazy_load():
    """evaluate_sample 호출 전까지 G2P가 로드되지 않음."""
    metric = PhonemeMetric(language="ko")
    assert metric._g2p is None
    assert metric._asr is None
