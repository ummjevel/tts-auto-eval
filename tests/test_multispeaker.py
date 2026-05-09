"""MultiSpeakerMetric 테스트."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from tts_auto_eval.metrics.multispeaker import MultiSpeakerMetric


def test_multispeaker_name():
    """name 프로퍼티 확인."""
    metric = MultiSpeakerMetric()
    assert metric.name == "multispeaker"


def test_multispeaker_no_speakers():
    """speakers kwarg 없으면 score 0.0."""
    metric = MultiSpeakerMetric()
    audio = np.zeros(16000, dtype=np.float32)
    result = metric.evaluate_sample(audio=audio, sr=16000, text="hello")
    assert result.score == 0.0
    assert "error" in result.details


def test_cosine_similarity():
    """_cosine_similarity 기본 동작 검증."""
    metric = MultiSpeakerMetric()

    # 동일 벡터 -> 1.0
    a = np.array([1.0, 0.0, 0.0])
    assert metric._cosine_similarity(a, a) == pytest.approx(1.0)

    # 직교 벡터 -> 0.0
    b = np.array([0.0, 1.0, 0.0])
    assert metric._cosine_similarity(a, b) == pytest.approx(0.0)

    # 반대 벡터 -> -1.0
    c = np.array([-1.0, 0.0, 0.0])
    assert metric._cosine_similarity(a, c) == pytest.approx(-1.0)

    # 영벡터 -> 0.0
    z = np.array([0.0, 0.0, 0.0])
    assert metric._cosine_similarity(a, z) == pytest.approx(0.0)


def test_multispeaker_evaluate():
    """모킹된 speaker model로 within/cross sim 검증."""
    metric = MultiSpeakerMetric()

    # 화자 A 임베딩: [1, 0, 0], 화자 B 임베딩: [0, 1, 0]
    emb_a = np.array([1.0, 0.0, 0.0])
    emb_b = np.array([0.0, 1.0, 0.0])

    mock_model = MagicMock()
    call_count = [0]
    embeddings = [emb_a, emb_a, emb_b, emb_b]

    def mock_encode(waveform):
        import torch

        emb = embeddings[call_count[0]]
        call_count[0] += 1
        return torch.tensor(emb).unsqueeze(0)

    mock_model.encode_batch = mock_encode
    metric._speaker_model = mock_model

    # 4초 오디오, 화자 A: 0-1s, 1-2s / 화자 B: 2-3s, 3-4s
    sr = 16000
    audio = np.zeros(sr * 4, dtype=np.float32)
    speakers = [
        {"speaker_id": "A", "start": 0.0, "end": 1.0},
        {"speaker_id": "A", "start": 1.0, "end": 2.0},
        {"speaker_id": "B", "start": 2.0, "end": 3.0},
        {"speaker_id": "B", "start": 3.0, "end": 4.0},
    ]

    result = metric.evaluate_sample(audio=audio, sr=sr, text="hello", speakers=speakers)

    assert result.details["num_speakers"] == 2
    assert result.details["num_segments"] == 4
    # within: cos(a, a) = 1.0, cos(b, b) = 1.0 -> mean = 1.0
    assert result.details["within_speaker_sim"] == pytest.approx(1.0)
    # cross: cos(a, b) = 0.0 for all pairs -> mean = 0.0
    assert result.details["cross_speaker_sim"] == pytest.approx(0.0)
    # score: (1.0 + (1.0 - 0.0)) / 2.0 = 1.0
    assert result.score == pytest.approx(1.0)
