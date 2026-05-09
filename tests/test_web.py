"""웹 대시보드 테스트."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from tts_auto_eval.web.app import _build_summary_table, _load_results_from_dir


def test_load_results_from_dir(tmp_path: Path):
    """result.json 파일 로드 테스트."""
    sub = tmp_path / "model_a"
    sub.mkdir()
    result_data = {
        "metadata": {"model_name": "test_model", "language": "ko", "total_samples": 10},
        "summary": {"utmos": {"mean": 3.5}},
    }
    (sub / "result.json").write_text(json.dumps(result_data), encoding="utf-8")

    results = _load_results_from_dir(tmp_path)
    assert len(results) == 1
    assert results[0]["metadata"]["model_name"] == "test_model"
    assert "_source_path" in results[0]


def test_build_summary_table():
    """요약 테이블 변환 테스트."""
    results = [
        {
            "metadata": {
                "model_name": "ModelA",
                "language": "ko",
                "total_samples": 5,
                "timestamp": "2025-01-01T12:00:00Z",
            },
            "summary": {
                "utmos": {"mean": 4.0},
                "wer": {"sentence_wer": 0.05},
            },
        },
    ]
    table = _build_summary_table(results)
    assert len(table) == 1
    row = table[0]
    assert row[0] == "ModelA"
    assert row[1] == "ko"
    assert row[2] == "5"
    assert row[4] == "4.0000"  # utmos
    assert row[5] == "0.0500"  # wer


def test_create_dashboard(tmp_path: Path):
    """create_dashboard가 정상 실행되는지 테스트 (gradio mock)."""
    mock_gr = MagicMock()
    mock_blocks = MagicMock()
    mock_gr.Blocks.return_value.__enter__ = MagicMock(return_value=mock_blocks)
    mock_gr.Blocks.return_value.__exit__ = MagicMock(return_value=False)
    mock_gr.themes.Soft.return_value = MagicMock()
    mock_gr.Tab.return_value.__enter__ = MagicMock()
    mock_gr.Tab.return_value.__exit__ = MagicMock(return_value=False)
    mock_gr.Row.return_value.__enter__ = MagicMock()
    mock_gr.Row.return_value.__exit__ = MagicMock(return_value=False)

    with patch.dict("sys.modules", {"gradio": mock_gr}):
        from tts_auto_eval.web.app import create_dashboard

        result = create_dashboard(results_dir=str(tmp_path))
        # Should return the Blocks context manager result
        assert result is not None
