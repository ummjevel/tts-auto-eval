"""Tests for tts_auto_eval.ab_test module."""

from pathlib import Path

import pytest

from tts_auto_eval.ab_test import _compute_win_rates, generate_ab_report_markdown


def _make_result(model_name, samples, summary=None):
    return {
        "metadata": {"model_name": model_name, "total_samples": len(samples)},
        "summary": summary or {},
        "samples": samples,
    }


class TestComputeWinRates:
    def test_compute_win_rates_basic(self):
        """model_a beats model_b on utmos (higher=better)."""
        samples_a = [
            {"id": "s1", "utmos": 4.5},
            {"id": "s2", "utmos": 4.0},
            {"id": "s3", "utmos": 3.8},
        ]
        samples_b = [
            {"id": "s1", "utmos": 3.0},
            {"id": "s2", "utmos": 3.5},
            {"id": "s3", "utmos": 3.2},
        ]
        result_a = _make_result("ModelA", samples_a, summary={"utmos": {"mean": 4.1}})
        result_b = _make_result("ModelB", samples_b, summary={"utmos": {"mean": 3.2}})

        wr = _compute_win_rates(result_a, result_b)

        assert "utmos" in wr
        assert wr["utmos"]["model_a_wins"] == 3
        assert wr["utmos"]["model_b_wins"] == 0
        assert wr["utmos"]["ties"] == 0
        assert wr["utmos"]["total"] == 3
        assert wr["utmos"]["model_a_win_rate"] == 1.0

    def test_compute_win_rates_lower_is_better(self):
        """model_a has lower WER (wins) since lower is better for WER."""
        samples_a = [
            {"id": "s1", "wer": 0.05},
            {"id": "s2", "wer": 0.10},
        ]
        samples_b = [
            {"id": "s1", "wer": 0.15},
            {"id": "s2", "wer": 0.20},
        ]
        result_a = _make_result("ModelA", samples_a, summary={"wer": {"sentence_wer": 0.075}})
        result_b = _make_result("ModelB", samples_b, summary={"wer": {"sentence_wer": 0.175}})

        wr = _compute_win_rates(result_a, result_b)

        assert wr["wer"]["model_a_wins"] == 2
        assert wr["wer"]["model_b_wins"] == 0

    def test_compute_win_rates_ties(self):
        """Same scores for both models, verify ties count."""
        samples_a = [
            {"id": "s1", "utmos": 4.0},
            {"id": "s2", "utmos": 3.5},
        ]
        samples_b = [
            {"id": "s1", "utmos": 4.0},
            {"id": "s2", "utmos": 3.5},
        ]
        result_a = _make_result("ModelA", samples_a, summary={"utmos": {"mean": 3.75}})
        result_b = _make_result("ModelB", samples_b, summary={"utmos": {"mean": 3.75}})

        wr = _compute_win_rates(result_a, result_b)

        assert wr["utmos"]["ties"] == 2
        assert wr["utmos"]["model_a_wins"] == 0
        assert wr["utmos"]["model_b_wins"] == 0

    def test_compute_win_rates_mixed(self):
        """Some samples a wins, some b wins."""
        samples_a = [
            {"id": "s1", "utmos": 4.5},
            {"id": "s2", "utmos": 3.0},
            {"id": "s3", "utmos": 4.0},
            {"id": "s4", "utmos": 2.5},
        ]
        samples_b = [
            {"id": "s1", "utmos": 3.0},
            {"id": "s2", "utmos": 4.0},
            {"id": "s3", "utmos": 3.5},
            {"id": "s4", "utmos": 3.0},
        ]
        result_a = _make_result("ModelA", samples_a, summary={"utmos": {"mean": 3.5}})
        result_b = _make_result("ModelB", samples_b, summary={"utmos": {"mean": 3.375}})

        wr = _compute_win_rates(result_a, result_b)

        assert wr["utmos"]["model_a_wins"] == 2
        assert wr["utmos"]["model_b_wins"] == 2
        assert wr["utmos"]["ties"] == 0
        assert wr["utmos"]["model_a_win_rate"] == 0.5

    def test_compute_win_rates_skips_prosody(self):
        """prosody (None in _HIGHER_IS_BETTER) should be skipped entirely."""
        samples_a = [
            {"id": "s1", "prosody": 0.8, "utmos": 4.0},
        ]
        samples_b = [
            {"id": "s1", "prosody": 0.6, "utmos": 3.0},
        ]
        result_a = _make_result(
            "ModelA", samples_a,
            summary={"prosody": {"mean": 0.8}, "utmos": {"mean": 4.0}},
        )
        result_b = _make_result(
            "ModelB", samples_b,
            summary={"prosody": {"mean": 0.6}, "utmos": {"mean": 3.0}},
        )

        wr = _compute_win_rates(result_a, result_b)

        assert "prosody" not in wr
        assert "utmos" in wr

    def test_compute_win_rates_no_common_samples(self):
        """Disjoint sample ids, all totals=0."""
        samples_a = [{"id": "a1", "utmos": 4.0}]
        samples_b = [{"id": "b1", "utmos": 3.0}]
        result_a = _make_result("ModelA", samples_a, summary={"utmos": {"mean": 4.0}})
        result_b = _make_result("ModelB", samples_b, summary={"utmos": {"mean": 3.0}})

        wr = _compute_win_rates(result_a, result_b)

        assert wr["utmos"]["total"] == 0
        assert wr["utmos"]["model_a_wins"] == 0
        assert wr["utmos"]["model_b_wins"] == 0
        assert wr["utmos"]["model_a_win_rate"] == 0.0


class TestGenerateAbReportMarkdown:
    def _make_ab_result(self):
        samples_a = [
            {"id": "s1", "utmos": 4.5, "wer": 0.05},
            {"id": "s2", "utmos": 4.0, "wer": 0.10},
        ]
        samples_b = [
            {"id": "s1", "utmos": 3.0, "wer": 0.15},
            {"id": "s2", "utmos": 3.5, "wer": 0.20},
        ]
        result_a = _make_result(
            "FastSpeech2", samples_a,
            summary={
                "utmos": {"mean": 4.25},
                "wer": {"mean": 0.075},
            },
        )
        result_b = _make_result(
            "VITS", samples_b,
            summary={
                "utmos": {"mean": 3.25},
                "wer": {"mean": 0.175},
            },
        )
        win_rates = _compute_win_rates(result_a, result_b)
        return {
            "model_a": result_a,
            "model_b": result_b,
            "win_rates": win_rates,
        }

    def test_generate_ab_report_creates_file(self, tmp_path):
        """Verify file is created and contains expected sections."""
        ab_result = self._make_ab_result()
        output_path = tmp_path / "report.md"

        result_path = generate_ab_report_markdown(ab_result, output_path)

        assert result_path.exists()
        content = result_path.read_text()
        assert "# A/B Test Report" in content
        assert "## Summary Comparison" in content
        assert "## Win Rates" in content

    def test_generate_ab_report_content(self, tmp_path):
        """Verify Win Rates section and model names in output."""
        ab_result = self._make_ab_result()
        output_path = tmp_path / "report.md"

        generate_ab_report_markdown(ab_result, output_path)
        content = output_path.read_text()

        assert "FastSpeech2" in content
        assert "VITS" in content
        assert "Win Rates" in content
        # Check that win rate table has metric rows
        assert "utmos" in content
        assert "wer" in content
