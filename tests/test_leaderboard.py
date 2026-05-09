"""Tests for tts_auto_eval.leaderboard module."""

import json
from pathlib import Path

import pytest

from tts_auto_eval.leaderboard import (
    _extract_score,
    _rank_scores,
    build_leaderboard,
    generate_leaderboard_markdown,
    load_results,
)


def _make_result(model_name, utmos_mean=None, wer_sentence=None):
    summary = {}
    if utmos_mean is not None:
        summary["utmos"] = {"mean": utmos_mean, "std": 0.1, "min": 3.0, "max": 5.0, "count": 10}
    if wer_sentence is not None:
        summary["wer"] = {"sentence_wer": wer_sentence, "global_wer": wer_sentence, "count": 10}
    return {
        "metadata": {
            "model_name": model_name,
            "language": "ko",
            "metrics": list(summary.keys()),
            "total_samples": 10,
        },
        "summary": summary,
        "samples": [],
    }


class TestExtractScore:
    def test_extract_score_utmos(self):
        summary = {"utmos": {"mean": 4.12}}
        assert _extract_score(summary, "utmos") == 4.12

    def test_extract_score_wer(self):
        summary = {"wer": {"sentence_wer": 0.05}}
        assert _extract_score(summary, "wer") == 0.05

    def test_extract_score_missing(self):
        assert _extract_score({}, "utmos") is None


class TestRankScores:
    def test_rank_scores_higher_better(self):
        scores = [4.0, 3.0, 5.0]
        ranks = _rank_scores(scores, higher_is_better=True)
        assert ranks == [2, 3, 1]

    def test_rank_scores_lower_better(self):
        scores = [0.1, 0.3, 0.05]
        ranks = _rank_scores(scores, higher_is_better=False)
        assert ranks == [2, 3, 1]

    def test_rank_scores_with_none(self):
        scores = [4.0, None, 3.0]
        ranks = _rank_scores(scores, higher_is_better=True)
        assert ranks == [1, None, 2]


class TestBuildLeaderboard:
    def test_build_leaderboard_basic(self):
        """3 models with utmos+wer, verify ranks and overall_ranks."""
        results = [
            _make_result("ModelA", utmos_mean=4.0, wer_sentence=0.10),
            _make_result("ModelB", utmos_mean=3.5, wer_sentence=0.05),
            _make_result("ModelC", utmos_mean=4.5, wer_sentence=0.15),
        ]
        lb = build_leaderboard(results)

        assert lb["models"] == ["ModelA", "ModelB", "ModelC"]
        assert "utmos" in lb["metrics"]
        assert "wer" in lb["metrics"]

        # utmos: C(4.5)=1, A(4.0)=2, B(3.5)=3
        assert lb["metrics"]["utmos"]["ranks"] == [2, 3, 1]
        # wer (lower=better): B(0.05)=1, A(0.10)=2, C(0.15)=3
        assert lb["metrics"]["wer"]["ranks"] == [2, 1, 3]

        # overall_ranks: A=(2+2)/2=2.0, B=(3+1)/2=2.0, C=(1+3)/2=2.0
        assert lb["overall_ranks"] == [2.0, 2.0, 2.0]

    def test_build_leaderboard_single_model(self):
        """Single model should not crash."""
        results = [_make_result("OnlyModel", utmos_mean=4.0, wer_sentence=0.10)]
        lb = build_leaderboard(results)

        assert lb["models"] == ["OnlyModel"]
        assert lb["metadata"]["total_models"] == 1


class TestGenerateLeaderboardMarkdown:
    def test_generate_leaderboard_markdown(self, tmp_path):
        """Verify file created, contains header and arrows."""
        results = [
            _make_result("ModelA", utmos_mean=4.0, wer_sentence=0.10),
            _make_result("ModelB", utmos_mean=3.5, wer_sentence=0.05),
        ]
        lb = build_leaderboard(results)
        output_path = tmp_path / "leaderboard.md"

        result_path = generate_leaderboard_markdown(lb, output_path)

        assert result_path.exists()
        content = result_path.read_text()
        assert "리더보드" in content
        assert "\u2191" in content  # up arrow
        assert "\u2193" in content  # down arrow
        assert "ModelA" in content
        assert "ModelB" in content


class TestLoadResults:
    def test_load_results(self, tmp_path):
        """Write 2 JSON files to tmp_path, load them, verify count."""
        r1 = _make_result("ModelA", utmos_mean=4.0)
        r2 = _make_result("ModelB", utmos_mean=3.5)

        p1 = tmp_path / "result1.json"
        p2 = tmp_path / "result2.json"
        p1.write_text(json.dumps(r1))
        p2.write_text(json.dumps(r2))

        loaded = load_results([p1, p2])

        assert len(loaded) == 2
        assert loaded[0]["metadata"]["model_name"] == "ModelA"
        assert loaded[1]["metadata"]["model_name"] == "ModelB"
