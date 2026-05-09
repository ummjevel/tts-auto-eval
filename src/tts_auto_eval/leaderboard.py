"""벤치마크 리더보드 생성.

여러 모델의 result.json을 누적하여 자동 랭킹 테이블을 생성한다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# 지표별 방향 (True=높을수록 좋음, False=낮을수록 좋음)
_HIGHER_IS_BETTER = {
    "utmos": True,
    "wer": False,
    "cer": False,
    "pesq": True,
    "stoi": True,
    "speaker_similarity": True,
    "itn": False,
    "fad": False,
}

# 각 지표의 summary에서 대표값 추출 키
_SCORE_KEYS = {
    "utmos": "mean",
    "wer": "sentence_wer",
    "pesq": "mean",
    "stoi": "mean",
    "speaker_similarity": "mean",
    "itn": "overall_wer",
    "fad": "fad_score",
    "prosody": "mean",
}


def load_results(result_paths: list[Path]) -> list[dict]:
    """여러 result.json 파일 로드."""
    results: list[dict] = []
    for path in result_paths:
        with open(path) as f:
            results.append(json.load(f))
    return results


def _extract_score(summary: dict, metric_name: str) -> float | None:
    """summary dict에서 지표의 대표 점수 추출.

    Uses _SCORE_KEYS to find the right key.
    Returns None if metric not found.
    """
    metric_summary = summary.get(metric_name)
    if metric_summary is None:
        return None

    key = _SCORE_KEYS.get(metric_name, "mean")
    value = metric_summary.get(key)
    if value is None:
        return None

    return float(value)


def _rank_scores(scores: list[float | None], higher_is_better: bool) -> list[int | None]:
    """점수 리스트에 순위를 매긴다. None인 항목은 None 순위."""
    indexed = [(i, s) for i, s in enumerate(scores) if s is not None]
    indexed.sort(key=lambda x: x[1], reverse=higher_is_better)

    ranks: list[int | None] = [None] * len(scores)
    for rank, (idx, _) in enumerate(indexed, start=1):
        ranks[idx] = rank
    return ranks


def build_leaderboard(results: list[dict]) -> dict:
    """여러 평가 결과에서 랭킹 테이블 생성."""
    models: list[str] = []
    summaries: list[dict] = []

    for r in results:
        model_name = r.get("metadata", {}).get("model_name", "unknown")
        models.append(model_name)
        summaries.append(r.get("summary", {}))

    # 공통 metric 파악: 2개 이상 모델에 존재하는 metric
    metric_counts: dict[str, int] = {}
    for s in summaries:
        for m in s:
            metric_counts[m] = metric_counts.get(m, 0) + 1

    common_metrics = [m for m, cnt in metric_counts.items() if cnt >= 2 or len(results) == 1]

    # 각 metric별 점수 추출 및 랭킹
    metrics_data: dict[str, dict] = {}
    for metric in common_metrics:
        scores = [_extract_score(s, metric) for s in summaries]
        higher = _HIGHER_IS_BETTER.get(metric, True)
        ranks = _rank_scores(scores, higher)
        metrics_data[metric] = {
            "scores": scores,
            "ranks": ranks,
        }

    # 종합 순위 = 각 지표 순위의 평균 (None 순위는 제외)
    overall_ranks: list[float] = []
    for i in range(len(models)):
        valid_ranks = [
            metrics_data[m]["ranks"][i]
            for m in metrics_data
            if metrics_data[m]["ranks"][i] is not None
        ]
        if valid_ranks:
            overall_ranks.append(sum(valid_ranks) / len(valid_ranks))
        else:
            overall_ranks.append(float("inf"))

    return {
        "models": models,
        "metrics": metrics_data,
        "overall_ranks": overall_ranks,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_models": len(models),
            "total_metrics": len(metrics_data),
        },
    }


def generate_leaderboard_markdown(leaderboard: dict, output_path: Path) -> Path:
    """마크다운 리더보드 테이블 생성."""
    models = leaderboard["models"]
    metrics = leaderboard["metrics"]
    overall_ranks = leaderboard["overall_ranks"]
    metric_names = list(metrics.keys())

    # 종합 순위 기준 정렬 인덱스
    sorted_indices = sorted(range(len(models)), key=lambda i: overall_ranks[i])

    # 헤더 생성
    header_cols = ["순위", "모델"]
    for m in metric_names:
        arrow = "↑" if _HIGHER_IS_BETTER.get(m, True) else "↓"
        header_cols.append(f"{m.upper()}{arrow}")
    header_cols.append("종합")

    separator = ["-" * max(len(c), 4) for c in header_cols]

    lines: list[str] = [
        "# TTS 벤치마크 리더보드",
        "",
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(separator) + " |",
    ]

    for rank_pos, idx in enumerate(sorted_indices, start=1):
        row = [str(rank_pos), models[idx]]
        for m in metric_names:
            score = metrics[m]["scores"][idx]
            r = metrics[m]["ranks"][idx]
            if score is not None and r is not None:
                row.append(f"{score:.2f} ({r})")
            else:
                row.append("-")
        row.append(f"{overall_ranks[idx]:.1f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(f"_생성 시각: {leaderboard['metadata']['generated_at']}_")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
