"""A/B 테스트 로직."""

from datetime import datetime, timezone
from pathlib import Path

from tts_auto_eval.config import Config
from tts_auto_eval.pipeline import run_evaluation

_HIGHER_IS_BETTER = {
    "utmos": True,
    "wer": False,
    "cer": False,
    "pesq": True,
    "stoi": True,
    "speaker_similarity": True,
    "itn": False,
    "fad": False,
    "prosody": None,  # not comparable
}


def run_ab_test(config_a: Config, config_b: Config) -> dict:
    """Run evaluation for both models and compare results.

    Uses run_evaluation() from pipeline.py for each model.
    Returns combined result with both results + win rates.
    """
    result_a = run_evaluation(config_a)
    result_b = run_evaluation(config_b)

    win_rates = _compute_win_rates(result_a, result_b)

    return {
        "model_a": result_a,
        "model_b": result_b,
        "win_rates": win_rates,
    }


def _compute_win_rates(result_a: dict, result_b: dict) -> dict:
    """Compare per-sample scores between two results.

    For each metric, iterate over samples matched by 'id'.
    Count wins for model_a, model_b, and ties.
    Use _HIGHER_IS_BETTER to determine which direction is better.
    Skip metrics with None in _HIGHER_IS_BETTER.
    """
    samples_a = {s["id"]: s for s in result_a["samples"]}
    samples_b = {s["id"]: s for s in result_b["samples"]}
    common_ids = sorted(set(samples_a.keys()) & set(samples_b.keys()))

    # Collect all metric names from both results
    metrics = set(result_a.get("summary", {}).keys()) | set(result_b.get("summary", {}).keys())

    win_rates = {}
    for metric in metrics:
        higher = _HIGHER_IS_BETTER.get(metric)
        if higher is None:
            continue

        a_wins = 0
        b_wins = 0
        ties = 0

        for sid in common_ids:
            score_a = samples_a[sid].get(metric)
            score_b = samples_b[sid].get(metric)

            if score_a is None or score_b is None:
                continue

            if score_a == score_b:
                ties += 1
            elif higher:
                if score_a > score_b:
                    a_wins += 1
                else:
                    b_wins += 1
            else:
                if score_a < score_b:
                    a_wins += 1
                else:
                    b_wins += 1

        total = a_wins + b_wins + ties
        win_rates[metric] = {
            "model_a_wins": a_wins,
            "model_b_wins": b_wins,
            "ties": ties,
            "model_a_win_rate": a_wins / total if total > 0 else 0.0,
            "total": total,
        }

    return win_rates


def generate_ab_report_markdown(ab_result: dict, output_path: Path) -> Path:
    """Generate markdown report with comparison tables."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_a = ab_result["model_a"]
    result_b = ab_result["model_b"]
    win_rates = ab_result["win_rates"]

    model_a_name = result_a["metadata"].get("model_name", "Model A")
    model_b_name = result_b["metadata"].get("model_name", "Model B")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# A/B Test Report",
        f"",
        f"- **Model A**: {model_a_name}",
        f"- **Model B**: {model_b_name}",
        f"- **Date**: {timestamp}",
        f"- **Samples**: {result_a['metadata'].get('total_samples', 'N/A')} (A) / {result_b['metadata'].get('total_samples', 'N/A')} (B)",
        f"",
        f"## Summary Comparison",
        f"",
        f"| Metric | {model_a_name} | {model_b_name} | Diff |",
        f"|--------|---------|---------|------|",
    ]

    summary_a = result_a.get("summary", {})
    summary_b = result_b.get("summary", {})
    all_metrics = sorted(set(summary_a.keys()) | set(summary_b.keys()))

    for metric in all_metrics:
        val_a = summary_a.get(metric, {}).get("mean")
        val_b = summary_b.get(metric, {}).get("mean")
        if val_a is not None and val_b is not None:
            diff = val_b - val_a
            sign = "+" if diff > 0 else ""
            lines.append(f"| {metric} | {val_a:.4f} | {val_b:.4f} | {sign}{diff:.4f} |")
        else:
            str_a = f"{val_a:.4f}" if val_a is not None else "N/A"
            str_b = f"{val_b:.4f}" if val_b is not None else "N/A"
            lines.append(f"| {metric} | {str_a} | {str_b} | - |")

    # Win rate table
    lines.extend([
        f"",
        f"## Win Rates",
        f"",
        f"| Metric | {model_a_name} Wins | {model_b_name} Wins | Ties | {model_a_name} Win Rate |",
        f"|--------|------------|------------|------|--------------|",
    ])

    for metric in sorted(win_rates.keys()):
        wr = win_rates[metric]
        lines.append(
            f"| {metric} | {wr['model_a_wins']} | {wr['model_b_wins']} | {wr['ties']} | {wr['model_a_win_rate']:.1%} |"
        )

    # Per-sample detail: top 5 biggest differences
    lines.extend([
        f"",
        f"## Per-Sample Details (Top 5 Biggest Differences)",
        f"",
    ])

    samples_a = {s["id"]: s for s in result_a["samples"]}
    samples_b = {s["id"]: s for s in result_b["samples"]}
    common_ids = sorted(set(samples_a.keys()) & set(samples_b.keys()))

    comparable_metrics = [m for m in all_metrics if _HIGHER_IS_BETTER.get(m) is not None]

    if comparable_metrics and common_ids:
        # Compute total absolute diff per sample
        sample_diffs = []
        for sid in common_ids:
            total_diff = 0.0
            metric_details = {}
            for metric in comparable_metrics:
                sa = samples_a[sid].get(metric)
                sb = samples_b[sid].get(metric)
                if sa is not None and sb is not None:
                    d = abs(sb - sa)
                    total_diff += d
                    metric_details[metric] = (sa, sb, sb - sa)
            sample_diffs.append((sid, total_diff, metric_details))

        sample_diffs.sort(key=lambda x: x[1], reverse=True)
        top5 = sample_diffs[:5]

        header_metrics = comparable_metrics[:6]  # limit columns
        lines.append(
            f"| ID | " + " | ".join(f"{m} (A/B)" for m in header_metrics) + " |"
        )
        lines.append(
            f"|----| " + " | ".join("---" for _ in header_metrics) + " |"
        )

        for sid, _, details in top5:
            cols = []
            for m in header_metrics:
                if m in details:
                    sa, sb, _ = details[m]
                    cols.append(f"{sa:.3f} / {sb:.3f}")
                else:
                    cols.append("N/A")
            lines.append(f"| {sid} | " + " | ".join(cols) + " |")
    else:
        lines.append("No comparable per-sample data available.")

    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
