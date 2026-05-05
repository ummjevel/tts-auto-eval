"""마크다운 리포트 생성."""

from pathlib import Path


def generate_markdown_report(
    eval_result: dict,
    output_path: str | Path,
    compare_result: dict | None = None,
) -> Path:
    """마크다운 요약 리포트 생성."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = eval_result["metadata"]
    summary = eval_result["summary"]
    lines = []

    # 헤더
    lines.append(f"# TTS Evaluation Report")
    lines.append("")
    lines.append(f"- **Model**: {metadata['model_name']}")
    lines.append(f"- **Language**: {metadata['language']}")
    lines.append(f"- **Samples**: {metadata['total_samples']}")
    lines.append(f"- **Date**: {metadata['timestamp'][:19]}")
    lines.append(f"- **Metrics**: {', '.join(metadata['metrics'])}")
    lines.append("")

    # 요약 테이블
    lines.append("## Summary")
    lines.append("")

    if "utmos" in summary:
        s = summary["utmos"]
        lines.append(f"### UTMOS (MOS Prediction)")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Mean | {s['mean']:.2f} |")
        lines.append(f"| Std | {s['std']:.2f} |")
        lines.append(f"| Min | {s['min']:.2f} |")
        lines.append(f"| Max | {s['max']:.2f} |")
        lines.append("")

    if "wer" in summary:
        s = summary["wer"]
        lines.append(f"### WER / CER")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Sentence WER | {s['sentence_wer']*100:.1f}% |")
        lines.append(f"| Global WER | {s['global_wer']*100:.1f}% |")
        lines.append(f"| Sentence CER | {s['sentence_cer']*100:.1f}% |")
        lines.append("")

    if "speaker_similarity" in summary:
        s = summary["speaker_similarity"]
        lines.append(f"### Speaker Similarity")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Mean Cosine Sim | {s['mean']:.4f} |")
        lines.append(f"| Std | {s['std']:.4f} |")
        lines.append("")

    if "prosody" in summary:
        s = summary["prosody"]
        lines.append(f"### Prosody")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| F0 Mean (avg) | {s.get('f0_mean_avg', 0):.1f} Hz |")
        lines.append(f"| F0 Std (avg) | {s.get('f0_std_avg', 0):.1f} Hz |")
        lines.append(f"| Speaking Rate (avg) | {s.get('speaking_rate_avg', 0):.1f} syl/s |")
        if "f0_rmse_avg" in s:
            lines.append(f"| F0 RMSE (avg) | {s['f0_rmse_avg']:.1f} Hz |")
        lines.append("")

    if "pesq" in summary:
        s = summary["pesq"]
        lines.append(f"### PESQ")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Mean | {s['mean']:.2f} |")
        lines.append("")

    if "stoi" in summary:
        s = summary["stoi"]
        lines.append(f"### STOI")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Mean | {s['mean']:.4f} |")
        lines.append("")

    if "itn" in summary:
        s = summary["itn"]
        lines.append(f"### ITN (Inverse Text Normalization)")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Overall WER | {s['overall_wer']*100:.1f}% |")
        lines.append(f"| Overall CER | {s['overall_cer']*100:.1f}% |")
        lines.append("")

        if "categories" in s and s["categories"]:
            lines.append(f"#### Category Breakdown")
            lines.append(f"| Category | WER | Samples |")
            lines.append(f"|----------|-----|---------|")
            for cat, vals in sorted(s["categories"].items()):
                lines.append(f"| {cat} | {vals['wer']*100:.1f}% | {vals['count']} |")
            lines.append("")

    # 비교 테이블
    if compare_result:
        prev_summary = compare_result.get("summary", {})
        lines.append("## Comparison with Previous Report")
        lines.append("")
        lines.append(f"Previous model: **{compare_result.get('metadata', {}).get('model_name', 'unknown')}**")
        lines.append("")
        lines.append("| Metric | Previous | Current | Diff |")
        lines.append("|--------|----------|---------|------|")

        comparisons = [
            ("UTMOS", "utmos", "mean", True),
            ("Sentence WER", "wer", "sentence_wer", False),
            ("Sentence CER", "wer", "sentence_cer", False),
        ]
        if "speaker_similarity" in summary:
            comparisons.append(("Speaker Sim", "speaker_similarity", "mean", True))
        if "itn" in summary:
            comparisons.append(("ITN WER", "itn", "overall_wer", False))

        for label, key, field, higher_better in comparisons:
            if key in summary and key in prev_summary:
                curr = summary[key].get(field, 0)
                prev = prev_summary[key].get(field, 0)
                diff = curr - prev
                arrow = "+" if diff > 0 else ""
                lines.append(f"| {label} | {prev:.4f} | {curr:.4f} | {arrow}{diff:.4f} |")

        lines.append("")

    # 샘플 상위/하위
    samples = eval_result.get("samples", [])
    if samples and "utmos" in summary:
        lines.append("## Sample Details (Top/Bottom 5 by UTMOS)")
        lines.append("")

        scored = [s for s in samples if "utmos" in s]
        scored.sort(key=lambda x: x.get("utmos", 0), reverse=True)

        lines.append("### Top 5")
        lines.append("| ID | Text | UTMOS |")
        lines.append("|----|------|-------|")
        for s in scored[:5]:
            lines.append(f"| {s['id']} | {s['text'][:50]}... | {s.get('utmos', 0):.2f} |")
        lines.append("")

        lines.append("### Bottom 5")
        lines.append("| ID | Text | UTMOS |")
        lines.append("|----|------|-------|")
        for s in scored[-5:]:
            lines.append(f"| {s['id']} | {s['text'][:50]}... | {s.get('utmos', 0):.2f} |")
        lines.append("")

    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")
    return output_path
