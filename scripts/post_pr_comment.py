"""평가 결과를 PR 코멘트로 게시."""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    pr_number = os.environ.get("PR_NUMBER")
    token = os.environ.get("GITHUB_TOKEN")

    if not pr_number or not token:
        print("PR_NUMBER 또는 GITHUB_TOKEN이 설정되지 않았습니다.")
        sys.exit(1)

    # 결과 파일 찾기
    result_path = Path("./eval_results/result.json")
    md_path = Path("./eval_results/report.md")

    if md_path.exists():
        # 마크다운 리포트가 있으면 그대로 사용
        report_content = md_path.read_text(encoding="utf-8")
    elif result_path.exists():
        # result.json에서 요약 생성
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
        report_content = _build_summary(result)
    else:
        print("평가 결과 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # 코멘트 본문 구성
    comment_body = f"## 🔊 TTS 평가 결과\n\n{report_content}\n\n---\n*자동 생성: tts-auto-eval*"

    # gh CLI로 코멘트 게시
    subprocess.run(
        ["gh", "pr", "comment", str(pr_number), "--body", comment_body],
        check=True,
        env={**os.environ, "GH_TOKEN": token},
    )
    print(f"PR #{pr_number}에 평가 리포트 코멘트 게시 완료")


def _build_summary(result: dict) -> str:
    """result.json에서 마크다운 요약 생성."""
    metadata = result.get("metadata", {})
    summary = result.get("summary", {})

    lines = [
        f"**모델**: {metadata.get('model_name', 'N/A')}",
        f"**언어**: {metadata.get('language', 'N/A')}",
        f"**샘플 수**: {metadata.get('total_samples', 0)}",
        f"**평가 시각**: {metadata.get('timestamp', 'N/A')}",
        "",
        "### 지표 요약",
        "",
        "| 지표 | 값 |",
        "|------|-----|",
    ]

    for metric_name, metric_data in summary.items():
        if isinstance(metric_data, dict):
            # Extract main score (first numeric value or specific keys)
            if "sentence_wer" in metric_data:
                lines.append(f"| WER | {metric_data['sentence_wer']:.4f} |")
            elif "overall_wer" in metric_data:
                lines.append(f"| ITN WER | {metric_data['overall_wer']:.4f} |")
            elif "mean" in metric_data:
                lines.append(f"| {metric_name} | {metric_data['mean']:.4f} |")
            else:
                # Get first numeric value
                for k, v in metric_data.items():
                    if isinstance(v, (int, float)) and k != "count":
                        lines.append(f"| {metric_name} ({k}) | {v:.4f} |")
                        break

    return "\n".join(lines)


if __name__ == "__main__":
    main()
