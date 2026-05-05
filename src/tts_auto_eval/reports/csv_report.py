"""CSV 리포트 생성."""

import csv
from pathlib import Path


def generate_csv_report(eval_result: dict, output_dir: str | Path) -> list[Path]:
    """CSV 리포트 생성.

    Returns:
        생성된 CSV 파일 경로 리스트
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []

    # 1. 요약 CSV
    summary_path = output_dir / "summary.csv"
    summary = eval_result["summary"]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "key", "value"])
        for metric_name, values in summary.items():
            for key, value in values.items():
                writer.writerow([metric_name, key, value])
    created.append(summary_path)

    # 2. 문장별 상세 CSV
    samples = eval_result["samples"]
    if samples:
        detail_path = output_dir / "per_sentence.csv"

        # 컬럼 결정: 기본 필드 + 각 지표 점수
        base_fields = ["id", "text", "category"]
        metric_fields = []
        for key in samples[0]:
            if key not in base_fields and not key.endswith("_details") and key != "tags":
                metric_fields.append(key)

        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(base_fields + metric_fields)
            for sample in samples:
                row = [sample.get(k, "") for k in base_fields]
                row += [sample.get(k, "") for k in metric_fields]
                writer.writerow(row)
        created.append(detail_path)

    return created
