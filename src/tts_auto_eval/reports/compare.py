"""비교 리포트 생성."""

import json
from pathlib import Path


def load_result(path: str | Path) -> dict:
    """이전 평가 결과 JSON 로드."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
