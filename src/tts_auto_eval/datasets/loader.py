"""테스트 데이터셋 로더."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class TestItem(BaseModel):
    """단일 테스트 항목."""

    id: str
    text: str
    expected_spoken: str | None = None
    tags: list[str] = []
    difficulty: str = "medium"


class TestDataset(BaseModel):
    """테스트 데이터셋."""

    language: str
    category: str
    items: list[TestItem]


def load_dataset(path: str | Path) -> TestDataset:
    """JSON 파일에서 데이터셋 로드."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return TestDataset(**data)


def load_datasets_from_dir(
    dataset_dir: str | Path,
    categories: list[str] | None = None,
    max_samples: int | None = None,
) -> list[TestDataset]:
    """디렉토리에서 모든 데이터셋 로드.

    Args:
        dataset_dir: 데이터셋 디렉토리 경로
        categories: 로드할 카테고리 (None이면 전부)
        max_samples: 카테고리당 최대 샘플 수
    """
    dataset_dir = Path(dataset_dir)
    datasets = []

    for json_file in sorted(dataset_dir.glob("*.json")):
        if json_file.name == "schema.json":
            continue

        dataset = load_dataset(json_file)

        if categories and dataset.category not in categories:
            continue

        if max_samples and len(dataset.items) > max_samples:
            dataset.items = dataset.items[:max_samples]

        datasets.append(dataset)

    return datasets
