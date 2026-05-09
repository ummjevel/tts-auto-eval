"""데이터셋 생성기 테스트."""

import json
from pathlib import Path

from tts_auto_eval.datasets.generator import (
    DOMAIN_PROMPTS,
    ITN_CATEGORIES,
    DatasetGenerator,
)


def test_generate_fallback():
    """OpenAI 없이 fallback 문장 생성 테스트."""
    gen = DatasetGenerator(api_key="fake")
    sentences = gen._generate_fallback(7)
    assert len(sentences) == 7
    assert all(isinstance(s, str) for s in sentences)
    assert all(len(s) > 0 for s in sentences)


def test_generate_output_format():
    """generate 반환 dict 구조 검증."""
    gen = DatasetGenerator(api_key="fake")
    # _call_llm will fail (no real API key) and fall back to fallback
    dataset = gen.generate(domain="general", language="ko", count=5)

    assert "language" in dataset
    assert "category" in dataset
    assert "items" in dataset
    assert dataset["language"] == "ko"
    assert dataset["category"] == "general"
    assert len(dataset["items"]) == 5

    item = dataset["items"][0]
    assert "id" in item
    assert "text" in item
    assert "tags" in item
    assert "difficulty" in item


def test_save_dataset(tmp_path: Path):
    """데이터셋 JSON 저장 테스트."""
    gen = DatasetGenerator(api_key="fake")
    dataset = {
        "language": "ko",
        "category": "general",
        "items": [{"id": "test_001", "text": "테스트 문장입니다."}],
    }
    output_path = tmp_path / "sub" / "dataset.json"
    result_path = gen.save(dataset, output_path)

    assert result_path.exists()
    with open(result_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["language"] == "ko"
    assert len(loaded["items"]) == 1
    assert loaded["items"][0]["text"] == "테스트 문장입니다."


def test_domain_prompts_exist():
    """모든 도메인 키 존재 확인."""
    expected = {"general", "medical", "legal", "broadcast", "customer_service", "finance"}
    assert set(DOMAIN_PROMPTS.keys()) == expected


def test_itn_categories_exist():
    """ITN 카테고리 키 존재 확인."""
    expected = {"numbers", "dates", "currency", "phone", "address"}
    assert set(ITN_CATEGORIES.keys()) == expected
