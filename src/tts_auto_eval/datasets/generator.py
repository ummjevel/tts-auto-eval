"""평가 데이터셋 자동 생성 — LLM 기반 도메인별 테스트 문장 생성."""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DOMAIN_PROMPTS = {
    "general": "일상 대화에서 사용되는 자연스러운 문장",
    "medical": "의료 용어와 진료 관련 문장 (예: 진단명, 처방, 환자 안내)",
    "legal": "법률 문서와 계약서에서 사용되는 문장 (예: 조항, 판결문)",
    "broadcast": "뉴스 방송 스크립트 (예: 보도, 날씨, 스포츠)",
    "customer_service": "고객센터 응대 문장 (예: 안내, 사과, 확인)",
    "finance": "금융 관련 문장 (예: 주가, 환율, 투자 안내)",
}

ITN_CATEGORIES = {
    "numbers": "숫자와 수량 표현 (예: 123개, 4,500원)",
    "dates": "날짜와 시간 표현 (예: 2024년 3월 15일, 오후 2시 30분)",
    "currency": "통화와 금액 표현 (예: $100, 52,000원, 3.5%)",
    "phone": "전화번호 표현 (예: 010-1234-5678, 02-123-4567)",
    "address": "주소와 우편번호 표현",
}


class DatasetGenerator:
    """LLM 기반 테스트 데이터셋 생성기."""

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model

    def _call_llm(self, prompt: str, count: int) -> list[str]:
        """LLM API 호출로 문장 생성."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self._api_key)

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 TTS 평가용 테스트 문장을 생성하는 전문가입니다. "
                            "다양하고 자연스러운 문장을 생성하세요."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{prompt}\n\n{count}개의 문장을 생성하세요. "
                            "한 줄에 하나씩, 번호 없이 문장만 출력하세요."
                        ),
                    },
                ],
                temperature=0.8,
            )

            text = response.choices[0].message.content or ""
            sentences = [s.strip() for s in text.strip().split("\n") if s.strip()]
            return sentences[:count]
        except ImportError:
            logger.error("openai package not installed. Install with: pip install openai")
            return self._generate_fallback(count)
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return self._generate_fallback(count)

    def _generate_fallback(self, count: int) -> list[str]:
        """LLM 사용 불가 시 기본 문장 생성."""
        templates = [
            "안녕하세요, 오늘 날씨가 좋습니다.",
            "이 제품의 가격은 얼마인가요?",
            "회의는 오후 세 시에 시작합니다.",
            "감사합니다, 좋은 하루 되세요.",
            "주문하신 상품이 배송 중입니다.",
        ]
        result = []
        for i in range(count):
            result.append(templates[i % len(templates)])
        return result

    def generate(
        self,
        domain: str = "general",
        language: str = "ko",
        count: int = 20,
        include_itn: bool = False,
    ) -> dict:
        """데이터셋 생성.

        Returns:
            TestDataset 형식의 dict: {language, category, items: [{id, text, ...}]}
        """
        lang_name = "한국어" if language == "ko" else "English"
        domain_desc = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])

        prompt = f"{lang_name}로 {domain_desc}을(를) 포함하는 TTS 평가용 문장을 생성하세요."
        sentences = self._call_llm(prompt, count)

        items = []
        for i, text in enumerate(sentences):
            item = {
                "id": f"{domain}_{i+1:03d}",
                "text": text,
                "tags": [domain],
                "difficulty": "medium",
            }
            items.append(item)

        # ITN 케이스 추가
        if include_itn:
            itn_items = self._generate_itn_cases(language, count=min(count, 20))
            items.extend(itn_items)

        return {
            "language": language,
            "category": domain,
            "items": items,
        }

    def _generate_itn_cases(self, language: str, count: int = 20) -> list[dict]:
        """ITN 테스트 케이스 생성."""
        lang_name = "한국어" if language == "ko" else "English"
        items = []

        for category, desc in ITN_CATEGORIES.items():
            cat_count = max(1, count // len(ITN_CATEGORIES))
            prompt = (
                f"{lang_name}로 {desc}이(가) 포함된 문장을 생성하세요. "
                "숫자/기호를 written form으로 작성하세요."
            )
            sentences = self._call_llm(prompt, cat_count)

            for j, text in enumerate(sentences):
                items.append(
                    {
                        "id": f"itn_{category}_{j+1:03d}",
                        "text": text,
                        "tags": ["itn", category],
                        "difficulty": "medium",
                    }
                )

        return items

    def save(self, dataset: dict, output_path: str | Path) -> Path:
        """데이터셋을 JSON 파일로 저장."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset saved: {path} ({len(dataset.get('items', []))} items)")
        return path
