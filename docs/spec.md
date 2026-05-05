# TTS Auto Eval - Specification

## 1. Overview

TTS 모델의 출력을 자동으로 평가하고, 정량적 리포트를 생성하는 Python CLI 도구.

- TTS 모델을 직접 연동하여 텍스트 → 음성 생성 → 평가를 원스톱으로 수행
- 다양한 평가 지표를 자동 측정
- HTML 대시보드 / CSV / 마크다운 형태로 리포트 출력
- 기존 리포트와의 비교 표 제공

## 2. 핵심 워크플로우

```
[테스트 텍스트 세트] → [TTS 모델] → [음성 파일] → [평가 파이프라인] → [리포트]
```

### 사용 예시

```bash
# 모델 평가 실행
tts-auto-eval run \
  --model my_model \
  --config config.yaml \
  --output results/

# 기존 리포트와 비교
tts-auto-eval compare results/report_v1.json results/report_v2.json \
  --output comparison.html

# 음성 파일만 평가 (이미 생성된 음성이 있을 때)
tts-auto-eval eval \
  --audio-dir ./generated_audio/ \
  --reference-texts test_set.json \
  --output results/
```

## 3. TTS 모델 연동

### 3.1 어댑터 패턴

모든 TTS 백엔드는 공통 인터페이스를 구현한다.

```python
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class BaseTTSModel(ABC):
    """TTS 모델 어댑터 기본 클래스."""

    @property
    @abstractmethod
    def name(self) -> str:
        """모델 이름 (리포트에 표시)."""
        ...

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        텍스트를 음성으로 변환.

        Returns:
            (audio_array, sample_rate) - numpy 배열과 샘플레이트
        """
        ...

    def synthesize_to_file(self, text: str, path: Path) -> Path:
        """음성을 파일로 저장. 기본 구현 제공."""
        import soundfile as sf
        audio, sr = self.synthesize(text)
        sf.write(str(path), audio, sr)
        return path
```

### 3.2 지원 백엔드 (확장 가능)

| 백엔드 | 유형 | 비고 |
|--------|------|------|
| F5-TTS | 로컬 오픈소스 | `pip install f5-tts` |
| CosyVoice | 로컬 오픈소스 | repo clone 필요 |
| Coqui/XTTS | 로컬 오픈소스 | `pip install TTS` |
| OpenAI TTS | 상용 API | API 키 필요, 유료 |
| Google Cloud TTS | 상용 API | 서비스 계정 필요 |
| Azure TTS | 상용 API | 구독 키 필요 |
| Custom | 사용자 정의 | BaseTTSModel 상속 |

### 3.3 사용자 정의 모델 등록

```python
# custom_model.py
from tts_auto_eval.models.base import BaseTTSModel

class MyCustomTTS(BaseTTSModel):
    @property
    def name(self) -> str:
        return "my-custom-tts-v1"

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        # 사용자의 TTS 모델 호출 로직
        audio = my_model.generate(text)
        return audio, 22050
```

```yaml
# config.yaml
model:
  type: custom
  module: custom_model.MyCustomTTS
  params:
    checkpoint: ./checkpoints/model_v1.pt
```

## 4. 평가 지표

### 4.1 음질 (Quality)

| 지표 | 설명 | 참조 음성 필요 | 라이브러리 |
|------|------|:-----------:|-----------|
| **UTMOS** | MOS 예측 (자연스러움 1-5점) | No | `speechmos` |
| **PESQ** | 지각적 음질 (ITU-T P.862) | Yes | `pesq` |
| **DNSMOS** | DNS 챌린지 기반 음질 예측 | No | Azure API / 로컬 |

### 4.2 명료도 (Intelligibility)

| 지표 | 설명 | 라이브러리 |
|------|------|-----------|
| **WER** | 단어 오류율 (Whisper STT → 원본 비교) | `jiwer` + `openai-whisper` |
| **CER** | 문자 오류율 | `jiwer` |
| **STOI** | 단시간 명료도 지수 | `pystoi` |

**WER 프로토콜 (Seed-TTS 참고):**
- **Sentence-level WER**: 문장별 WER을 계산한 후 평균 (모든 문장에 동일 가중치)
- **Global WER**: 전체 substitutions + deletions + insertions / 전체 단어 수
- 두 가지 모두 리포트에 표시. Seed-TTS 논문 등 최신 논문에서는 Sentence-level WER을 주로 사용

**ASR 백엔드:**
- 기본: Whisper-large-v3 (다국어)
- 설정으로 ASR 백엔드 선택 가능 (향후 Paraformer 등 추가)

### 4.3 화자 유사도 (Speaker Similarity)

| 지표 | 설명 | 라이브러리 |
|------|------|-----------|
| **Speaker Cosine Sim** | ECAPA-TDNN 임베딩 코사인 유사도 | `speechbrain` |

### 4.4 운율 (Prosody)

| 지표 | 설명 | 라이브러리 |
|------|------|-----------|
| **F0 RMSE** | 피치 오차 | `parselmouth` |
| **Energy Correlation** | 에너지 패턴 상관도 | `librosa` |
| **Speaking Rate** | 발화 속도 (음절/초) | `librosa` + `jiwer` |

### 4.5 분포 품질 (Distribution-level)

| 지표 | 설명 | 라이브러리 |
|------|------|-----------|
| **FAD** | Frechet Audio Distance | `fadtk` |

### 4.6 ITN 능력 (Inverse Text Normalization)

숫자, 날짜, 통화 등 비표준 텍스트를 올바르게 발음하는 능력을 측정.

**평가 방법:**
1. ITN 카테고리별 테스트 텍스트 준비 (written form + expected spoken form)
2. TTS로 음성 생성
3. Whisper STT로 전사
4. 기대 발음과 비교하여 카테고리별 WER 산출

**카테고리:**

| 카테고리 | 예시 (한국어) | 기대 발음 |
|----------|-------------|----------|
| 숫자 | "123명" | "백이십삼 명" |
| 날짜 | "2024년 3월 15일" | "이천이십사년 삼월 십오일" |
| 시간 | "오후 3:30" | "오후 세시 삼십분" |
| 통화 | "₩15,000" | "만 오천 원" |
| 전화번호 | "010-1234-5678" | "공일공 일이삼사 오육칠팔" |
| 비율/퍼센트 | "3.14%" | "삼 점 일사 퍼센트" |
| 단위 | "100km/h" | "시속 백 킬로미터" |
| 약어 | "WHO" | "더블유에이치오" |

## 5. 테스트 데이터셋

### 5.1 구조

```
datasets/
├── ko/                     # 한국어
│   ├── general.json        # 일반 문장
│   ├── itn.json            # ITN 테스트 세트
│   └── long_form.json      # 긴 문장/문단
├── en/                     # 영어 (추후)
│   ├── general.json
│   ├── itn.json
│   └── long_form.json
└── schema.json             # 데이터 스키마
```

### 5.2 데이터 형식

```json
{
  "language": "ko",
  "category": "itn",
  "items": [
    {
      "id": "itn_ko_001",
      "text": "총 비용은 ₩1,500,000입니다.",
      "expected_spoken": "총 비용은 백오십만 원입니다.",
      "tags": ["currency"],
      "difficulty": "medium"
    }
  ]
}
```

### 5.3 언어 지원

- **Phase 1**: 한국어 (ko)
- **Phase 2**: 영어 (en)
- **Phase 3**: 기타 언어 (확장 구조 준비)

각 언어별로 Whisper 모델 크기, ITN 규칙, 음절 카운팅 방식을 설정 파일로 분리.

## 6. 리포트

### 6.1 HTML 대시보드

- **요약 카드**: 전체 점수 요약 (UTMOS, WER, Speaker Sim 등)
- **지표별 상세 차트**: 바 차트, 분포 히스토그램
- **ITN 카테고리별 성적표**: 카테고리별 WER 히트맵
- **샘플 오디오 재생**: 각 테스트 문장의 생성 음성을 인라인 재생
- **비교 표**: 기존 리포트 로드 시, 이전 결과와 나란히 비교
- Jinja2 기반 단일 HTML 파일 (외부 의존성 없음)

### 6.2 CSV

- 문장별 상세 결과 (`per_sentence.csv`)
- 지표 요약 (`summary.csv`)
- ITN 카테고리별 결과 (`itn_breakdown.csv`)

### 6.3 마크다운

- 요약 리포트 (`report.md`)
- GitHub/GitLab 등에서 바로 렌더링 가능

### 6.4 비교 리포트

```bash
tts-auto-eval compare report_a.json report_b.json
```

- JSON 형태의 원시 결과 파일을 로드
- 지표별 diff 테이블 생성 (향상/하락 표시)
- HTML, CSV, 마크다운 모두 지원

## 7. 설정 파일 (config.yaml)

```yaml
# 모델 설정
model:
  type: f5-tts                    # 또는 custom, openai, azure 등
  params:
    checkpoint: ./model.pt
    ref_audio: ./reference.wav    # 음성 복제 시

# 평가 설정
evaluation:
  language: ko
  metrics:
    - utmos
    - wer
    - cer
    - speaker_similarity
    - pesq                        # 참조 음성 있을 때만
    - stoi
    - prosody
    - fad
    - itn
  whisper_model: large-v3         # STT 모델 크기
  reference_audio: ./ref.wav      # 화자 유사도 기준 음성

# 데이터셋
dataset:
  path: ./datasets/ko/
  categories:
    - general
    - itn
    - long_form
  max_samples: 100                # 전체 중 샘플링 (선택)

# 리포트
report:
  formats:
    - html
    - csv
    - markdown
  output_dir: ./results/
  compare_with: ./results/previous_report.json  # 비교 대상 (선택)
  include_audio: true             # HTML에 오디오 임베드
```

## 8. 프로젝트 구조

```
tts-auto-eval/
├── src/tts_auto_eval/
│   ├── __init__.py
│   ├── cli.py                    # CLI 엔트리포인트 (click/typer)
│   ├── config.py                 # 설정 파싱
│   ├── pipeline.py               # 메인 파이프라인 오케스트레이션
│   ├── audio.py                  # 오디오 유틸 (load_waveform, 리샘플링, 모노 변환)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseTTSModel ABC
│   │   ├── f5_tts.py
│   │   ├── cosyvoice.py
│   │   ├── coqui.py
│   │   └── openai_tts.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseMetric ABC
│   │   ├── quality.py            # UTMOS, PESQ, DNSMOS
│   │   ├── intelligibility.py    # WER, CER, STOI
│   │   ├── speaker.py            # Speaker Similarity
│   │   ├── prosody.py            # F0, Energy, Rate
│   │   ├── distribution.py       # FAD
│   │   └── itn.py                # ITN 카테고리별 WER
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── loader.py             # 데이터셋 로더
│   │   └── itn_generator.py      # ITN 테스트 케이스 자동 생성
│   └── reports/
│       ├── __init__.py
│       ├── html.py               # HTML 대시보드 생성
│       ├── csv_report.py         # CSV 생성
│       ├── markdown.py           # 마크다운 생성
│       └── compare.py            # 비교 리포트
├── templates/
│   └── dashboard.html            # Jinja2 HTML 템플릿
├── datasets/
│   ├── ko/
│   │   ├── general.json
│   │   ├── itn.json
│   │   └── long_form.json
│   └── schema.json
├── tests/
│   ├── test_metrics/
│   ├── test_models/
│   └── test_reports/
├── docs/
│   └── spec.md                   # 이 문서
├── pyproject.toml
├── README.md
└── config.example.yaml
```

## 9. 구현 단계

### Phase 1: MVP (핵심 파이프라인) -- DONE
- [x] 프로젝트 스캐폴딩 (pyproject.toml, CLI, uv 가상환경)
- [x] BaseTTSModel 어댑터 패턴 + 사용자 커스텀 모델 지원
- [x] 핵심 지표: UTMOS, WER/CER (Seed-TTS 프로토콜)
- [x] 한국어 일반 테스트 세트 (20문장)
- [x] HTML 대시보드 (Chart.js, 오디오 재생, 비교 표)
- [x] CSV 출력 (summary + per_sentence)
- [x] eval-only 모드 (기존 음성 파일 평가)
- [x] 비교 리포트 (compare 서브커맨드)

### Phase 2: 전체 지표 + ITN -- DONE
- [x] PESQ (지각적 음질, 참조 음성 필요)
- [x] STOI (단시간 명료도, 참조 음성 필요)
- [x] Speaker Similarity (ECAPA-TDNN 코사인 유사도)
- [x] Prosody (F0 RMSE, 에너지 상관도, 발화 속도)
- [x] FAD (Frechet Audio Distance, 분포 수준)
- [x] ITN 테스트 세트 한국어 20문항 (숫자/날짜/시간/통화/전화번호/퍼센트/단위/약어/이메일/비율/서수)
- [x] ITN 카테고리별 WER 리포트 (HTML 테이블 + 마크다운)
- [x] 마크다운 리포트 (요약 + 비교 표 + Top/Bottom 샘플)
- [x] 모델 캐시 디렉토리 통합 (.cache/ 하위, 설정으로 변경 가능)
- [x] 참조 음성 로드 및 전달 (speaker_similarity, pesq, stoi)

### Phase 3: 다국어 + 추가 백엔드
- [ ] 영어 데이터셋 및 ITN
- [ ] 추가 TTS 백엔드 (CosyVoice, Coqui, F5-TTS 어댑터)
- [ ] 리포트 비교 대시보드 고도화

### Phase 4: 고도화
- [ ] **실시간 A/B 테스트**: 두 모델을 동시에 돌려서 동일 텍스트로 비교 평가
- [ ] **CI/CD 통합**: GitHub Actions로 모델 체크포인트 변경 시 자동 평가 + 리포트 PR 코멘트
- [ ] **Whisper 외 ASR 백엔드**: Paraformer (중국어), Conformer, faster-whisper (속도 최적화)
- [ ] **MOS 주관 평가 연동**: 자동 평가와 사람 평가를 매핑하여 상관도 분석
- [ ] **긴 문장/문단 평가**: long-form 데이터셋 + 문단 단위 운율 일관성 지표
- [ ] **스트리밍 TTS 평가**: 첫 음절 지연시간(TTFB), 청크 간 이음새 품질
- [ ] **감정/스타일 평가**: 감정 분류기로 의도한 감정 전달 정확도 측정
- [ ] **다화자 대화 TTS**: cpSIM/cpWER (ZipVoice 참고) + 화자 전환 자연스러움
- [ ] **웹 UI 대시보드**: Streamlit/Gradio 기반 인터랙티브 평가 + 오디오 비교 재생
- [ ] **벤치마크 리더보드**: 여러 모델 결과를 누적하여 자동 랭킹 테이블 생성
- [ ] **평가 데이터셋 자동 생성**: LLM으로 도메인별 테스트 문장 생성 (의료, 법률, 방송 등)

## 10. 기술 스택

| 구분 | 선택 | 이유 |
|------|------|------|
| 언어 | Python 3.10+ | TTS/오디오 생태계 |
| CLI | `typer` | 타입 안전, 자동 도움말 |
| 설정 | `pyyaml` + `pydantic` | 유효성 검증 포함 |
| 오디오 I/O | `soundfile`, `librosa` | 표준 |
| STT | `openai-whisper` | 다국어 지원 |
| HTML 템플릿 | `jinja2` | 단일 파일 생성 |
| 차트 | 인라인 Chart.js | 외부 의존성 없는 HTML |
| 패키징 | `pyproject.toml` (hatch) | 현대적 Python 패키징 |

## 11. 핵심 의존성

```
# 필수
typer>=0.9
pydantic>=2.0
pyyaml
jinja2
soundfile
librosa
numpy
tqdm

# 평가
jiwer                  # WER/CER
openai-whisper         # STT
pesq                   # PESQ
pystoi                 # STOI
speechbrain            # Speaker Embedding
parselmouth            # Prosody (Praat)

# 선택 (Phase 2+)
fadtk                  # FAD
num2words              # ITN 기대값 생성
```

## 12. ZipVoice 참고 사항

[k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) eval 모듈 분석 결과 반영:

- **자체 모델 구현**: ZipVoice는 UTMOS, ECAPA-TDNN을 PyTorch로 직접 구현하여 외부 의존성 최소화. 본 프로젝트는 MVP에서 pip 패키지 사용, 안정화 후 필요시 벤더링 고려
- **Seed-TTS WER 프로토콜**: Sentence-level WER (문장별 평균)과 Global WER 두 가지 모두 제공
- **model-dir 패턴**: 평가용 모델 체크포인트를 단일 루트 디렉토리로 관리 (`--model-dir`)
- **load_waveform 유틸**: 스테레오→모노, 리샘플링, max_seconds truncation 포함한 공통 오디오 로딩 유틸리티
- **eval-only 모드**: ZipVoice처럼 이미 생성된 음성 파일만으로도 평가 가능 (`tts-auto-eval eval` 서브커맨드)
- **cpSIM/cpWER**: 다화자 대화 TTS 평가 지표. Phase 3 이후 확장 시 참고
- **torch.set_num_threads(1)**: 배치 평가 시 스레드 경합 방지를 위해 적용
