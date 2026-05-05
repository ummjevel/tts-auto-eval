# TTS Auto Eval

TTS 모델 자동 평가 및 리포트 생성 도구.

## 설치

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## 사용법

```bash
# 모델 평가
tts-auto-eval run --config config.yaml --output results/

# 기존 음성 파일 평가
tts-auto-eval eval --audio-dir ./audio/ --config config.yaml --output results/

# 결과 비교
tts-auto-eval compare result_a.json result_b.json --output comparison/
```

설정 예시: `config.example.yaml`
