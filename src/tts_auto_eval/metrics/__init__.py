"""평가 지표 모듈."""

from tts_auto_eval.metrics.registry import register_metric

# 기존 메트릭 등록
register_metric("utmos", "tts_auto_eval.metrics.quality", "UTMOSMetric")
register_metric("wer", "tts_auto_eval.metrics.intelligibility", "WERMetric")
register_metric("pesq", "tts_auto_eval.metrics.signal_quality", "PESQMetric")
register_metric("stoi", "tts_auto_eval.metrics.signal_quality", "STOIMetric")
register_metric("speaker_similarity", "tts_auto_eval.metrics.speaker", "SpeakerSimilarityMetric")
register_metric("prosody", "tts_auto_eval.metrics.prosody", "ProsodyMetric")
register_metric("fad", "tts_auto_eval.metrics.distribution", "FADMetric")
register_metric("itn", "tts_auto_eval.metrics.itn", "ITNMetric")

# Phase 5 메트릭
register_metric("phoneme", "tts_auto_eval.metrics.phoneme", "PhonemeMetric")

# Phase 6 메트릭
register_metric("emotion", "tts_auto_eval.metrics.emotion", "EmotionMetric")
register_metric("multispeaker", "tts_auto_eval.metrics.multispeaker", "MultiSpeakerMetric")
