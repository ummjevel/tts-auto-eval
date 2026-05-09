"""플러그인 공개 API.

외부 패키지에서 사용자 정의 메트릭을 등록하기 위한 API.

Example (in external package):
    from tts_auto_eval.plugins import register_metric
    register_metric("custom_metric", "my_package.metrics", "CustomMetric")
"""

from tts_auto_eval.metrics.registry import (
    create_metric,
    discover_plugins,
    list_metrics,
    register_metric,
)

__all__ = ["register_metric", "create_metric", "discover_plugins", "list_metrics"]
