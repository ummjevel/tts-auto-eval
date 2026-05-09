"""메트릭 레지스트리 — 플러그인 시스템 기반 지표 관리."""

import importlib
import logging
from importlib.metadata import entry_points
from typing import Any

from tts_auto_eval.metrics.base import BaseMetric

logger = logging.getLogger(__name__)

_METRIC_REGISTRY: dict[str, tuple[str, str]] = {}


def register_metric(name: str, module_path: str, class_name: str) -> None:
    """메트릭을 레지스트리에 등록."""
    _METRIC_REGISTRY[name] = (module_path, class_name)


def create_metric(name: str, **kwargs: Any) -> BaseMetric:
    """레지스트리에서 메트릭 인스턴스 생성."""
    if name not in _METRIC_REGISTRY:
        raise KeyError(f"Unknown metric: {name}")
    module_path, class_name = _METRIC_REGISTRY[name]
    module = importlib.import_module(module_path)
    metric_class = getattr(module, class_name)
    return metric_class(**kwargs)


def discover_plugins() -> None:
    """entry_points에서 외부 플러그인 메트릭 자동 탐색 및 등록."""
    try:
        eps = entry_points()
        # Python 3.12+: eps is a dict-like; Python 3.10-3.11: use select()
        if hasattr(eps, "select"):
            metric_eps = eps.select(group="tts_auto_eval.metrics")
        else:
            metric_eps = eps.get("tts_auto_eval.metrics", [])
        for ep in metric_eps:
            if ep.name not in _METRIC_REGISTRY:
                # entry_point value format: "module:ClassName"
                module_path, class_name = ep.value.split(":")
                register_metric(ep.name, module_path, class_name)
                logger.debug(f"Plugin metric registered: {ep.name}")
    except Exception as e:
        logger.debug(f"Plugin discovery skipped: {e}")


def list_metrics() -> list[str]:
    """등록된 메트릭 목록 반환."""
    return sorted(_METRIC_REGISTRY.keys())
