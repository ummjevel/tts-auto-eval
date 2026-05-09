"""분산 평가 — Ray 기반 대규모 데이터셋 병렬 평가."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tts_auto_eval.config import Config

logger = logging.getLogger(__name__)


def _split_into_chunks(items: list, num_chunks: int) -> list[list]:
    """리스트를 num_chunks개로 분할."""
    if num_chunks <= 0:
        return [items]
    chunk_size = max(1, len(items) // num_chunks)
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunks.append(items[i:i + chunk_size])
    # Merge excess chunks into the last one
    while len(chunks) > num_chunks:
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    return chunks


def _merge_results(chunk_results: list[dict]) -> dict:
    """여러 청크의 평가 결과를 병합."""
    if not chunk_results:
        return {}

    # Use first result as base for metadata
    merged = {
        "metadata": chunk_results[0].get("metadata", {}),
        "summary": {},
        "samples": [],
    }

    # Merge samples
    for cr in chunk_results:
        merged["samples"].extend(cr.get("samples", []))

    # Update total count
    merged["metadata"]["total_samples"] = len(merged["samples"])

    # Re-aggregate summary from all samples
    # Simple approach: average the summaries
    all_summaries = [cr.get("summary", {}) for cr in chunk_results]
    all_metrics = set()
    for s in all_summaries:
        all_metrics.update(s.keys())

    for metric in all_metrics:
        metric_values = []
        for s in all_summaries:
            if metric in s and isinstance(s[metric], dict):
                if "mean" in s[metric]:
                    metric_values.append(s[metric]["mean"])
                elif "sentence_wer" in s[metric]:
                    metric_values.append(s[metric]["sentence_wer"])

        if metric_values:
            merged["summary"][metric] = {
                "mean": float(np.mean(metric_values)),
                "count": len(merged["samples"]),
            }

    return merged


def run_distributed_evaluation(config: Config, num_workers: int = 4) -> dict:
    """Ray를 사용한 분산 평가 실행.

    Args:
        config: 평가 설정
        num_workers: 워커 수

    Returns:
        병합된 평가 결과
    """
    import ray

    ray.init(ignore_reinit_error=True)
    logger.info(f"Ray initialized: {num_workers} workers")

    # Load datasets
    from tts_auto_eval.datasets.loader import load_datasets_from_dir

    datasets = load_datasets_from_dir(
        config.dataset.path,
        categories=config.dataset.categories,
        max_samples=config.dataset.max_samples,
    )

    # Flatten all items
    all_items = []
    for ds in datasets:
        for item in ds.items:
            all_items.append({
                "id": item.id,
                "text": item.text,
                "category": ds.category,
                "language": ds.language,
                "expected_spoken": item.expected_spoken,
                "tags": item.tags,
            })

    if not all_items:
        raise ValueError("No items to evaluate")

    # Split into chunks
    chunks = _split_into_chunks(all_items, num_workers)
    logger.info(f"Split {len(all_items)} items into {len(chunks)} chunks")

    config_dict = config.model_dump()

    @ray.remote
    def evaluate_chunk(chunk_items: list[dict], cfg_dict: dict) -> dict:
        """워커에서 청크 평가 실행."""
        import json
        import tempfile
        from pathlib import Path

        from tts_auto_eval.config import Config
        from tts_auto_eval.pipeline import run_evaluation

        cfg = Config(**cfg_dict)

        # Write chunk items as temporary dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = {
                "language": chunk_items[0]["language"],
                "category": "distributed_chunk",
                "items": chunk_items,
            }
            ds_path = Path(tmpdir) / "chunk.json"
            with open(ds_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False)

            cfg.dataset.path = tmpdir
            cfg.report.output_dir = tmpdir

            return run_evaluation(cfg)

    # Submit tasks
    futures = [evaluate_chunk.remote(chunk, config_dict) for chunk in chunks]

    # Collect results
    logger.info("Waiting for distributed evaluation to complete...")
    chunk_results = ray.get(futures)

    # Merge
    merged = _merge_results(chunk_results)
    merged["metadata"]["distributed"] = True
    merged["metadata"]["num_workers"] = num_workers

    ray.shutdown()
    logger.info("Distributed evaluation complete")

    return merged
