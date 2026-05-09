"""웹 UI 대시보드 — Gradio 기반 인터랙티브 평가 결과 뷰어."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_results_from_dir(results_dir: str | Path) -> list[dict]:
    """디렉토리에서 모든 result.json 로드."""
    results_dir = Path(results_dir)
    results = []
    for path in sorted(results_dir.glob("**/result.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data["_source_path"] = str(path)
            results.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    return results


def _build_summary_table(results: list[dict]) -> list[list[str]]:
    """결과를 요약 테이블로 변환."""
    rows = []
    for r in results:
        meta = r.get("metadata", {})
        summary = r.get("summary", {})
        row = [
            meta.get("model_name", "N/A"),
            meta.get("language", "N/A"),
            str(meta.get("total_samples", 0)),
            meta.get("timestamp", "N/A")[:19],
        ]
        # Add key metrics
        for metric_name in ["utmos", "wer", "pesq", "stoi"]:
            if metric_name in summary:
                val = summary[metric_name]
                if isinstance(val, dict):
                    score = val.get("mean") or val.get("sentence_wer") or val.get("overall_wer")
                    row.append(f"{score:.4f}" if score is not None else "-")
                else:
                    row.append(str(val))
            else:
                row.append("-")
        rows.append(row)
    return rows


def create_dashboard(results_dir: str = "./results"):
    """Gradio 대시보드 생성."""
    import gradio as gr

    results = _load_results_from_dir(results_dir)

    with gr.Blocks(title="TTS Auto Eval Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# TTS Auto Eval Dashboard")

        with gr.Tab("Results Overview"):
            gr.Markdown("## 평가 결과 요약")

            headers = ["모델", "언어", "샘플 수", "시각", "UTMOS", "WER", "PESQ", "STOI"]
            table_data = _build_summary_table(results)

            gr.Dataframe(
                value=table_data,
                headers=headers,
                label="평가 결과",
            )

        with gr.Tab("Compare"):
            gr.Markdown("## 모델 비교")

            model_names = [
                r.get("metadata", {}).get("model_name", f"Model {i}")
                for i, r in enumerate(results)
            ]

            if len(model_names) >= 2:
                with gr.Row():
                    model_a = gr.Dropdown(
                        choices=model_names,
                        label="Model A",
                        value=model_names[0] if model_names else None,
                    )
                    model_b = gr.Dropdown(
                        choices=model_names,
                        label="Model B",
                        value=model_names[1] if len(model_names) > 1 else None,
                    )

                compare_output = gr.JSON(label="비교 결과")

                def compare_models(a_name, b_name):
                    a = next(
                        (r for r in results if r.get("metadata", {}).get("model_name") == a_name),
                        None,
                    )
                    b = next(
                        (r for r in results if r.get("metadata", {}).get("model_name") == b_name),
                        None,
                    )
                    if not a or not b:
                        return {"error": "모델을 찾을 수 없습니다"}
                    return {
                        "model_a": {"name": a_name, "summary": a.get("summary", {})},
                        "model_b": {"name": b_name, "summary": b.get("summary", {})},
                    }

                gr.Button("비교").click(
                    compare_models, inputs=[model_a, model_b], outputs=compare_output
                )
            else:
                gr.Markdown("비교할 모델이 2개 이상 필요합니다.")

        with gr.Tab("Upload"):
            gr.Markdown("## 결과 파일 업로드")
            file_upload = gr.File(label="result.json 업로드", file_types=[".json"])
            upload_output = gr.JSON(label="업로드된 결과 요약")

            def handle_upload(file):
                if file is None:
                    return None
                try:
                    with open(file.name, encoding="utf-8") as f:
                        data = json.load(f)
                    return {
                        "metadata": data.get("metadata", {}),
                        "summary": data.get("summary", {}),
                    }
                except Exception as e:
                    return {"error": str(e)}

            file_upload.change(handle_upload, inputs=file_upload, outputs=upload_output)

    return demo


def launch_dashboard(results_dir: str = "./results", port: int = 7860, share: bool = False):
    """대시보드 실행."""
    demo = create_dashboard(results_dir)
    demo.launch(server_port=port, share=share)
