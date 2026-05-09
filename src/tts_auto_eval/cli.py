"""CLI 엔트리포인트."""

import json
import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="tts-auto-eval",
    help="TTS 모델 자동 평가 및 리포트 생성 도구",
    no_args_is_help=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="설정 파일 경로 (YAML)"),
    output: Path = typer.Option("./results", "--output", "-o", help="결과 출력 디렉토리"),
):
    """TTS 모델 평가 실행 (텍스트 → 음성 생성 → 평가)."""
    from tts_auto_eval.config import load_config
    from tts_auto_eval.pipeline import run_evaluation
    from tts_auto_eval.reports.csv_report import generate_csv_report
    from tts_auto_eval.reports.html import generate_html_report

    cfg = load_config(config)
    cfg.report.output_dir = str(output)

    typer.echo(f"평가 시작: {cfg.evaluation.metrics}")

    result = run_evaluation(cfg)

    # 리포트 생성
    _generate_reports(cfg, result, output)

    typer.echo(f"\n평가 완료! 결과: {output}")


@app.command()
def eval(
    audio_dir: Path = typer.Option(..., "--audio-dir", "-a", help="음성 파일 디렉토리"),
    config: Path = typer.Option(..., "--config", "-c", help="설정 파일 경로 (YAML)"),
    output: Path = typer.Option("./results", "--output", "-o", help="결과 출력 디렉토리"),
):
    """이미 생성된 음성 파일로 평가 (eval-only 모드)."""
    from tts_auto_eval.config import load_config
    from tts_auto_eval.pipeline import run_evaluation

    cfg = load_config(config)
    cfg.report.output_dir = str(output)

    typer.echo(f"Eval-only 모드: {audio_dir}")

    result = run_evaluation(cfg, audio_dir=audio_dir)

    _generate_reports(cfg, result, output, audio_dir=audio_dir)

    typer.echo(f"\n평가 완료! 결과: {output}")


@app.command()
def compare(
    report_a: Path = typer.Argument(..., help="이전 리포트 (result.json)"),
    report_b: Path = typer.Argument(..., help="현재 리포트 (result.json)"),
    output: Path = typer.Option("./comparison", "--output", "-o", help="비교 결과 출력 디렉토리"),
):
    """두 평가 결과를 비교."""
    from tts_auto_eval.reports.compare import load_result
    from tts_auto_eval.reports.html import generate_html_report

    prev = load_result(report_a)
    curr = load_result(report_b)

    output.mkdir(parents=True, exist_ok=True)
    html_path = generate_html_report(
        curr,
        output / "comparison.html",
        compare_result=prev,
        include_audio=False,
    )

    typer.echo(f"비교 리포트 생성: {html_path}")


def _generate_reports(cfg, result, output: Path, audio_dir: Path | None = None):
    """설정에 따라 리포트 생성."""
    from tts_auto_eval.reports.csv_report import generate_csv_report
    from tts_auto_eval.reports.html import generate_html_report

    # 비교 대상 로드
    compare_result = None
    if cfg.report.compare_with:
        from tts_auto_eval.reports.compare import load_result

        compare_path = Path(cfg.report.compare_with)
        if compare_path.exists():
            compare_result = load_result(compare_path)

    if "html" in cfg.report.formats:
        audio_path = audio_dir or (output / "audio")
        html_path = generate_html_report(
            result,
            output / "report.html",
            compare_result=compare_result,
            include_audio=cfg.report.include_audio,
            audio_dir=audio_path if audio_path.exists() else None,
        )
        typer.echo(f"  HTML: {html_path}")

    if "csv" in cfg.report.formats:
        csv_paths = generate_csv_report(result, output)
        for p in csv_paths:
            typer.echo(f"  CSV: {p}")

    if "markdown" in cfg.report.formats:
        from tts_auto_eval.reports.markdown import generate_markdown_report

        md_path = generate_markdown_report(
            result,
            output / "report.md",
            compare_result=compare_result,
        )
        typer.echo(f"  Markdown: {md_path}")


@app.command()
def leaderboard(
    results: list[Path] = typer.Argument(None, help="result.json 파일 경로들"),
    output: Path = typer.Option("./leaderboard", "--output", "-o", help="출력 디렉토리"),
    results_dir: Path | None = typer.Option(None, "--results-dir", "-d", help="result.json을 스캔할 디렉토리"),
):
    """여러 모델 평가 결과를 종합하여 랭킹 리더보드 생성."""
    from tts_auto_eval.leaderboard import (
        build_leaderboard,
        generate_leaderboard_markdown,
        load_results,
    )

    # Collect result paths
    all_paths = list(results) if results else []
    if results_dir and results_dir.is_dir():
        all_paths.extend(sorted(results_dir.glob("**/result.json")))

    if not all_paths:
        typer.echo("result.json 파일을 지정하세요.", err=True)
        raise typer.Exit(1)

    typer.echo(f"리더보드 생성: {len(all_paths)}개 결과 파일")

    loaded = load_results(all_paths)
    lb = build_leaderboard(loaded)

    output.mkdir(parents=True, exist_ok=True)
    md_path = generate_leaderboard_markdown(lb, output / "leaderboard.md")

    typer.echo(f"리더보드 생성 완료: {md_path}")


if __name__ == "__main__":
    app()
