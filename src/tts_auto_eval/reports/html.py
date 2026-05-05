"""HTML 대시보드 리포트 생성."""

import base64
import json
from pathlib import Path

from jinja2 import Template

# 단일 HTML 파일로 생성 (Chart.js CDN 인라인)
DASHBOARD_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TTS Auto Eval Report — {{ metadata.model_name }}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root { --bg: #f8f9fa; --card: #fff; --text: #212529; --border: #dee2e6;
          --blue: #4361ee; --green: #2ec4b6; --red: #e63946; --gray: #6c757d; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); padding: 2rem; }
  .header { text-align: center; margin-bottom: 2rem; }
  .header h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
  .header .meta { color: var(--gray); font-size: 0.9rem; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 1rem; margin-bottom: 2rem; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
          padding: 1.2rem; text-align: center; }
  .card .label { font-size: 0.85rem; color: var(--gray); margin-bottom: 0.3rem; }
  .card .value { font-size: 2rem; font-weight: 700; }
  .card .sub { font-size: 0.8rem; color: var(--gray); margin-top: 0.3rem; }
  .section { background: var(--card); border: 1px solid var(--border); border-radius: 8px;
             padding: 1.5rem; margin-bottom: 1.5rem; }
  .section h2 { font-size: 1.2rem; margin-bottom: 1rem; border-bottom: 2px solid var(--blue);
                 padding-bottom: 0.5rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th, td { padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }
  th { background: var(--bg); font-weight: 600; }
  tr:hover { background: #f1f3f5; }
  .chart-container { max-width: 600px; margin: 0 auto; }
  audio { width: 100%; max-width: 300px; }
  .good { color: var(--green); } .bad { color: var(--red); }
  .compare-better { background: #d4edda; } .compare-worse { background: #f8d7da; }
</style>
</head>
<body>

<div class="header">
  <h1>TTS Evaluation Report</h1>
  <div class="meta">
    Model: <strong>{{ metadata.model_name }}</strong> |
    Language: {{ metadata.language }} |
    Samples: {{ metadata.total_samples }} |
    {{ metadata.timestamp[:19] }}
  </div>
</div>

<!-- 요약 카드 -->
<div class="cards">
{% if 'utmos' in summary %}
  <div class="card">
    <div class="label">UTMOS (MOS)</div>
    <div class="value">{{ "%.2f"|format(summary.utmos.mean) }}</div>
    <div class="sub">&plusmn; {{ "%.2f"|format(summary.utmos.std) }} ({{ summary.utmos.count }} samples)</div>
  </div>
{% endif %}
{% if 'wer' in summary %}
  <div class="card">
    <div class="label">Sentence WER</div>
    <div class="value">{{ "%.1f"|format(summary.wer.sentence_wer * 100) }}%</div>
    <div class="sub">Global WER: {{ "%.1f"|format(summary.wer.global_wer * 100) }}%</div>
  </div>
  <div class="card">
    <div class="label">Sentence CER</div>
    <div class="value">{{ "%.1f"|format(summary.wer.sentence_cer * 100) }}%</div>
    <div class="sub">{{ summary.wer.count }} samples</div>
  </div>
{% endif %}
</div>

{% if compare %}
<!-- 비교 표 -->
<div class="section">
  <h2>Previous Report Comparison</h2>
  <table>
    <thead>
      <tr><th>Metric</th><th>Previous</th><th>Current</th><th>Diff</th></tr>
    </thead>
    <tbody>
    {% for row in compare %}
      <tr class="{{ row.css_class }}">
        <td>{{ row.metric }}</td>
        <td>{{ row.prev }}</td>
        <td>{{ row.curr }}</td>
        <td>{{ row.diff }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
{% endif %}

<!-- UTMOS 분포 차트 -->
{% if 'utmos' in summary %}
<div class="section">
  <h2>UTMOS Score Distribution</h2>
  <div class="chart-container">
    <canvas id="utmosChart"></canvas>
  </div>
</div>
{% endif %}

<!-- 샘플별 상세 결과 -->
<div class="section">
  <h2>Sample Details</h2>
  <table>
    <thead>
      <tr>
        <th>ID</th><th>Text</th>
        {% if 'utmos' in summary %}<th>UTMOS</th>{% endif %}
        {% if 'wer' in summary %}<th>WER</th><th>Transcription</th>{% endif %}
        {% if include_audio %}<th>Audio</th>{% endif %}
      </tr>
    </thead>
    <tbody>
    {% for s in samples %}
      <tr>
        <td>{{ s.id }}</td>
        <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis">{{ s.text[:80] }}</td>
        {% if 'utmos' in summary %}<td>{{ "%.2f"|format(s.get('utmos', 0)) }}</td>{% endif %}
        {% if 'wer' in summary %}
          <td>{{ "%.1f"|format(s.get('wer', 0) * 100) }}%</td>
          <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis">{{ s.get('wer_details', {}).get('transcription', '') [:80] }}</td>
        {% endif %}
        {% if include_audio and s.get('audio_b64') %}
          <td><audio controls src="data:audio/wav;base64,{{ s.audio_b64 }}"></audio></td>
        {% elif include_audio %}
          <td>-</td>
        {% endif %}
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<script>
{% if 'utmos' in summary %}
const utmosScores = {{ utmos_scores_json }};
const bins = [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
const counts = new Array(bins.length - 1).fill(0);
utmosScores.forEach(s => {
  for (let i = 0; i < bins.length - 1; i++) {
    if (s >= bins[i] && s < bins[i+1]) { counts[i]++; break; }
  }
});
new Chart(document.getElementById('utmosChart'), {
  type: 'bar',
  data: {
    labels: bins.slice(0,-1).map((b,i) => b.toFixed(1)+'-'+bins[i+1].toFixed(1)),
    datasets: [{ label: 'Count', data: counts,
                 backgroundColor: 'rgba(67,97,238,0.6)', borderColor: 'rgba(67,97,238,1)', borderWidth: 1 }]
  },
  options: { scales: { y: { beginAtZero: true } }, plugins: { legend: { display: false } } }
});
{% endif %}
</script>

</body>
</html>
"""


def generate_html_report(
    eval_result: dict,
    output_path: str | Path,
    compare_result: dict | None = None,
    include_audio: bool = True,
    audio_dir: Path | None = None,
) -> Path:
    """HTML 대시보드 리포트 생성."""
    output_path = Path(output_path)

    metadata = eval_result["metadata"]
    summary = eval_result["summary"]
    samples = eval_result["samples"]

    # 오디오 base64 인코딩 (include_audio 시)
    if include_audio and audio_dir:
        for sample in samples:
            audio_path = audio_dir / f"{sample['id']}.wav"
            if audio_path.exists():
                with open(audio_path, "rb") as f:
                    sample["audio_b64"] = base64.b64encode(f.read()).decode()

    # UTMOS 점수 배열
    utmos_scores = [s.get("utmos", 0) for s in samples] if "utmos" in summary else []

    # 비교 데이터
    compare = None
    if compare_result:
        compare = _build_compare_rows(summary, compare_result.get("summary", {}))

    template = Template(DASHBOARD_TEMPLATE)
    html = template.render(
        metadata=metadata,
        summary=summary,
        samples=samples,
        compare=compare,
        include_audio=include_audio,
        utmos_scores_json=json.dumps(utmos_scores),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _build_compare_rows(current: dict, previous: dict) -> list[dict]:
    """비교 표 행 생성."""
    rows = []
    metric_display = {
        "utmos": ("UTMOS (mean)", "mean", True),  # higher is better
        "wer": ("Sentence WER", "sentence_wer", False),  # lower is better
    }

    for key, (label, field, higher_better) in metric_display.items():
        if key in current and key in previous:
            curr_val = current[key].get(field, 0)
            prev_val = previous[key].get(field, 0)
            diff = curr_val - prev_val

            if higher_better:
                css = "compare-better" if diff > 0 else ("compare-worse" if diff < 0 else "")
            else:
                css = "compare-better" if diff < 0 else ("compare-worse" if diff > 0 else "")

            rows.append({
                "metric": label,
                "prev": f"{prev_val:.4f}",
                "curr": f"{curr_val:.4f}",
                "diff": f"{diff:+.4f}",
                "css_class": css,
            })

    return rows
