"""
Reporting utilities for generating JSON and HTML evaluation reports.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def save_json_report(metrics: dict, path: Path):
    """
    Save metrics dictionary as a JSON report.
    Handles numpy types that aren't JSON serializable.
    """
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    report = {
        "generated_at": datetime.now().isoformat(),
        "metrics": metrics
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=convert)

    print(f"JSON report saved to {path}")


def save_html_report(
    metrics: dict,
    comparison: dict,
    model_name: str,
    task: str,
    threshold: float = None,
    path: Path = None,
):
    """
    Generate a clean HTML evaluation report.
    Includes metrics table, workflow comparison and key findings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Format metrics table rows
    metric_rows = ""
    for k, v in metrics.items():
        if v is not None and not isinstance(v, str):
            metric_rows += f"""
            <tr>
                <td>{k}</td>
                <td>{float(v):.4f}</td>
            </tr>"""

    # Format comparison table rows
    comparison_rows = ""
    for stage, score in comparison.items():
        if score is not None:
            comparison_rows += f"""
            <tr>
                <td>{stage}</td>
                <td>{float(score):.4f}</td>
            </tr>"""

    threshold_section = ""
    if threshold is not None:
        threshold_section = f"""
        <div class="section">
            <h2>Threshold</h2>
            <p>Optimal classification threshold: <strong>{threshold:.4f}</strong>
            (default: 0.5)</p>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 900px; margin: 40px auto; padding: 0 20px;
               color: #333; line-height: 1.6; }}
        h1 {{ font-size: 24px; font-weight: 500; border-bottom: 1px solid #eee;
              padding-bottom: 12px; }}
        h2 {{ font-size: 18px; font-weight: 500; margin-top: 32px; }}
        .meta {{ color: #666; font-size: 14px; margin-bottom: 32px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th {{ text-align: left; padding: 10px 12px; background: #f5f5f5;
              font-weight: 500; font-size: 14px; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #eee; font-size: 14px; }}
        tr:last-child td {{ border-bottom: none; }}
        .section {{ margin-bottom: 32px; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px;
                  font-size: 13px; background: #e8f4fd; color: #1a6fa8; }}
    </style>
</head>
<body>
    <h1>Model Evaluation Report</h1>
    <div class="meta">
        <span class="badge">{task}</span>&nbsp;
        <span class="badge">{model_name}</span>&nbsp;
        <span style="color: #999;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
    </div>

    <div class="section">
        <h2>Test set metrics</h2>
        <table>
            <thead><tr><th>Metric</th><th>Score</th></tr></thead>
            <tbody>{metric_rows}</tbody>
        </table>
    </div>

    {threshold_section}

    <div class="section">
        <h2>Workflow comparison</h2>
        <table>
            <thead><tr><th>Stage</th><th>Score</th></tr></thead>
            <tbody>{comparison_rows}</tbody>
        </table>
    </div>

    <div class="section">
        <p style="color: #999; font-size: 13px;">
        Test set was held out throughout the entire workflow and used
        only once for this final evaluation.
        </p>
    </div>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)

    print(f"HTML report saved to {path}")