from __future__ import annotations

import json
from pathlib import Path

from .types import BatchSummary, LintReport, PageResult


def write_json_artifact(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_lint_report(path: Path, lint_report: LintReport) -> None:
    write_json_artifact(path, lint_report.to_dict())


def write_page_report(path: Path, page_result: PageResult) -> None:
    write_json_artifact(path, page_result.to_dict())


def render_batch_markdown(summary: BatchSummary) -> str:
    lines = [
        "# OMR Batch Report",
        "",
        f"- Model: `{summary.model_name}`",
        f"- Device: `{summary.device}`",
        f"- Inference dtype: `{summary.inference_dtype}`",
        f"- Items: `{len(summary.results)}`",
        "",
        "## Results",
        "",
    ]
    for result in summary.results:
        job_label = result.job.job_key
        lines.append(f"### {job_label}")
        lines.append("")
        lines.append(f"- Status: `{result.status}`")
        lines.append(f"- Prepared image: `{result.artifact_paths.prepared_image}`")
        lines.append(f"- Raw output: `{result.artifact_paths.raw_kern}`")
        lines.append(f"- Normalized ekern: `{result.artifact_paths.normalized_ekern}`")
        if result.artifact_paths.musicxml:
            lines.append(f"- MusicXML: `{result.artifact_paths.musicxml}`")
        if result.musicxml_error:
            lines.append(f"- MusicXML error: `{result.musicxml_error}`")
        lines.append(f"- Lint warnings: `{result.lint_report.warning_count}`")
        lines.append(f"- Lint errors: `{result.lint_report.error_count}`")
        if result.validation_report.issues:
            lines.append("- Validation notes:")
            for issue in result.validation_report.issues[:5]:
                lines.append(f"  - {issue}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def write_batch_report(output_dir: Path, summary: BatchSummary) -> tuple[Path, Path]:
    json_path = output_dir / "batch-summary.json"
    markdown_path = output_dir / "meeting-report.md"
    write_json_artifact(json_path, summary.to_dict())
    markdown_path.write_text(render_batch_markdown(summary), encoding="utf-8")
    return json_path, markdown_path
