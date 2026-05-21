from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LintIssue:
    severity: str
    code: str
    message: str
    line_number: int | None = None
    context: str | None = None


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)
    line_count: int = 0
    min_columns: int = 0
    max_columns: int = 0
    initial_columns: int = 0
    counts_by_code: dict[str, int] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def add_issue(
        self,
        severity: str,
        code: str,
        message: str,
        line_number: int | None = None,
        context: str | None = None,
    ) -> None:
        self.issues.append(
            LintIssue(
                severity=severity,
                code=code,
                message=message,
                line_number=line_number,
                context=context,
            )
        )
        self.counts_by_code[code] = self.counts_by_code.get(code, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_count": self.line_count,
            "min_columns": self.min_columns,
            "max_columns": self.max_columns,
            "initial_columns": self.initial_columns,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "counts_by_code": self.counts_by_code,
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass
class ValidationReport:
    status: str
    issues: list[str] = field(default_factory=list)
    measure_count: int = 0
    parsable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactPaths:
    prepared_image: Path
    transcription_txt: Path
    raw_kern: Path
    normalized_ekern: Path
    export_kern: Path
    lint_json: Path
    page_report_json: Path
    musicxml: Path | None = None
    score_svg: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "prepared_image": str(self.prepared_image),
            "transcription_txt": str(self.transcription_txt),
            "raw_kern": str(self.raw_kern),
            "normalized_ekern": str(self.normalized_ekern),
            "export_kern": str(self.export_kern),
            "lint_json": str(self.lint_json),
            "page_report_json": str(self.page_report_json),
            "musicxml": None if self.musicxml is None else str(self.musicxml),
            "score_svg": None if self.score_svg is None else str(self.score_svg),
        }


@dataclass
class PageJob:
    input_path: Path
    page_number: int | None
    output_dir: Path

    @property
    def job_key(self) -> str:
        if self.page_number is None:
            return self.input_path.stem
        return f"{self.input_path.stem}.page-{self.page_number}"


@dataclass
class PageResult:
    job: PageJob
    model_name: str
    device: str
    inference_dtype: str
    original_size: tuple[int, int]
    prepared_size: tuple[int, int]
    transcription: str
    normalized_ekern_text: str
    lint_report: LintReport
    validation_report: ValidationReport
    artifact_paths: ArtifactPaths
    musicxml_error: str | None = None
    runtime_seconds: float | None = None

    @property
    def status(self) -> str:
        if self.musicxml_error:
            return "failed_export"
        return self.validation_report.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": {
                "input_path": str(self.job.input_path),
                "page_number": self.job.page_number,
                "output_dir": str(self.job.output_dir),
            },
            "model_name": self.model_name,
            "device": self.device,
            "inference_dtype": self.inference_dtype,
            "original_size": list(self.original_size),
            "prepared_size": list(self.prepared_size),
            "status": self.status,
            "runtime_seconds": self.runtime_seconds,
            "musicxml_error": self.musicxml_error,
            "artifact_paths": self.artifact_paths.to_dict(),
            "lint_report": self.lint_report.to_dict(),
            "validation_report": self.validation_report.to_dict(),
        }


@dataclass
class BatchSummary:
    results: list[PageResult]
    device: str
    inference_dtype: str
    model_name: str

    def to_dict(self) -> dict[str, Any]:
        total_errors = sum(result.lint_report.error_count for result in self.results)
        total_warnings = sum(result.lint_report.warning_count for result in self.results)
        return {
            "model_name": self.model_name,
            "device": self.device,
            "inference_dtype": self.inference_dtype,
            "item_count": len(self.results),
            "total_lint_errors": total_errors,
            "total_lint_warnings": total_warnings,
            "results": [result.to_dict() for result in self.results],
        }
