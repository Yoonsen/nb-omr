from __future__ import annotations

from pathlib import Path

from .ekern import lint_raw_transcription, normalize_to_ekern, validate_normalized_ekern
from .export import convert_normalized_ekern_to_musicxml, render_musicxml_to_svg
from .inference import InferenceSession
from .preprocess import prepare_input_image, resize_for_model
from .reporting import write_lint_report, write_page_report
from .types import ArtifactPaths, PageJob, PageResult


class OMRPipeline:
    def __init__(self, session: InferenceSession) -> None:
        self.session = session

    def run_job(self, job: PageJob) -> PageResult:
        rendered_path = prepare_input_image(job.input_path, job.page_number, job.output_dir)
        max_width, max_height = self.session.max_size
        prepared_path, original_size, prepared_size = resize_for_model(
            rendered_path,
            job.output_dir / f"{rendered_path.stem}.fit.png",
            max_width=max_width,
            max_height=max_height,
        )

        transcription, runtime_seconds = self.session.transcribe_prepared_image(prepared_path)
        lint_report = lint_raw_transcription(transcription)
        normalized_ekern = normalize_to_ekern(transcription, lint_report)
        validation_report = validate_normalized_ekern(normalized_ekern, lint_report)

        output_prefix = job.output_dir / f"{prepared_path.stem}.{self.session.model_name.split('/')[-1]}"
        artifacts = ArtifactPaths(
            prepared_image=prepared_path,
            transcription_txt=Path(f"{output_prefix}.txt"),
            raw_kern=Path(f"{output_prefix}.raw.krn"),
            normalized_ekern=Path(f"{output_prefix}.normalized.ekrn"),
            export_kern=Path(f"{output_prefix}.export.krn"),
            lint_json=Path(f"{output_prefix}.lint.json"),
            page_report_json=Path(f"{output_prefix}.report.json"),
            musicxml=Path(f"{output_prefix}.musicxml"),
            score_svg=Path(f"{output_prefix}.preview.svg"),
        )

        artifacts.transcription_txt.write_text(transcription, encoding="utf-8")
        artifacts.raw_kern.write_text(
            transcription + ("\n" if not transcription.endswith("\n") else ""),
            encoding="utf-8",
        )
        artifacts.normalized_ekern.write_text(normalized_ekern, encoding="utf-8")
        write_lint_report(artifacts.lint_json, lint_report)

        musicxml_error: str | None = None
        if validation_report.parsable:
            try:
                convert_normalized_ekern_to_musicxml(
                    normalized_ekern,
                    artifacts.musicxml,
                    export_source_path=artifacts.export_kern,
                )
                render_musicxml_to_svg(artifacts.musicxml, artifacts.score_svg)
            except Exception as exc:
                musicxml_error = str(exc)
                artifacts.musicxml = None
                artifacts.score_svg = None
        else:
            musicxml_error = "Normalized ekern did not pass validation."
            artifacts.musicxml = None
            artifacts.score_svg = None

        result = PageResult(
            job=job,
            model_name=self.session.model_name,
            device=str(self.session.device),
            inference_dtype=self.session.dtype_label,
            original_size=original_size,
            prepared_size=prepared_size,
            transcription=transcription,
            normalized_ekern_text=normalized_ekern,
            lint_report=lint_report,
            validation_report=validation_report,
            artifact_paths=artifacts,
            musicxml_error=musicxml_error,
            runtime_seconds=runtime_seconds,
        )
        write_page_report(artifacts.page_report_json, result)
        return result
