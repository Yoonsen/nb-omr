from __future__ import annotations

from pathlib import Path

from .pipeline import OMRPipeline
from .preprocess import SUPPORTED_IMAGE_SUFFIXES, get_pdf_page_count
from .reporting import write_batch_report
from .types import BatchSummary, PageJob, PageResult


def enumerate_jobs(
    input_path: Path,
    output_dir: Path,
    page_number: int = 1,
    all_pages: bool = False,
) -> list[PageJob]:
    if input_path.is_dir():
        jobs: list[PageJob] = []
        for candidate in sorted(input_path.iterdir()):
            if candidate.is_dir():
                continue
            suffix = candidate.suffix.lower()
            if suffix in SUPPORTED_IMAGE_SUFFIXES:
                jobs.append(PageJob(input_path=candidate, page_number=None, output_dir=output_dir))
            elif suffix == ".pdf":
                if all_pages:
                    jobs.extend(
                        PageJob(input_path=candidate, page_number=page, output_dir=output_dir)
                        for page in range(1, get_pdf_page_count(candidate) + 1)
                    )
                else:
                    jobs.append(PageJob(input_path=candidate, page_number=page_number, output_dir=output_dir))
        return jobs

    if input_path.suffix.lower() == ".pdf":
        if all_pages:
            return [
                PageJob(input_path=input_path, page_number=page, output_dir=output_dir)
                for page in range(1, get_pdf_page_count(input_path) + 1)
            ]
        return [PageJob(input_path=input_path, page_number=page_number, output_dir=output_dir)]

    return [PageJob(input_path=input_path, page_number=None, output_dir=output_dir)]


def run_jobs(
    pipeline: OMRPipeline,
    jobs: list[PageJob],
    output_dir: Path,
) -> tuple[list[PageResult], Path, Path]:
    results = [pipeline.run_job(job) for job in jobs]
    summary = BatchSummary(
        results=results,
        device=str(pipeline.session.device),
        inference_dtype=pipeline.session.dtype_label,
        model_name=pipeline.session.model_name,
    )
    json_path, markdown_path = write_batch_report(output_dir, summary)
    return results, json_path, markdown_path
