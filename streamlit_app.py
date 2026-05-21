from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path

import streamlit as st

from nb_omr import InferenceSession, OMRPipeline, enumerate_jobs, run_jobs

DEFAULT_MODEL = "PRAIG/smt-fp-grandstaff"

st.set_page_config(page_title="nb-omr", layout="wide")
st.title("nb-omr")
st.caption("Internal OMR workbench for raw SMT output, normalized ekern, and MusicXML artifacts.")

if "run_results" not in st.session_state:
    st.session_state["run_results"] = []
if "run_summary" not in st.session_state:
    st.session_state["run_summary"] = None

uploaded_file = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"])
all_pages = st.checkbox("Process all pages in a PDF", value=False)
page_number = st.number_input("PDF page", min_value=1, value=1, step=1)

with st.expander("Runtime options"):
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    device = st.text_input("Device", value="auto")
    dtype = st.selectbox("Dtype", ["auto", "float32", "float16", "bfloat16"], index=0)
    compile_model = st.checkbox("Compile model", value=False)


def svg_has_visible_size(svg_text: str | None) -> bool:
    if not svg_text:
        return False
    width_match = re.search(r'width="([0-9.]+)px"', svg_text)
    height_match = re.search(r'height="([0-9.]+)px"', svg_text)
    if not width_match or not height_match:
        return True
    return float(width_match.group(1)) > 0 and float(height_match.group(1)) > 0


def snapshot_result(result) -> dict:
    return {
        "job_key": result.job.job_key,
        "input_path": str(result.job.input_path),
        "page_number": result.job.page_number,
        "status": result.status,
        "device": result.device,
        "dtype": result.inference_dtype,
        "original_size": result.original_size,
        "prepared_size": result.prepared_size,
        "runtime_seconds": result.runtime_seconds,
        "transcription": result.transcription,
        "normalized_ekern_text": result.normalized_ekern_text,
        "lint_warning_count": result.lint_report.warning_count,
        "lint_error_count": result.lint_report.error_count,
        "validation_status": result.validation_report.status,
        "validation_issues": list(result.validation_report.issues),
        "musicxml_error": result.musicxml_error,
        "prepared_image_name": result.artifact_paths.prepared_image.name,
        "prepared_image_bytes": result.artifact_paths.prepared_image.read_bytes(),
        "raw_kern_name": result.artifact_paths.raw_kern.name,
        "raw_kern_text": result.artifact_paths.raw_kern.read_text(encoding="utf-8"),
        "normalized_ekern_name": result.artifact_paths.normalized_ekern.name,
        "lint_json_name": result.artifact_paths.lint_json.name,
        "lint_json_text": result.artifact_paths.lint_json.read_text(encoding="utf-8"),
        "export_kern_name": result.artifact_paths.export_kern.name,
        "export_kern_text": result.artifact_paths.export_kern.read_text(encoding="utf-8"),
        "musicxml_name": None if result.artifact_paths.musicxml is None else result.artifact_paths.musicxml.name,
        "musicxml_text": (
            None
            if result.artifact_paths.musicxml is None
            else result.artifact_paths.musicxml.read_text(encoding="utf-8")
        ),
        "score_svg_name": None if result.artifact_paths.score_svg is None else result.artifact_paths.score_svg.name,
        "score_svg_text": (
            None
            if result.artifact_paths.score_svg is None
            else result.artifact_paths.score_svg.read_text(encoding="utf-8")
        ),
    }


def render_result(result: dict) -> None:
    with st.expander(f"{result['job_key']} ({result['status']})", expanded=True):
        left, right = st.columns([1, 1])
        with left:
            st.image(result["prepared_image_bytes"], caption=result["prepared_image_name"])
            st.metric("Lint warnings", result["lint_warning_count"])
            st.metric("Lint errors", result["lint_error_count"])
            st.metric("Validation status", result["validation_status"])
        with right:
            st.write(
                {
                    "device": result["device"],
                    "dtype": result["dtype"],
                    "original_size": result["original_size"],
                    "prepared_size": result["prepared_size"],
                    "runtime_seconds": result["runtime_seconds"],
                }
            )
            if result["validation_issues"]:
                st.warning("\n".join(result["validation_issues"]))
            if result["musicxml_error"]:
                st.error(result["musicxml_error"])

        if result["score_svg_text"] and svg_has_visible_size(result["score_svg_text"]):
            st.subheader("Score preview")
            try:
                svg_b64 = base64.b64encode(result["score_svg_text"].encode("utf-8")).decode("ascii")
                st.markdown(
                    f'<img src="data:image/svg+xml;base64,{svg_b64}" '
                    'style="width: 100%; height: auto;" />',
                    unsafe_allow_html=True,
                )
                if result["score_svg_name"]:
                    st.caption(result["score_svg_name"])
            except Exception as exc:
                st.warning(f"Could not display SVG preview in the app: {exc}")
        elif result["score_svg_text"]:
            st.info("SVG preview was generated, but it has zero width/height and was hidden.")

        text_col, norm_col = st.columns(2)
        with text_col:
            st.subheader("Raw transcription")
            st.text_area(
                "Raw transcription text",
                result["transcription"],
                height=320,
                key=f"raw-{result['job_key']}",
            )
        with norm_col:
            st.subheader("Normalized ekern")
            st.text_area(
                "Normalized ekern text",
                result["normalized_ekern_text"],
                height=320,
                key=f"norm-{result['job_key']}",
            )

        download_cols = st.columns(6)
        with download_cols[0]:
            st.download_button(
                "Download raw.krn",
                result["raw_kern_text"],
                file_name=result["raw_kern_name"],
                mime="text/plain",
                key=f"download-raw-{result['job_key']}",
            )
        with download_cols[1]:
            st.download_button(
                "Download normalized.ekrn",
                result["normalized_ekern_text"],
                file_name=result["normalized_ekern_name"],
                mime="text/plain",
                key=f"download-norm-{result['job_key']}",
            )
        with download_cols[2]:
            st.download_button(
                "Download lint.json",
                result["lint_json_text"],
                file_name=result["lint_json_name"],
                mime="application/json",
                key=f"download-lint-{result['job_key']}",
            )
        with download_cols[3]:
            st.download_button(
                "Download export.krn",
                result["export_kern_text"],
                file_name=result["export_kern_name"],
                mime="text/plain",
                key=f"download-export-{result['job_key']}",
            )
        with download_cols[4]:
            if result["score_svg_text"] and svg_has_visible_size(result["score_svg_text"]):
                st.download_button(
                    "Download preview.svg",
                    result["score_svg_text"],
                    file_name=result["score_svg_name"],
                    mime="image/svg+xml",
                    key=f"download-svg-{result['job_key']}",
                )
        with download_cols[5]:
            if result["musicxml_text"]:
                st.download_button(
                    "Download MusicXML",
                    result["musicxml_text"],
                    file_name=result["musicxml_name"],
                    mime="application/xml",
                    key=f"download-xml-{result['job_key']}",
                )


if uploaded_file is not None and st.button("Run OMR", type="primary"):
    with tempfile.TemporaryDirectory(prefix="nb-omr-streamlit-") as temp_dir:
        temp_root = Path(temp_dir)
        input_path = temp_root / uploaded_file.name
        input_path.write_bytes(uploaded_file.getbuffer())
        output_dir = temp_root / "outputs"

        with st.spinner("Running inference and post-processing..."):
            session = InferenceSession(
                model_name=model_name,
                device_arg=device,
                dtype_arg=dtype,
                compile_model=compile_model,
            )
            pipeline = OMRPipeline(session)
            jobs = enumerate_jobs(input_path, output_dir, page_number=int(page_number), all_pages=all_pages)
            results, summary_json, meeting_report = run_jobs(pipeline, jobs, output_dir)
            st.session_state["run_results"] = [snapshot_result(result) for result in results]
            st.session_state["run_summary"] = {
                "result_count": len(results),
                "batch_summary_name": summary_json.name,
                "batch_summary_text": summary_json.read_text(encoding="utf-8"),
                "meeting_report_name": meeting_report.name,
                "meeting_report_text": meeting_report.read_text(encoding="utf-8"),
            }

if st.session_state["run_summary"] is not None:
    summary = st.session_state["run_summary"]
    st.success(f"Processed {summary['result_count']} item(s).")
    st.write(
        {
            "batch_summary": summary["batch_summary_name"],
            "meeting_report": summary["meeting_report_name"],
        }
    )
    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.download_button(
            "Download batch-summary.json",
            summary["batch_summary_text"],
            file_name=summary["batch_summary_name"],
            mime="application/json",
            key="download-batch-summary",
        )
    with summary_cols[1]:
        st.download_button(
            "Download meeting-report.md",
            summary["meeting_report_text"],
            file_name=summary["meeting_report_name"],
            mime="text/markdown",
            key="download-meeting-report",
        )
    for result in st.session_state["run_results"]:
        render_result(result)
