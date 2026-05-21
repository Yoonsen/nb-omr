from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from nb_omr import InferenceSession, OMRPipeline, enumerate_jobs, run_jobs

DEFAULT_MODEL = "PRAIG/smt-fp-grandstaff"

st.set_page_config(page_title="nb-omr", layout="wide")
st.title("nb-omr")
st.caption("Internal OMR workbench for raw SMT output, normalized ekern, and MusicXML artifacts.")

uploaded_file = st.file_uploader("Upload a PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"])
all_pages = st.checkbox("Process all pages in a PDF", value=False)
page_number = st.number_input("PDF page", min_value=1, value=1, step=1)

with st.expander("Runtime options"):
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    device = st.text_input("Device", value="auto")
    dtype = st.selectbox("Dtype", ["auto", "float32", "float16", "bfloat16"], index=0)
    compile_model = st.checkbox("Compile model", value=False)


def render_result(result) -> None:
    with st.expander(f"{result.job.job_key} ({result.status})", expanded=True):
        left, right = st.columns([1, 1])
        with left:
            st.image(str(result.artifact_paths.prepared_image), caption="Prepared image")
            st.metric("Lint warnings", result.lint_report.warning_count)
            st.metric("Lint errors", result.lint_report.error_count)
            st.metric("Validation status", result.validation_report.status)
        with right:
            st.write(
                {
                    "device": result.device,
                    "dtype": result.inference_dtype,
                    "original_size": result.original_size,
                    "prepared_size": result.prepared_size,
                    "runtime_seconds": result.runtime_seconds,
                }
            )
            if result.validation_report.issues:
                st.warning("\n".join(result.validation_report.issues))
            if result.musicxml_error:
                st.error(result.musicxml_error)

        text_col, norm_col = st.columns(2)
        with text_col:
            st.subheader("Raw transcription")
            st.text_area(
                "Raw transcription text",
                result.transcription,
                height=320,
                key=f"raw-{result.job.job_key}",
            )
        with norm_col:
            st.subheader("Normalized ekern")
            st.text_area(
                "Normalized ekern text",
                result.normalized_ekern_text,
                height=320,
                key=f"norm-{result.job.job_key}",
            )

        download_cols = st.columns(5)
        with download_cols[0]:
            st.download_button(
                "Download raw.krn",
                Path(result.artifact_paths.raw_kern).read_text(encoding="utf-8"),
                file_name=result.artifact_paths.raw_kern.name,
                mime="text/plain",
                key=f"download-raw-{result.job.job_key}",
            )
        with download_cols[1]:
            st.download_button(
                "Download normalized.ekrn",
                Path(result.artifact_paths.normalized_ekern).read_text(encoding="utf-8"),
                file_name=result.artifact_paths.normalized_ekern.name,
                mime="text/plain",
                key=f"download-norm-{result.job.job_key}",
            )
        with download_cols[2]:
            st.download_button(
                "Download lint.json",
                Path(result.artifact_paths.lint_json).read_text(encoding="utf-8"),
                file_name=result.artifact_paths.lint_json.name,
                mime="application/json",
                key=f"download-lint-{result.job.job_key}",
            )
        with download_cols[3]:
            st.download_button(
                "Download export.krn",
                Path(result.artifact_paths.export_kern).read_text(encoding="utf-8"),
                file_name=result.artifact_paths.export_kern.name,
                mime="text/plain",
                key=f"download-export-{result.job.job_key}",
            )
        with download_cols[4]:
            if result.artifact_paths.musicxml and result.artifact_paths.musicxml.exists():
                st.download_button(
                    "Download MusicXML",
                    Path(result.artifact_paths.musicxml).read_text(encoding="utf-8"),
                    file_name=result.artifact_paths.musicxml.name,
                    mime="application/xml",
                    key=f"download-xml-{result.job.job_key}",
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

        st.success(f"Processed {len(results)} item(s).")
        st.write({"batch_summary": str(summary_json), "meeting_report": str(meeting_report)})
        for result in results:
            render_result(result)
