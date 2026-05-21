import argparse
from pathlib import Path

from nb_omr import InferenceSession, OMRPipeline, enumerate_jobs, run_jobs

DEFAULT_MODEL = "PRAIG/smt-fp-grandstaff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal OMR prototype with Sheet Music Transformer."
    )
    parser.add_argument("input_path", type=Path, help="Path to a PDF, image file, or directory.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id to load. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="1-based PDF page number to render when not using --all-pages.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated artifacts.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device to use. Default: auto (prefers cuda, then mps, then cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Inference dtype. Default: auto (prefers bfloat16 on CUDA when supported).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch.compile for repeated GPU inference runs.",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Process all pages in a PDF, or all PDFs in a directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input_path}")

    session = InferenceSession(
        model_name=args.model,
        device_arg=args.device,
        dtype_arg=args.dtype,
        compile_model=args.compile,
    )
    pipeline = OMRPipeline(session)
    jobs = enumerate_jobs(args.input_path, args.output_dir, page_number=args.page, all_pages=args.all_pages)
    results, summary_json, meeting_report = run_jobs(pipeline, jobs, args.output_dir)

    print(f"Model: {session.model_name}")
    print(f"Device: {session.device}")
    if session.device.type == "cuda":
        import torch

        print(f"GPU: {torch.cuda.get_device_name(session.device)}")
    print(f"Inference dtype: {session.dtype_label}")
    print(f"Processed items: {len(results)}")
    print(f"Batch summary saved to: {summary_json}")
    print(f"Meeting report saved to: {meeting_report}")
    print()

    for result in results:
        print(f"Input: {result.job.input_path}")
        if result.job.page_number is not None:
            print(f"Page: {result.job.page_number}")
        print(f"Prepared image: {result.artifact_paths.prepared_image}")
        print(f"Original image size: {result.original_size[0]}x{result.original_size[1]}")
        print(f"Prepared image size: {result.prepared_size[0]}x{result.prepared_size[1]}")
        print(f"Transcription saved to: {result.artifact_paths.transcription_txt}")
        print(f"Raw kern-like output saved to: {result.artifact_paths.raw_kern}")
        print(f"Normalized ekern saved to: {result.artifact_paths.normalized_ekern}")
        print(f"Export-safe kern saved to: {result.artifact_paths.export_kern}")
        print(f"Lint report saved to: {result.artifact_paths.lint_json}")
        print(f"Page report saved to: {result.artifact_paths.page_report_json}")
        if result.artifact_paths.musicxml is not None:
            print(f"MusicXML saved to: {result.artifact_paths.musicxml}")
        else:
            print(f"MusicXML conversion failed: {result.musicxml_error}")
        print(f"Status: {result.status}")
        print()
        print(result.transcription)
        print()


if __name__ == "__main__":
    main()
