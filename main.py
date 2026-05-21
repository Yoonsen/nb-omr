import argparse
from pathlib import Path

import cv2
import fitz
import torch
from PIL import Image

from data_augmentation import convert_img_to_tensor
from smt_model import SMTModelForCausalLM

DEFAULT_MODEL = "PRAIG/smt-fp-grandstaff"
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal OMR prototype with Sheet Music Transformer."
    )
    parser.add_argument("input_path", type=Path, help="Path to a PDF or image file.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model id to load. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=1,
        help="1-based PDF page number to render. Ignored for images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for the rendered page image and transcription output.",
    )
    return parser.parse_args()


def render_pdf_page_to_png(pdf_path: Path, png_path: Path, page_number: int) -> None:
    doc = fitz.open(pdf_path)
    try:
        page_index = page_number - 1
        if page_index < 0 or page_index >= doc.page_count:
            raise ValueError(f"Invalid PDF page {page_number}. Document has {doc.page_count} pages.")
        pix = doc[page_index].get_pixmap(dpi=300)
        pix.save(str(png_path))
    finally:
        doc.close()


def normalize_image_to_png(src_path: Path, dst_png_path: Path) -> None:
    with Image.open(src_path) as image:
        image.convert("RGB").save(dst_png_path, format="PNG")


def resize_for_model(src_path: Path, dst_path: Path, max_width: int, max_height: int) -> Path:
    with Image.open(src_path) as image:
        width, height = image.size
        if width <= max_width and height <= max_height:
            return src_path

        resized = image.copy()
        resized.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        resized.save(dst_path, format="PNG")
        return dst_path


def prepare_input_image(input_path: Path, page_number: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        output_path = output_dir / f"{input_path.stem}.page-{page_number}.png"
        render_pdf_page_to_png(input_path, output_path, page_number)
        return output_path

    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        output_path = output_dir / f"{input_path.stem}.normalized.png"
        normalize_image_to_png(input_path, output_path)
        return output_path

    raise ValueError(f"Unsupported input type: {input_path.suffix}")


def format_prediction(tokens: list[str]) -> str:
    return "".join(tokens).replace("<b>", "\n").replace("<s>", " ").replace("<t>", "\t")


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input_path}")

    rendered_path = prepare_input_image(args.input_path, args.page, args.output_dir)
    device = resolve_device()
    model = SMTModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    if "<bos>" not in model.w2i or "<eos>" not in model.i2w.values():
        raise RuntimeError(
            "Loaded SMT vocabulary does not look complete. "
            "The model config is missing expected <bos>/<eos> tokens."
        )

    prepared_path = resize_for_model(
        rendered_path,
        args.output_dir / f"{rendered_path.stem}.fit.png",
        max_width=model.config.maxw,
        max_height=model.config.maxh,
    )
    image = cv2.imread(str(prepared_path))
    if image is None:
        raise RuntimeError(f"Could not read prepared input image: {prepared_path}")

    input_tensor = convert_img_to_tensor(image).unsqueeze(0).to(device)
    predictions, _ = model.predict(input_tensor, convert_to_str=True)
    transcription = format_prediction(predictions)

    txt_output = args.output_dir / f"{prepared_path.stem}.{args.model.split('/')[-1]}.txt"
    txt_output.write_text(transcription, encoding="utf-8")

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Prepared image: {prepared_path}")
    print(f"Transcription saved to: {txt_output}")
    print()
    print(transcription)


if __name__ == "__main__":
    main()
