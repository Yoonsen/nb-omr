from __future__ import annotations

from pathlib import Path

import fitz
from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


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


def resize_for_model(
    src_path: Path,
    dst_path: Path,
    max_width: int,
    max_height: int,
) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    with Image.open(src_path) as image:
        width, height = image.size
        if width <= max_width and height <= max_height:
            return src_path, (width, height), (width, height)

        resized = image.copy()
        resized.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        resized.save(dst_path, format="PNG")
        return dst_path, (width, height), resized.size


def prepare_input_image(input_path: Path, page_number: int | None, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        if page_number is None:
            raise ValueError("A page number is required for PDF preprocessing.")
        output_path = output_dir / f"{input_path.stem}.page-{page_number}.png"
        render_pdf_page_to_png(input_path, output_path, page_number)
        return output_path

    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        output_path = output_dir / f"{input_path.stem}.normalized.png"
        normalize_image_to_png(input_path, output_path)
        return output_path

    raise ValueError(f"Unsupported input type: {input_path.suffix}")


def get_pdf_page_count(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()
