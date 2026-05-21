from __future__ import annotations

import re
from pathlib import Path

from music21 import converter


def simplify_export_duration(token: str) -> str:
    match = re.match(r"^(\d+)(.*)$", token)
    if not match:
        return token

    duration_value = int(match.group(1))
    suffix = match.group(2)
    allowed = [1, 2, 4, 8, 16, 32, 64]
    best = min(allowed, key=lambda candidate: abs(candidate - duration_value))
    return f"{best}{suffix}"


def build_musicxml_export_source(normalized_ekern: str) -> str:
    output_lines: list[str] = []
    for line in normalized_ekern.splitlines():
        if not line:
            continue
        if line.startswith("*") or line.startswith("="):
            output_lines.append(line)
            continue

        cols = line.split("\t")
        export_cols: list[str] = []
        for col in cols:
            if col == ".":
                export_cols.append(col)
                continue
            export_tokens = [simplify_export_duration(token) for token in col.split(" ") if token]
            export_cols.append(" ".join(export_tokens) if export_tokens else ".")
        output_lines.append("\t".join(export_cols))

    return "\n".join(output_lines) + "\n"


def convert_normalized_ekern_to_musicxml(
    normalized_ekern: str,
    musicxml_output_path: Path,
    export_source_path: Path | None = None,
) -> None:
    export_source = build_musicxml_export_source(normalized_ekern)
    if export_source_path is not None:
        export_source_path.write_text(export_source, encoding="utf-8")

    score = converter.parseData(export_source, format="humdrum")
    score.write("musicxml", fp=str(musicxml_output_path))
