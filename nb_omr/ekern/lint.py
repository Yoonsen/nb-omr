from __future__ import annotations

import re

from nb_omr.types import LintReport

KNOWN_INTERPRETATION_PREFIXES = ("*clef", "*M", "*k[", "*met", "*MM", "**")
NOTE_LIKE_RE = re.compile(r"^[0-9.]+[A-Ga-gr#\-\[\]_:JLjknq>\\ ]*$")


def lint_raw_transcription(transcription: str) -> LintReport:
    lines = [line for line in transcription.splitlines() if line.strip()]
    report = LintReport(line_count=len(lines))
    if not lines:
        report.add_issue("error", "empty_output", "The transcription is empty.")
        return report

    column_counts = [len(line.split("\t")) for line in lines]
    report.initial_columns = column_counts[0]
    report.min_columns = min(column_counts)
    report.max_columns = max(column_counts)

    if not any(token.startswith("**") for token in lines[0].split("\t")):
        report.add_issue(
            "warning",
            "missing_exclusive_interpretation",
            "The transcription starts without a Humdrum exclusive interpretation.",
            line_number=1,
            context=lines[0],
        )

    previous_columns = column_counts[0]
    for index, line in enumerate(lines, start=1):
        cols = [col.strip() for col in line.split("\t")]
        is_interpretation = all(col.startswith("*") for col in cols)
        is_barline = all(col.startswith("=") or col == "." for col in cols)
        has_interpretation = any(col.startswith("*") for col in cols)
        has_barline = any(col.startswith("=") for col in cols)
        has_data = any(not col.startswith("*") and not col.startswith("=") for col in cols)

        if index > 1 and len(cols) != previous_columns and not (is_interpretation or has_interpretation):
            report.add_issue(
                "warning",
                "column_jump",
                f"Column count changed from {previous_columns} to {len(cols)} outside an interpretation row.",
                line_number=index,
                context=line,
            )

        if has_interpretation and has_data:
            report.add_issue(
                "warning",
                "mixed_record",
                "The line mixes interpretation and data tokens.",
                line_number=index,
                context=line,
            )

        if has_barline and has_data:
            report.add_issue(
                "warning",
                "mixed_barline_data",
                "The line mixes barlines and data tokens.",
                line_number=index,
                context=line,
            )

        if not is_interpretation and not is_barline:
            for col in cols:
                if col in {".", "*", ""}:
                    continue
                if col.startswith("*") or col.startswith("="):
                    continue
                if not NOTE_LIKE_RE.match(col):
                    report.add_issue(
                        "warning",
                        "suspicious_token",
                        f"The token {col!r} does not look like a note/rest token.",
                        line_number=index,
                        context=line,
                    )

        if is_interpretation:
            for col in cols:
                if col == "*":
                    continue
                if col in {"*^", "*v", "*-", "*+"}:
                    continue
                if not col.startswith(KNOWN_INTERPRETATION_PREFIXES):
                    report.add_issue(
                        "warning",
                        "unknown_interpretation",
                        f"The interpretation {col!r} is not recognized by the current normalizer.",
                        line_number=index,
                        context=line,
                    )

        previous_columns = len(cols)

    if not any(line.replace("\t", "") == "*-" * report.initial_columns for line in lines[-3:]):
        if "*-" not in lines[-1]:
            report.add_issue(
                "warning",
                "missing_termination",
                "The transcription does not end with a clear *- termination row.",
                line_number=len(lines),
                context=lines[-1],
            )

    return report
