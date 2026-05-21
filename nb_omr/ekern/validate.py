from __future__ import annotations

from music21 import converter

from nb_omr.types import LintReport, ValidationReport


def validate_normalized_ekern(ekern_text: str, lint_report: LintReport) -> ValidationReport:
    lines = [line for line in ekern_text.splitlines() if line.strip()]
    issues: list[str] = []

    if not lines:
        issues.append("Normalized ekern output is empty.")
        return ValidationReport(status="failed_validation", issues=issues, measure_count=0, parsable=False)

    if lines[0] != "**kern\t**kern":
        issues.append("Normalized output does not begin with a two-staff **kern header.")

    if lines[-1] != "*-\t*-":
        issues.append("Normalized output does not end with a two-staff termination row.")

    bad_width_lines = [
        index for index, line in enumerate(lines, start=1) if len(line.split("\t")) != 2
    ]
    if bad_width_lines:
        issues.append(f"Normalized output has non-rectangular rows at lines: {bad_width_lines[:10]}")

    unresolved_path_ops = [
        index
        for index, line in enumerate(lines, start=1)
        if any(token in {"*^", "*v", "*+"} for token in line.split("\t"))
    ]
    if unresolved_path_ops:
        issues.append(
            f"Normalized output still contains unresolved spine-path tokens at lines: {unresolved_path_ops[:10]}"
        )

    measure_count = sum(1 for line in lines if line.startswith("="))
    parsable = False
    try:
        converter.parseData(ekern_text, format="humdrum")
        parsable = True
    except Exception as exc:
        issues.append(f"music21 could not parse the normalized output: {exc}")

    if issues:
        status = "failed_validation" if not parsable else "repaired_with_warnings"
    elif lint_report.warning_count or lint_report.error_count:
        status = "repaired_with_warnings"
    else:
        status = "ok"

    return ValidationReport(status=status, issues=issues, measure_count=measure_count, parsable=parsable)
