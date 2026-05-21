from __future__ import annotations

import re
from dataclasses import dataclass, replace

from nb_omr.types import LintReport

DATA_ONLY_RE = re.compile(r"[0-9.]")


@dataclass
class SpineState:
    staff: str
    clef: str | None = None
    key: str | None = None
    meter: str | None = None


def sanitize_data_token(token: str) -> str:
    token = token.strip()
    if not token or token == ".":
        return "."
    if token.startswith(".") and len(token) > 1:
        return "."
    if re.fullmatch(r"[0-9.]+", token):
        return token + "r"
    return token


def classify_staff_from_clef(clef: str | None, fallback_index: int, total: int) -> str:
    if clef and clef.startswith("*clefF"):
        return "lower"
    if clef and clef.startswith("*clefG"):
        return "upper"
    return "lower" if fallback_index < max(total // 2, 1) else "upper"


def merge_staff_tokens(tokens: list[str]) -> str:
    sanitized = [sanitize_data_token(token) for token in tokens if token and token != "."]
    if not sanitized:
        return "."

    note_tokens = [token for token in sanitized if "r" not in token]
    if note_tokens:
        return " ".join(dict.fromkeys(note_tokens))

    return sanitized[0]


def normalize_to_ekern(transcription: str, lint_report: LintReport) -> str:
    lines = [line for line in transcription.splitlines() if line.strip()]
    if not lines:
        return "**kern\t**kern\n*-\t*-\n"

    initial_columns = max(len(lines[0].split("\t")), 2)
    spines = [
        SpineState("lower" if index < max(initial_columns // 2, 1) else "upper")
        for index in range(initial_columns)
    ]
    output_lines = ["**kern\t**kern"]
    emitted = {"clef": (None, None), "key": (None, None), "meter": (None, None)}

    def ensure_spine_count(target: int, line_number: int, line: str) -> None:
        nonlocal spines
        if target <= len(spines):
            return
        for index in range(len(spines), target):
            spines.append(SpineState("lower" if index < max(target // 2, 1) else "upper"))
        lint_report.add_issue(
            "warning",
            "expanded_spines",
            f"Expanded active spine count to {target} to match the data row.",
            line_number=line_number,
            context=line,
        )

    def current_staff_value(field_name: str, staff_name: str) -> str | None:
        values = [getattr(spine, field_name) for spine in spines if spine.staff == staff_name]
        for value in values:
            if value:
                return value
        return None

    def emit_interpretation(field_name: str) -> None:
        current = (
            current_staff_value(field_name, "lower"),
            current_staff_value(field_name, "upper"),
        )
        if current == emitted[field_name]:
            return
        if all(value is None for value in current):
            return
        left = current[0] or "*"
        right = current[1] or "*"
        output_lines.append(f"{left}\t{right}")
        emitted[field_name] = current

    def update_path_row(tokens: list[str], line_number: int, line: str) -> bool:
        nonlocal spines
        if all(token == "*-" for token in tokens):
            output_lines.append("*-\t*-")
            spines = []
            return True

        next_spines: list[SpineState] = []
        index = 0
        while index < len(tokens) and index < len(spines):
            token = tokens[index]
            state = spines[index]
            if token == "*^":
                next_spines.append(state)
                next_spines.append(replace(state))
                index += 1
                continue
            if token == "*v":
                group_end = index
                while group_end < len(tokens) and tokens[group_end] == "*v":
                    group_end += 1
                group = spines[index:group_end]
                if not group:
                    lint_report.add_issue(
                        "warning",
                        "orphan_join",
                        "Encountered *v without an active spine group.",
                        line_number=line_number,
                        context=line,
                    )
                else:
                    merged = group[0]
                    for candidate in group[1:]:
                        if merged.clef is None and candidate.clef is not None:
                            merged = replace(merged, clef=candidate.clef)
                        if merged.key is None and candidate.key is not None:
                            merged = replace(merged, key=candidate.key)
                        if merged.meter is None and candidate.meter is not None:
                            merged = replace(merged, meter=candidate.meter)
                    next_spines.append(merged)
                index = group_end
                continue
            if token == "*-":
                index += 1
                continue

            next_spines.append(state)
            index += 1

        while index < len(spines):
            next_spines.append(spines[index])
            index += 1

        if not next_spines:
            next_spines = [SpineState("lower"), SpineState("upper")]

        spines = next_spines
        return False

    for line_number, line in enumerate(lines, start=1):
        tokens = [token.strip() for token in line.split("\t")]
        if not tokens:
            continue

        if all(token.startswith("*") for token in tokens):
            ensure_spine_count(len(tokens), line_number, line)
            aligned = tokens + ["*"] * max(0, len(spines) - len(tokens))

            for index, token in enumerate(aligned[: len(spines)]):
                if token.startswith("*clef"):
                    spines[index].clef = token
                    spines[index].staff = classify_staff_from_clef(token, index, len(spines))
                elif token.startswith("*k["):
                    spines[index].key = token
                elif token.startswith("*M"):
                    spines[index].meter = token

            emit_interpretation("clef")
            emit_interpretation("key")
            emit_interpretation("meter")

            if any(token in {"*^", "*v", "*-", "*+"} for token in aligned):
                terminated = update_path_row(aligned, line_number, line)
                if terminated:
                    break
            continue

        if any(token.startswith("=") for token in tokens):
            bar_token = next((token for token in tokens if token.startswith("=")), "=")
            output_lines.append(f"{bar_token}\t{bar_token}")
            continue

        ensure_spine_count(len(tokens), line_number, line)
        aligned = tokens + ["."] * max(0, len(spines) - len(tokens))
        left_tokens: list[str] = []
        right_tokens: list[str] = []

        for index, token in enumerate(aligned[: len(spines)]):
            if token in {"", ".", "*"}:
                continue
            if token.startswith("*") or token.startswith("="):
                lint_report.add_issue(
                    "warning",
                    "embedded_control_token",
                    f"Skipped control token {token!r} inside a data row.",
                    line_number=line_number,
                    context=line,
                )
                continue
            if not DATA_ONLY_RE.search(token):
                lint_report.add_issue(
                    "warning",
                    "non_notelike_data",
                    f"Skipped token {token!r} because it does not look like note data.",
                    line_number=line_number,
                    context=line,
                )
                continue

            staff = spines[index].staff or classify_staff_from_clef(spines[index].clef, index, len(spines))
            if staff == "lower":
                left_tokens.append(token)
            else:
                right_tokens.append(token)

        output_lines.append(f"{merge_staff_tokens(left_tokens)}\t{merge_staff_tokens(right_tokens)}")

    if not output_lines[-1].startswith("*-"):
        output_lines.append("*-\t*-")

    return "\n".join(output_lines) + "\n"
