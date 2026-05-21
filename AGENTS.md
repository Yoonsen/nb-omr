# AGENTS.md

## Current State

This repository contains a minimal OMR prototype built around the Hugging Face Sheet Music Transformer model `PRAIG/smt-fp-grandstaff`.

The current flow is:

1. Accept a local `pdf` or image file.
2. Render PDF pages to PNG with `pymupdf` when needed.
3. Normalize image input to PNG.
4. Resize oversized pages to the SMT model limits from the loaded config.
5. Run inference with a local copy of the SMT model code.
6. Write the raw SMT transcription to `outputs/`.

## Important Implementation Notes

- The prototype uses local copies of the upstream SMT modules in `smt_model/` and `data_augmentation/`.
- The current default model is `PRAIG/smt-fp-grandstaff`.
- The code now prefers `cuda`, then `mps`, then `cpu`.
- Oversized pages must be resized before inference. The model config for `PRAIG/smt-fp-grandstaff` reports `maxw=2100` and `maxh=2970`.
- A real run failed before this resize step was added because the page image exceeded those limits.

## Dependency Notes

- `transformers` 5.x did not work with this setup.
- Keep `transformers` on the 4.x line for now. The project is currently pinned to `transformers>=4.49.0,<5`.
- The model loaded successfully after pinning to `transformers` 4.x.

## Known Gaps

- Output is still the raw SMT token transcription, not `MusicXML`.
- There is no batch mode for multi-page PDFs yet.
- There is no evaluation harness yet.
- Logging is still minimal.

## Suggested Next Steps

1. Verify that long runs on Apple Silicon actually use `mps` end to end.
2. Add clearer startup logging for selected device, original image size, and resized image size.
3. Add multi-page PDF processing.
4. Investigate post-processing from SMT tokens to a more useful notation format.
5. Benchmark on a larger NVIDIA GPU machine.

## Files Added So Far

- `main.py`: CLI entrypoint for PDF/image inference.
- `smt_model/`: local SMT config and model implementation.
- `data_augmentation/`: minimal image-to-tensor helper and copied transform module.
- `README.md`: basic usage notes.
- `pyproject.toml`: project dependencies.

## Commit Guidance For The Next Agent

- Avoid upgrading `transformers` beyond 4.x unless you are also updating the SMT integration.
- Do not commit large local test images or generated outputs unless the user explicitly wants fixtures in the repo.
- If inference is slow or appears stuck, first check whether multiple `uv run python main.py ...` processes are competing for the same machine resources.
