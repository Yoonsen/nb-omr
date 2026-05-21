# AGENTS.md

## Current State

This repository contains a minimal OMR prototype built around the Hugging Face Sheet Music Transformer model `PRAIG/smt-fp-grandstaff`.

The current flow is:

1. Accept a local `pdf` or image file.
2. Render PDF pages to PNG with `pymupdf` when needed.
3. Normalize image input to PNG.
4. Resize oversized pages to the SMT model limits from the loaded config.
5. Run inference with a local copy of the SMT model code.
6. Write raw, normalized, and reporting artifacts to `outputs/`.

## Preferred Runtime

- The preferred machine for this repository is currently the Linux workstation with an `NVIDIA RTX A6000` (48 GB VRAM).
- On that machine, GPU inference now works end to end with `uv` after pinning Linux installs to the PyTorch `cu121` wheels.
- A successful real run on `page-000.png` completed on `cuda:0` using `bfloat16` and finished quickly enough that this box should be treated as the default place to benchmark and extend inference work for now.
- `nvidia-smi` may still warn about an NVML driver/library mismatch on this machine; use `uv run python -c "import torch; print(torch.cuda.is_available())"` as the source of truth for whether PyTorch can use CUDA.

## Important Implementation Notes

- The prototype uses local copies of the upstream SMT modules in `smt_model/` and `data_augmentation/`.
- The current default model is `PRAIG/smt-fp-grandstaff`.
- The code now prefers `cuda`, then `mps`, then `cpu`.
- Oversized pages must be resized before inference. The model config for `PRAIG/smt-fp-grandstaff` reports `maxw=2100` and `maxh=2970`.
- A real run failed before this resize step was added because the page image exceeded those limits.
- The CLI is now a thin wrapper over the reusable `nb_omr/` pipeline modules.
- The raw model transcription should be treated as Humdrum-like text with unstable spine structure until it passes through the `nb_omr/ekern/` normalization layer.

## Dependency Notes

- `transformers` 5.x did not work with this setup.
- Keep `transformers` on the 4.x line for now. The project is currently pinned to `transformers>=4.49.0,<5`.
- The model loaded successfully after pinning to `transformers` 4.x.
- On Linux, `torch` and `torchvision` are pinned via `uv` to the PyTorch `cu121` index. Do not casually upgrade to a newer CUDA wheel set unless you also verify compatibility with the workstation's NVIDIA driver.

## Known Gaps

- The primary output is still the raw SMT transcription, which appears to be `**kern`/Humdrum-like or close to `ekern`.
- The pipeline now writes `normalized.ekrn`, `lint.json`, per-page reports, and batch summaries, but the normalization is still heuristic and should be treated as repair logic rather than a guaranteed faithful reconstruction.
- The MusicXML export now runs from an export-safe view of the normalized ekern, but that export is still best-effort.
- There is no evaluation harness yet.
- Logging is still minimal.
- There is not yet a strong quantitative benchmark or regression suite for musical correctness.

## Suggested Next Steps

1. Improve the `nb_omr/ekern/` spine state machine so staff assignment is more faithful on difficult pages.
2. Compare `music21` export against `hum2xml`/humlib once a stronger normalized ekern layer exists.
3. Add a small benchmark/evaluation harness that records runtime plus structural repair counts on representative pages.
4. Expand the Streamlit app for internal review workflows and musician feedback.
5. Only return to `mps` verification if Apple Silicon becomes a target again.

## Files Added So Far

- `main.py`: CLI entrypoint for PDF/image inference.
- `nb_omr/`: reusable preprocessing, inference, normalization, export, batch, and reporting pipeline modules.
- `smt_model/`: local SMT config and model implementation.
- `data_augmentation/`: minimal image-to-tensor helper and copied transform module.
- `streamlit_app.py`: internal inspection UI built on the same pipeline.
- `README.md`: basic usage notes.
- `pyproject.toml`: project dependencies.

## Commit Guidance For The Next Agent

- Avoid upgrading `transformers` beyond 4.x unless you are also updating the SMT integration.
- Do not commit large local test images or generated outputs unless the user explicitly wants fixtures in the repo.
- If inference is slow or appears stuck, first check whether multiple `uv run python main.py ...` processes are competing for the same machine resources.
- When working on inference performance, treat the Linux A6000 workstation as the preferred baseline environment.
- If CUDA suddenly appears unavailable in `uv`, check whether the environment drifted away from the pinned `cu121` wheels before changing application code.
- Keep the reusable logic in `nb_omr/` and avoid rebuilding business logic directly inside `main.py` or `streamlit_app.py`.
- Treat `normalized.ekrn`, lint reports, and batch reports as important meeting/debug artifacts when discussing failures with model developers.
