# nb-omr

Minimal OMR prototype for experimenting with the Hugging Face `PRAIG/smt-fp-grandstaff` Sheet Music Transformer model.

## Usage

Install dependencies:

```bash
uv sync
```

On Linux, the project pins `torch` and `torchvision` to the PyTorch CUDA 12.1 wheels so the
environment stays compatible with older NVIDIA driver stacks like the one on this workstation.

Run on an image:

```bash
uv run python main.py path/to/page.png
```

Run on a PDF page:

```bash
uv run python main.py path/to/score.pdf --page 1
```

Run on every page in a PDF:

```bash
uv run python main.py path/to/score.pdf --all-pages
```

Run on every supported image/PDF in a directory:

```bash
uv run python main.py path/to/folder
```

On a Linux machine with an NVIDIA GPU, the script will prefer `cuda` automatically. You can also
select the device and inference precision explicitly:

```bash
uv run python main.py path/to/score.pdf --page 1 --device cuda:0 --dtype auto
```

For repeated runs on the same GPU machine, you can try `--compile` to trade slower startup for
faster steady-state inference:

```bash
uv run python main.py path/to/score.pdf --page 1 --device cuda:0 --dtype auto --compile
```

## Artifacts

For each processed page, the pipeline writes:

- the rendered and resized page image
- the raw SMT transcription as `.txt`
- the raw kern-like transcription as `.raw.krn`
- a normalized two-staff Humdrum file as `.normalized.ekrn`
- an export-safe Humdrum view as `.export.krn`
- a per-page lint report as `.lint.json`
- a per-page summary as `.report.json`
- a best-effort `.musicxml` export when conversion succeeds

For batch runs, the pipeline also writes:

- `outputs/batch-summary.json`
- `outputs/meeting-report.md`

## Streamlit

Launch the internal review app with:

```bash
uv run streamlit run streamlit_app.py
```

The app uses the same pipeline as the CLI and is meant for quick inspection of the input image,
raw transcription, normalized `ekern`, lint warnings, and generated MusicXML artifacts.
