# nb-omr

Minimal OMR prototype for experimenting with the Hugging Face `PRAIG/smt-fp-grandstaff` Sheet Music Transformer model.

## Usage

Install dependencies:

```bash
uv sync
```

Run on an image:

```bash
uv run python main.py path/to/page.png
```

Run on a PDF page:

```bash
uv run python main.py path/to/score.pdf --page 1
```

The script writes the rendered input image and the raw SMT transcription to `outputs/`.
