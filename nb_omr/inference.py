from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import torch

from data_augmentation import convert_img_to_tensor
from smt_model import SMTModelForCausalLM

from .runtime import configure_runtime, format_dtype, resolve_autocast_dtype, resolve_device


def format_prediction(tokens: list[str]) -> str:
    return "".join(tokens).replace("<b>", "\n").replace("<s>", " ").replace("<t>", "\t")


class InferenceSession:
    def __init__(
        self,
        model_name: str,
        device_arg: str = "auto",
        dtype_arg: str = "auto",
        compile_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device_arg)
        self.autocast_dtype = resolve_autocast_dtype(self.device, dtype_arg)
        configure_runtime(self.device)
        self.model = SMTModelForCausalLM.from_pretrained(model_name).to(self.device)
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")
        self.model.eval()

        if "<bos>" not in self.model.w2i or "<eos>" not in self.model.i2w.values():
            raise RuntimeError(
                "Loaded SMT vocabulary does not look complete. "
                "The model config is missing expected <bos>/<eos> tokens."
            )

    @property
    def dtype_label(self) -> str:
        return format_dtype(self.autocast_dtype)

    @property
    def max_size(self) -> tuple[int, int]:
        return self.model.config.maxw, self.model.config.maxh

    def transcribe_prepared_image(self, prepared_path: Path) -> tuple[str, float]:
        image = cv2.imread(str(prepared_path))
        if image is None:
            raise RuntimeError(f"Could not read prepared input image: {prepared_path}")

        input_tensor = convert_img_to_tensor(image).unsqueeze(0).to(
            self.device, non_blocking=self.device.type == "cuda"
        )
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype is not None and self.device.type == "cuda"
            else nullcontext()
        )

        start = time.perf_counter()
        with torch.inference_mode(), autocast_context:
            predictions, _ = self.model.predict(input_tensor, convert_to_str=True)
        elapsed = time.perf_counter() - start
        return format_prediction(predictions), elapsed
