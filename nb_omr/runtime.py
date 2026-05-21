from __future__ import annotations

import torch


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_autocast_dtype(device: torch.device, dtype_arg: str) -> torch.dtype | None:
    if dtype_arg == "float32":
        return None

    if device.type != "cuda":
        if dtype_arg == "auto":
            return None
        raise ValueError(f"{dtype_arg} autocast is only supported on CUDA in this prototype.")

    if dtype_arg == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if dtype_arg == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 was requested, but this CUDA device does not support it.")
        return torch.bfloat16

    if dtype_arg == "float16":
        return torch.float16

    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def configure_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def format_dtype(dtype: torch.dtype | None) -> str:
    return "float32" if dtype is None else str(dtype).split(".")[-1]
