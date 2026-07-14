"""Shared Modal application, Volume, and CUDA image configuration."""

from __future__ import annotations

import modal

APP_NAME = "private-text-pii-redaction"
VOLUME_NAME = "private-text-artifacts"
ARTIFACTS_PATH = "/models"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# This image intentionally pins the PyTorch/CUDA pair expected by the matching
# Torch-TensorRT release. TensorRT is never installed or required for local tests.
cuda_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", add_python="3.11")
    .pip_install(
        "accelerate>=1.0,<2.0",
        "datasets>=3.0,<4.0",
        "fastapi>=0.115,<1.0",
        "scikit-learn>=1.5,<2.0",
        "seqeval>=1.2,<2.0",
        "torch-tensorrt==2.5.0",
        "transformers>=4.46,<5.0",
    )
    .add_local_python_source("app", "modal_app")
)
