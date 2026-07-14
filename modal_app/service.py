"""Public CUDA-backed Modal ASGI deployment for PrivateText."""

from __future__ import annotations

import modal

from modal_app.runtime import ARTIFACTS_PATH, app, cuda_image, model_volume

service_image = cuda_image.add_local_dir("app/static", remote_path="/root/app/static")


@app.function(
    image=service_image,
    gpu="T4",
    timeout=60 * 10,
    volumes={ARTIFACTS_PATH: model_volume},
)
@modal.asgi_app()
def fastapi_app():
    from app.main import app as api

    return api
