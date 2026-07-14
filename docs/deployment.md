# Training and deployment

PrivateText does not ship a model artifact in Git. The Modal Volume is the source of truth for evaluated checkpoints and the production TensorRT artifact.

```bash
# 1. Fine-tune on one deterministic public-data sample.
modal run modal_app/train.py

# 2. Compile and promote the returned run ID.
modal run modal_app/export.py --run-id <training-run-id>

# 3. Deploy the CUDA-backed FastAPI service.
modal deploy modal_app/service.py
```

The training job writes `runs/<run-id>/fp32`, `labels.json`, and `metrics.json`. The compiler adds `model-tensorrt-fp16.ts`, benchmarks baseline PyTorch FP32 against TensorRT FP16 on the same T4 GPU, then writes the `production.json` pointer. The service returns `503` until that pointer exists and all referenced artifact files are present.

For local container serving after copying an evaluated artifact directory to `./artifacts`:

```bash
docker build -t private-text .
docker run --rm --gpus all -p 8000:8000 -v "$PWD/artifacts:/models:ro" private-text
```

The service requires an NVIDIA GPU and matching CUDA/TensorRT runtime. Run the test suite on CPU; it does not load the TensorRT engine.
