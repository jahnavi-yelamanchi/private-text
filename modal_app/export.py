"""Compile a promoted PrivateText checkpoint directly with Torch-TensorRT."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median

from app.benchmarking import percentile
from modal_app.runtime import ARTIFACTS_PATH, app, cuda_image, model_volume


def _benchmark(callable_) -> dict[str, float]:
    import torch

    for _ in range(25):
        callable_()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(150):
        started = time.perf_counter()
        callable_()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1_000)
    return {
        "p50_ms": round(float(median(timings)), 3),
        "p95_ms": round(percentile(timings, 0.95), 3),
        "throughput_per_second": round(1_000 / float(median(timings)), 2),
    }


@app.function(image=cuda_image, gpu="T4", timeout=60 * 30, volumes={ARTIFACTS_PATH: model_volume})
def compile_and_promote(run_id: str) -> dict[str, object]:
    """Create one FP16 TensorRT TorchScript artifact and promote it for serving."""

    import torch
    import torch_tensorrt
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    run_path = Path(ARTIFACTS_PATH) / "runs" / run_id
    source_path = run_path / "fp32"
    metrics_path = run_path / "metrics.json"
    if not source_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"No complete FP32 training run exists for {run_id!r}.")
    tokenizer = AutoTokenizer.from_pretrained(source_path)
    model = AutoModelForTokenClassification.from_pretrained(source_path).eval().cuda()
    sample = tokenizer(
        "Email Ada at ada@example.com and call +1 415 555 0198.",
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = sample["input_ids"].cuda()
    attention_mask = sample["attention_mask"].cuda()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, input_ids, attention_mask):
            return self.wrapped_model(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapper = LogitsOnly(model).eval()
    traced = torch.jit.trace(wrapper, (input_ids, attention_mask), strict=False)
    compiled = torch_tensorrt.compile(
        traced,
        ir="ts",
        inputs=[
            torch_tensorrt.Input(min_shape=(1, 8), opt_shape=(1, 128), max_shape=(1, 256), dtype=torch.int32),
            torch_tensorrt.Input(min_shape=(1, 8), opt_shape=(1, 128), max_shape=(1, 256), dtype=torch.int32),
        ],
        enabled_precisions={torch.float16},
        # DistilBERT's traced graph contains a few shape constants represented
        # as int64/float64; TensorRT needs them narrowed during conversion.
        truncate_long_and_double=True,
    )
    artifact_name = "model-tensorrt-fp16.ts"
    artifact_path = run_path / artifact_name
    torch.jit.save(compiled, str(artifact_path))

    baseline = _benchmark(lambda: wrapper(input_ids, attention_mask))
    trt_input_ids = input_ids.to(dtype=torch.int32)
    trt_mask = attention_mask.to(dtype=torch.int32)
    tensorrt = _benchmark(lambda: compiled(trt_input_ids, trt_mask))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["optimization"] = {
        "pytorch_fp32_gpu": baseline,
        "tensorrt_fp16_gpu": {"size_bytes": artifact_path.stat().st_size, **tensorrt},
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (Path(ARTIFACTS_PATH) / "production.json").write_text(
        json.dumps({"run_id": run_id, "artifact": artifact_name, "runtime": "torch-tensorrt-fp16"}, indent=2),
        encoding="utf-8",
    )
    model_volume.commit()
    return {"run_id": run_id, "optimization": metrics["optimization"]}


@app.local_entrypoint()
def main(run_id: str) -> None:
    print(json.dumps(compile_and_promote.remote(run_id), indent=2))
