const form = document.querySelector("#redaction-form");
const sourceText = document.querySelector("#source-text");
const redactButton = document.querySelector("#redact-button");
const status = document.querySelector("#request-status");
const redactedText = document.querySelector("#redacted-text");
const entityList = document.querySelector("#entity-list");
const copyButton = document.querySelector("#copy-button");

const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;

function renderEntities(entities) {
  entityList.replaceChildren();
  if (!entities.length) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "No supported PII entities were detected in this text.";
    entityList.append(empty);
    return;
  }
  for (const entity of entities) {
    const row = document.createElement("li");
    const type = document.createElement("span");
    type.className = "entity-type";
    type.textContent = entity.type;
    const value = document.createElement("span");
    value.className = "entity-value";
    value.textContent = entity.text;
    value.title = entity.text;
    const confidence = document.createElement("span");
    confidence.textContent = formatPercent(entity.confidence);
    row.append(type, value, confidence);
    entityList.append(row);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = sourceText.value.trim();
  if (!text) return;
  redactButton.disabled = true;
  status.textContent = "Redacting…";
  try {
    const response = await fetch("/redact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The redaction request could not be completed.");
    redactedText.textContent = payload.redacted;
    renderEntities(payload.entities);
    copyButton.disabled = false;
    status.textContent = `${payload.entities.length} ${payload.entities.length === 1 ? "entity" : "entities"} redacted`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    redactButton.disabled = false;
  }
});

copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(redactedText.textContent);
  copyButton.textContent = "Copied";
  setTimeout(() => { copyButton.textContent = "Copy"; }, 1600);
});

async function loadMetrics() {
  try {
    const health = await fetch("/health");
    if (!health.ok || (await health.json()).status !== "ready") return;
    const response = await fetch("/metrics");
    if (!response.ok) return;
    const metrics = await response.json();
    const test = metrics.test || {};
    const runtime = metrics.optimization?.tensorrt_fp16_gpu || {};
    document.querySelector("#metric-f1").textContent = Number.isFinite(test.entity_f1) ? formatPercent(test.entity_f1) : "—";
    document.querySelector("#metric-latency").textContent = Number.isFinite(runtime.p95_ms) ? `${runtime.p95_ms} ms` : "—";
    document.querySelector("#metric-throughput").textContent = Number.isFinite(runtime.throughput_per_second) ? `${runtime.throughput_per_second}/s` : "—";
    document.querySelector("#metrics-status").textContent = `Recorded for promoted run ${metrics.run_id}.`;
  } catch (_) {
    // An untrained service deliberately keeps placeholders instead of claiming benchmark values.
  }
}

loadMetrics();
