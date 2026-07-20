# EdgePilot

A fully local, privacy-preserving multi-agent voice assistant with typed executors and adaptive recovery. No cloud APIs, no data ever leaves the machine.

Based on the mechanism in [AnovaX (arXiv:2607.15367)](https://arxiv.org/abs/2607.15367).

## Architecture

```
mic --> faster-whisper (STT) --> intent parse
    --> SLM planner (emits typed tool-call JSON)
    --> schema validation gate --fail--> adaptive recovery (re-plan)
    --> typed executor registry (dispatch)
    --> result --> piper (TTS) + local dashboard
```

### Core Mechanisms

**Typed Executors:** Every tool is registered with a strongly-typed Pydantic schema. The planner emits tool calls as structured JSON validated against the schema *before* execution. Malformed calls are rejected pre-execution.

**Adaptive Recovery:** On tool failure or validation rejection, a recovery controller re-plans with the error fed back as context, up to a bounded retry budget. Every recovery event is logged and metered with failure type classification (validation / execution / unknown_tool).

**Local SLM Planning:** A quantized small model (e.g. Qwen2.5 3B via Ollama) handles intent-to-plan. The planner is swappable behind an abstract interface.

**Voice Pipeline:** Full mic -> faster-whisper STT -> plan -> execute -> piper TTS -> speaker loop, with VAD-based silence detection. All models run on-device.

## Setup

Prerequisites:
- Python 3.12+
- [Ollama](https://ollama.com/) installed and running
- A small model pulled: `ollama pull qwen2.5:3b`
- Docker (for monitoring dashboard only)

```bash
# Text mode (no voice dependencies)
make setup
make run

# Voice mode (downloads ~200MB of models on first run)
make setup-voice
make run-voice

# API server
make serve
```

## Monitoring Dashboard

Local Prometheus + Grafana stack — all telemetry stays on your machine.

```bash
# Start the monitoring stack (requires Docker)
make dashboard

# Open in browser:
#   Grafana:    http://localhost:3000  (admin/admin)
#   Prometheus: http://localhost:9090

# Then start EdgePilot with the API server
make serve

# Stop monitoring
make dashboard-down
```

### Dashboard Panels

| Section | Panels |
|---------|--------|
| Validation Gate | Pass/reject rate, rejection %, rates over time |
| Adaptive Recovery | Success rate gauge, attempts/successes/exhausted counters, rates over time |
| Planner Performance | Tokens/sec, latency percentiles (p50/p95/p99), tool execution breakdown |
| End-to-End Latency | p50/p95 latency, request rate by status, active requests |
| Voice Pipeline | STT latency, real-time factor, TTS latency, audio duration |

## Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| STT | faster-whisper | Local speech-to-text |
| Planner | Ollama (GGUF models) | Local SLM for intent -> plan |
| Executors | Pydantic v2 | Typed tool schemas + validation |
| TTS | piper | Local text-to-speech |
| Audio I/O | sounddevice | Mic recording + speaker playback |
| API | FastAPI | localhost service |
| Metrics | Prometheus + Grafana | Local telemetry dashboard |
| Logging | Structured JSON | Machine-parseable recovery events |

## Development

```bash
make test       # run tests
make lint       # lint + format check
make ci         # lint + tests (CI)
make eval       # run 20-scenario eval suite
```

## Eval

```bash
python -m eval.harness              # mock planner (no Ollama needed)
python -m eval.harness --live       # live Ollama planner
```

Metrics captured:
- (a) % malformed tool calls caught at validation gate
- (b) Task completion rate with/without adaptive recovery
- (c) Planner tokens/sec on local device
- (d) End-to-end voice-to-action latency (p50/p95)

## License

MIT
