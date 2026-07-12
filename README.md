# ValueStream AI: Enterprise Multi-Agent Video Captioning Pipeline

**ValueStream AI** is an enterprise-grade, multi-agent video captioning studio built on **Fireworks AI**. It transforms raw video streams into high-impact, four-tone captions verified through an autonomous, self-improving collaborative agentic loop.

---

## 🚀 Project Overview & Multi-Agent Architecture

ValueStream AI replaces traditional single-pass captioning with a collaborative **3-Agent Dual-Model Ensemble**:

```
[ Raw Video + Audio ]
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ AGENT 1: Perception & Merge Layer                                      │
│  • Audio Engine: Whisper v3 turbo + Gemma ASR Correction               │
│  • Vision Engine: MiniMax-M3 (Capped at 4 keyframes, max 720p)         │
│  • Output: Interleaved chronological [SPEECH]/[VISUAL] document        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ AGENT 2: Creator & Refiner (Gemma 4 31B IT)                            │
│  • Generates strict 4-tone JSON captions:                              │
│    - formal | sarcastic | humorous_tech | humorous_non_tech            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│ AGENT 3A: Text Judge Panel        │   │ AGENT 3B: Vision Judge Panel      │
│  • Model: gpt-oss-120b            │   │  • Model: Qwen 3.7 Plus           │
│  • Factual Grounding (0-30 pts)   │   │  • Visual Claim Accuracy (0-20)   │
│  • Schema Compliance (0-10 pts)   │   │  • Tone Separation (0-25 pts)     │
│  • Value Density (0-15 pts)       │   │                                   │
└─────────────────┬─────────────────┘   └─────────────────┬─────────────────┘
                  │                                       │
                  └─────────────────┬─────────────────────┘
                                    │
                                    ▼
                      Score >= 90 or Max Iterations (3)?
                                    │
                      ┌─────────────┴─────────────┐
                      │ Yes                       │ No (< 90 & iter < 3)
                      ▼                           ▼
            [ Final Verified Output ]   [ Feedback Loop to Agent 2 ]
```

### Key Architectural Highlights
1. **Dynamic LLM Evaluation & True Concurrency**:
   - Judges evaluate in parallel using `asyncio.gather()`.
   - Dynamic Pydantic schema parsing ensures exact scores from `gpt-oss-120b` and `Qwen 3.7 Plus`.
   - Automatic retries handle any LLM formatting anomalies without falling back to static mock scores.
2. **Real-Time Live Timers & Telemetry**:
   - Server-side `time.perf_counter()` records exact execution durations for Agent 1 (`agent1_sec`), Agent 2 (`agent2_sec`), Agent 3 (`agent3_sec`), and total pipeline duration (`total_pipeline_sec`).
   - Frontend polls `/job_status/{job_id}` in real time for accurate badge status updates.
3. **High-Performance Video Ingestion**:
   - Caps visual extraction at 4 evenly spaced keyframes downscaled to max 720p.
   - Eliminates POS_FRAMES seek lag for ultra-fast OpenCV video processing.
4. **Autonomous Self-Correcting Loop**:
   - Executes up to `MAX_ITERATIONS = 3`.
   - Target consensus threshold is `>= 90/100`. The loop terminates immediately upon achieving `score >= 90`.
   - Tracks and preserves the `best_draft` across iterations.

---

## 🤖 Models & Deployment Setup

> [!IMPORTANT]
> **Deploy Models Before Start**: Ensure your Fireworks AI account has deployed or enabled access to **Gemma 4 31B IT** (`accounts/fireworks/models/gemma-4-31b-it`) and the judge endpoints before running the pipeline.

ValueStream AI leverages exclusively enterprise models hosted on **Fireworks AI**:
- **Whisper v3 turbo** (`accounts/fireworks/models/whisper-v3-turbo`): High-speed ASR speech transcription.
- **MiniMax-M3** (`accounts/fireworks/models/minimax-m3`): Multimodal perception & chronological document merge.
- **Gemma 4 31B IT** (`accounts/fireworks/models/gemma-4-31b-it`): Core caption creator & iterative refiner.
- **gpt-oss-120b** (`accounts/fireworks/models/gpt-oss-120b`): Text Judge evaluating factual grounding, schema compliance, and value density.
- **Qwen 3.7 Plus** (`accounts/fireworks/models/qwen-3.7-plus`): Multimodal Vision Judge verifying visual claims against real keyframes and tone separation.

---

## ⚙️ Environment Configuration (`.env`)

Create a `.env` file in the project root directory:

```env
# Fireworks AI API Key
FIREWORKS_API_KEY=your_fireworks_api_key_here

# Optional Application Settings
PORT=8000
```

---

## 🏃 Quickstart Guide

### Option A: Run Locally via Python

1. **Clone & Install Dependencies**:
   ```bash
   git clone <repo_url>
   cd video_cap
   python -m venv venv
   # Windows: venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Access the Studio UI**:
   Open your browser to `http://localhost:8000`.

---

### Option B: Run via Docker Compose

1. **Configure `.env`**:
   Ensure `.env` contains your `FIREWORKS_API_KEY`.

2. **Launch with Docker Compose**:
   ```bash
   docker-compose up --build -d
   ```

3. **Access the Studio UI**:
   Navigate to `http://localhost:8000`.

---

## 📡 API Endpoints

- **`POST /api/v1/caption`**: Standard synchronous endpoint returning 4-tone captions.
- **`POST /api/v1/caption/agentic`**: Returns full 3-Agent ensemble telemetry, judge critique report, and exact execution timings.
- **`POST /analyze_job`**: Asynchronous job launcher returning `{ "job_id": "..." }`.
- **`GET /job_status/{job_id}`**: Real-time polling endpoint returning live agent stage, status, and precise `perf_counter` execution metrics.
