import os
import re
import json
import asyncio
import logging
from typing import List
from openai import AsyncOpenAI, RateLimitError
from pydantic import ValidationError

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)
logger = logging.getLogger("video_cap_pipeline")

from contextvars import ContextVar
from typing import Optional

# ---------------------------------------------------------------------------
# Fireworks AI Configuration — Secure Dynamic API Key Management
# ---------------------------------------------------------------------------
_request_api_key: ContextVar[Optional[str]] = ContextVar("request_api_key", default=None)

def set_current_api_key(api_key: Optional[str]):
    """Set the API key for the current request context."""
    _request_api_key.set(api_key if (api_key and api_key.strip()) else None)

def get_client() -> AsyncOpenAI:
    key = _request_api_key.get() or os.getenv("FIREWORKS_API_KEY")
    if not key:
        raise ValueError("Fireworks AI API Key not provided. Please enter your API Key in the UI.")
    return AsyncOpenAI(
        api_key=key,
        base_url="https://api.fireworks.ai/inference/v1"
    )

class _ClientProxy:
    @property
    def chat(self):
        return get_client().chat
    @property
    def audio(self):
        return get_client().audio
    @property
    def models(self):
        return get_client().models

client = _ClientProxy()

# Model Constants
# Agent 2 (Value Creator) — Gemma 4 31B IT
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "accounts/fireworks/models/gemma-4-31b-it")
# Agent 1 (Perception Layer — Vision)
MINIMAX_VISION_MODEL = os.getenv("MINIMAX_VISION_MODEL", "accounts/fireworks/models/minimax-m3")
# Agent 1 (Perception Layer — Audio)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "accounts/fireworks/models/whisper-v3-turbo")
# Agent 3 — Text Judge (factual grounding, schema, value density — 55 pts)
GPT_OSS_MODEL = os.getenv("GPT_OSS_MODEL", "accounts/fireworks/models/gpt-oss-120b")
# Agent 3 — Multimodal Judge (visual accuracy, tone separation — 45 pts)
QWEN_MODEL = os.getenv("QWEN_MODEL", "accounts/fireworks/models/qwen3p7-plus")

# Concurrency limiter
api_semaphore = asyncio.Semaphore(5)


# ---------------------------------------------------------------------------
# JSON Sanitization Middleware
# ---------------------------------------------------------------------------
def sanitize_json_string(raw: str) -> dict:
    """
    Self-correcting JSON parser. Handles common LLM output failures:
    1. Removes <think>...</think> reasoning blocks
    2. Conversational text/reasoning before or after JSON
    3. Markdown code fences
    4. Misspelled/hyphenated keys
    5. Trailing commas
    """
    # Remove reasoning blocks from thinking models (DeepSeek, Qwen, etc.)
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    key_fixes = {
        r'"humorous[\s_-]?tech"': '"humorous_tech"',
        r'"humorous[\s_-]?non[\s_-]?tech"': '"humorous_non_tech"',
    }
    for pattern, replacement in key_fixes.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Exponential Backoff Wrapper
# ---------------------------------------------------------------------------
async def call_with_backoff(coro_factory, max_retries=5):
    for attempt in range(max_retries):
        try:
            async with api_semaphore:
                return await coro_factory()
        except RateLimitError:
            wait = min(2 ** attempt, 16)
            logger.warning(f"[Rate-Limit] 429 received. Backing off {wait}s (attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(wait)
    async with api_semaphore:
        return await coro_factory()


async def call_chat_with_fallback(models: List[str], **kwargs):
    """
    Attempts chat completion across a candidate list of model IDs.
    If a model returns 404 (NOT_FOUND / undeployed model), automatically falls back to the next model.
    """
    last_error = None
    for idx, model_id in enumerate(models):
        try:
            return await call_with_backoff(
                lambda m=model_id: client.chat.completions.create(model=m, **kwargs)
            )
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "NOT_FOUND" in err_str or "not found" in err_str.lower():
                logger.warning(f"[Model Fallback] Model '{model_id}' returned 404/NOT_FOUND. Falling back to next model...")
                last_error = e
                continue
            raise e
    if last_error:
        raise last_error


# ---------------------------------------------------------------------------
# Audio Transcription (Whisper on Fireworks AI)
# ---------------------------------------------------------------------------
async def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio via Whisper. Returns empty string on failure."""
    try:
        with open(audio_path, "rb") as file:
            audio_bytes = file.read()
        result = await call_with_backoff(
            lambda: client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), audio_bytes),
                model=WHISPER_MODEL,
            )
        )
        return result.text
    except Exception as e:
        logger.warning(f"[Audio Transcription] Whisper unavailable or failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# ASR Error Correction Layer
# ---------------------------------------------------------------------------
async def clean_transcript(raw_transcript: str) -> str:
    if not raw_transcript.strip():
        return ""
    system_prompt = (
        "You are an expert audio transcription editor. Fix minor errors, fill gaps, "
        "and make the transcript grammatically coherent. Do NOT hallucinate or add new context. "
        "Output ONLY the cleaned transcript text, nothing else."
    )
    try:
        result = await call_chat_with_fallback(
            models=[GEMMA_MODEL, "accounts/fireworks/models/deepseek-v3", "accounts/fireworks/models/llama-v3p3-70b-instruct"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Raw Transcript:\n{raw_transcript}"}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"[ASR Correction] Failed: {e}. Returning raw transcript.")
        return raw_transcript
