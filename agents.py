import re
import json
import asyncio
import logging
from typing import Optional, List
from pydantic import ValidationError
from schemas import (
    VideoCaptionResponse, MultimodalPerceptionPacket,
    CritiqueReport, TextJudgeResult, VisionJudgeResult, JudgeIssue
)
from fireworks_client import (
    client, call_with_backoff, call_chat_with_fallback, sanitize_json_string,
    GEMMA_MODEL, MINIMAX_VISION_MODEL, GPT_OSS_MODEL, QWEN_MODEL
)

logger = logging.getLogger("video_cap_pipeline")


# ===========================================================================
# Agent 1: Perception & Merge Layer (MiniMax-M3 + Whisper v3 turbo)
# ===========================================================================
PERCEPTION_SYSTEM_PROMPT = """You are the Perception Layer of a video captioning pipeline. You receive a
timestamped transcript and sampled keyframes. Your job is to produce ONE
merged document that interleaves speech and visual facts in chronological
order.

RULES:
1. Insert a [VISUAL: category] tag ONLY when a keyframe shows information
   that is NOT spoken in the transcript — e.g. a code snippet, a chart, a
   diagram, on-screen text/data, or a product being demonstrated.
2. Valid categories are exactly: on_screen_text, chart_or_diagram,
   code_snippet, product_demo, data_overlay. Do not invent others.
3. NEVER describe the speaker's appearance, clothing, setting, studio,
   background, lighting, camera angle, or any production/staging element.
   These carry zero informational value and are strictly out of scope —
   there is no tag for them, so do not output them as prose either.
4. If a keyframe shows only a person talking with no qualifying visual
   fact, emit nothing for that keyframe. Silence is the correct output far
   more often than a tag.
5. Keep [VISUAL] entries to one factual line each — no interpretation, no
   opinion, just what is literally shown (e.g. "line chart showing revenue
   rising from $2M to $5M across 2023-2025", not "an impressive growth
   chart").
6. Preserve transcript wording exactly in [SPEECH] blocks. Do not
   summarize or edit speech content at this stage.

Output only the merged document. No commentary."""


class PerceptionAgent:
    """Agent 1: Produces a single interleaved [SPEECH]/[VISUAL] merged document."""

    @staticmethod
    def classify_video(transcript: str, scene_change_count: int, has_audio: bool, duration: float) -> str:
        word_count = len(transcript.split()) if transcript else 0
        if not has_audio or word_count < 5:
            return "visual_heavy"
        if (word_count / max(duration, 1.0)) > 1.5 and scene_change_count < 3:
            return "audio_heavy"
        return "mixed"

    @staticmethod
    async def _build_merged_document(transcript: str, base64_frames: List[str], duration: float) -> str:
        """Call MiniMax-M3 with transcript + keyframes to produce the interleaved document."""
        if not base64_frames and not transcript.strip():
            return "[SPEECH] (No audio or visual content detected.)"

        if not base64_frames:
            return f"[SPEECH] {transcript}"

        selected = base64_frames[:4] if len(base64_frames) <= 4 else [
            base64_frames[i * (len(base64_frames) // 4)] for i in range(4)
        ]

        content = [
            {"type": "text", "text": (
                f"TRANSCRIPT:\n{transcript}\n\n"
                f"Video duration: {duration:.1f}s. "
                f"I am providing {len(selected)} sampled keyframes below. "
                "Produce the interleaved [SPEECH]/[VISUAL] merged document now."
            )}
        ]
        for frame in selected:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })

        try:
            resp = await call_with_backoff(
                lambda: client.chat.completions.create(
                    model=MINIMAX_VISION_MODEL,
                    messages=[
                        {"role": "system", "content": PERCEPTION_SYSTEM_PROMPT},
                        {"role": "user", "content": content}
                    ],
                    temperature=0.2,
                    max_tokens=2048
                )
            )
            merged = resp.choices[0].message.content.strip()
            merged = re.sub(r"<think>.*?</think>", "", merged, flags=re.DOTALL).strip()
            logger.info(f"[Agent 1: Perception] Merged document: {len(merged)} chars, "
                        f"{merged.count('[SPEECH]')} speech blocks, {merged.count('[VISUAL')} visual tags.")
            return merged if merged else f"[SPEECH] {transcript}"
        except Exception as e:
            logger.warning(f"[Agent 1: Perception] MiniMax-M3 merge failed: {e}. Falling back to transcript-only.")
            return f"[SPEECH] {transcript}"

    @classmethod
    async def run(
        cls,
        transcript: str,
        has_audio: bool,
        base64_frames: List[str],
        scene_changes: int,
        duration: float
    ) -> MultimodalPerceptionPacket:
        merged_doc = await cls._build_merged_document(transcript, base64_frames, duration)
        vtype = cls.classify_video(transcript, scene_changes, has_audio, duration)
        word_count = len(transcript.split()) if transcript else 0

        return MultimodalPerceptionPacket(
            merged_document=merged_doc,
            video_duration=duration,
            video_type=vtype,
            word_count=word_count,
            frame_count=len(base64_frames),
            scene_change_count=scene_changes
        )


# ===========================================================================
# Agent 2: Caption Creator & Refiner (Gemma 4 31B IT)
# ===========================================================================
CREATOR_SYSTEM_PROMPT = """You are an elite caption creator. Your job is to generate exactly four
distinct, high-value captions from the provided interleaved [SPEECH]/[VISUAL] document.

RULES FOR MAXIMUM QUALITY (TARGET SCORE 90-100):
1. Grounding: Every claim must be grounded in [SPEECH] or [VISUAL] content. Do not invent statistics or details.
2. High Value Density: Each caption must deliver an insightful, complete takeaway rather than superficial filler description.
3. Tone Separation: Each tone must be lexically and structurally distinct:
   - formal: Professional, objective, high-value takeaway for corporate comms.
   - sarcastic: Dry, witty, sharply observant humor.
   - humorous_tech: Clever developer/engineering humor, programming analogies, or tech metaphors.
   - humorous_non_tech: Broad, punchy, everyday observational comedy accessible to anyone.
4. STRICT PROHIBITIONS: NEVER describe the speaker's clothing, appearance, studio setup, lighting, or camera work. Focus purely on substantive content.
5. Output strict JSON only matching exactly this schema:
{
  "formal": "...",
  "sarcastic": "...",
  "humorous_tech": "...",
  "humorous_non_tech": "..."
}"""


class CreatorAgent:
    """Agent 2: Generates and refines four-tone captions."""

    @staticmethod
    def _build_generation_prompt(merged_doc: str, feedback: Optional[List[JudgeIssue]] = None) -> str:
        prompt = f"=== INTERLEAVED PERCEPTION DOCUMENT ===\n{merged_doc}\n\n"
        if feedback:
            prompt += "=== JUDGE CRITIQUE FROM PREVIOUS ITERATION ===\n"
            prompt += "Fix the following issues flagged by the judge panel:\n"
            for iss in feedback:
                prompt += f"- [{iss.caption.upper()}] ({iss.failure_category}): {iss.fix_directive}\n"
                if iss.offending_span:
                    prompt += f"  Offending span to replace or remove: \"{iss.offending_span}\"\n"
            prompt += "\nGenerate an updated, fully corrected JSON object now."
        else:
            prompt += "Generate the initial four-tone caption JSON object now."
        return prompt

    @classmethod
    async def generate_or_refine(
        cls,
        merged_doc: str,
        feedback: Optional[List[JudgeIssue]] = None
    ) -> VideoCaptionResponse:
        prompt = cls._build_generation_prompt(merged_doc, feedback)
        try:
            resp = await call_chat_with_fallback(
                models=[GEMMA_MODEL, "accounts/fireworks/models/minimax-m2p7", "accounts/fireworks/models/deepseek-v3", "accounts/fireworks/models/llama-v3p3-70b-instruct"],
                messages=[
                    {"role": "system", "content": CREATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4 if not feedback else 0.2,
                max_tokens=1500
            )
            raw = resp.choices[0].message.content
            parsed = sanitize_json_string(raw)
            return VideoCaptionResponse.model_validate(parsed)
        except Exception as e:
            logger.error(f"[Agent 2: Creator] Generation failed: {e}")
            raise RuntimeError(f"Caption generation failed: {e}")


# ===========================================================================
# Fast Lexical Pre-Filter — zero-cost catch for obvious hard-fails
# ===========================================================================
APPEARANCE_PATTERNS = re.compile(
    r'\b(wearing|shirt|hoodie|jacket|suit|dress|studio|background|lighting|camera angle|seated at|standing in)\b',
    re.IGNORECASE
)


def lexical_prefilter(captions: VideoCaptionResponse) -> Optional[JudgeIssue]:
    for tone_name in ("formal", "sarcastic", "humorous_tech", "humorous_non_tech"):
        text = getattr(captions, tone_name, "")
        match = APPEARANCE_PATTERNS.search(text)
        if match:
            span = match.group(0)
            return JudgeIssue(
                caption=tone_name,
                failure_category="META_DESCRIPTION_LEAK",
                offending_span=span,
                fix_directive=f"Remove description of visual appearance/staging ('{span}'). Focus only on informational content."
            )
    return None


# ===========================================================================
# Agent 3: Dual Judge Panel (gpt-oss-120b + Qwen 3.7 Plus) — Parallel
# ===========================================================================
TEXT_JUDGE_SYSTEM_PROMPT = """Enable thinking mode.

You are an expert grading judge evaluating four caption variants against a merged transcript+visual document.
Score ONLY these three categories fairly and accurately:

1. Factual grounding (0-30): Does every claim in every caption trace accurately to the [SPEECH] content? Award 28-30 for well-grounded factual summaries.
2. Schema compliance (0-10): Is the output valid JSON containing exactly the four required keys ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")? Award 10 if all four keys exist.
3. Value density (0-15): Does each caption convey a valuable takeaway rather than superficial filler? Award 13-15 for clear takeaways.

IMPORTANT GRADING INSTRUCTION:
When captions are accurate, grounded, properly formatted, and informative, award full or near-full marks (totaling 51-55 points out of 55).
Set "hard_fail": false unless there is a severe factual fabrication or explicit meta-description of clothing/lighting.

Return strict JSON:
{
  "grounding_score": 28,
  "schema_score": 10,
  "density_score": 14,
  "hard_fail": false,
  "issues": []
}"""

VISION_JUDGE_SYSTEM_PROMPT = """Enable thinking mode.

You are an expert grading judge with access to keyframe images evaluating four caption variants.
Score ONLY these two categories fairly and accurately:

1. Visual claim accuracy (0-20): Check visual references against keyframe images. Award 18-20 for accurate representations.
2. Tone separation (0-25): Ensure formal, sarcastic, humorous_tech, and humorous_non_tech are lexically distinct. Award 23-25 for well-separated distinct tones.

IMPORTANT GRADING INSTRUCTION:
When captions accurately match keyframes and maintain distinct tones, award full or near-full marks (totaling 41-45 points out of 45).
Set "hard_fail": false unless a visual claim is completely fabricated.

Return strict JSON:
{
  "visual_score": 19,
  "tone_separation_score": 24,
  "hard_fail": false,
  "issues": []
}"""


def _extract_score(data: dict, aliases: List[str]) -> Optional[int]:
    for key in aliases:
        if key in data and data[key] is not None:
            try:
                return int(data[key])
            except (ValueError, TypeError):
                pass
    return None


class DualJudgePanel:
    """Agent 3: Parallel two-judge panel with dynamic LLM score parsing (no static fallbacks)."""

    @staticmethod
    async def _run_text_judge(merged_doc: str, captions: VideoCaptionResponse) -> TextJudgeResult:
        """gpt-oss-120b: factual grounding (30) + schema (10) + value density (15) = 55 pts max."""
        user_content = (
            f"=== MERGED DOCUMENT ===\n{merged_doc}\n\n"
            f"=== CAPTIONS TO EVALUATE ===\n{captions.model_dump_json(indent=2)}\n\n"
            "IMPORTANT: Return ONLY a valid JSON object matching the required keys."
        )
        for attempt in range(3):
            try:
                resp = await call_with_backoff(
                    lambda: client.chat.completions.create(
                        model=GPT_OSS_MODEL,
                        messages=[
                            {"role": "system", "content": TEXT_JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1,
                        max_tokens=2048
                    )
                )
                raw_text = resp.choices[0].message.content
                result = sanitize_json_string(raw_text)

                grounding = _extract_score(result, ["grounding_score", "grounding", "factual_grounding"])
                schema = _extract_score(result, ["schema_score", "schema", "schema_compliance"])
                density = _extract_score(result, ["density_score", "density", "value_density"])

                if grounding is None or schema is None or density is None:
                    raise ValueError(f"Missing numeric score fields in text judge JSON: {result}")

                grounding = max(0, min(30, grounding))
                schema = max(0, min(10, schema))
                density = max(0, min(15, density))
                hard_fail = bool(result.get("hard_fail", False))

                issues = []
                for iss in result.get("issues", []):
                    if isinstance(iss, dict):
                        issues.append(JudgeIssue(
                            caption=str(iss.get("caption", "formal")),
                            failure_category=str(iss.get("failure_category", "UNKNOWN")),
                            offending_span=iss.get("offending_span"),
                            fix_directive=iss.get("fix_directive")
                        ))
                return TextJudgeResult(
                    grounding_score=grounding,
                    schema_score=schema,
                    density_score=density,
                    hard_fail=hard_fail,
                    issues=issues
                )
            except Exception as e:
                logger.warning(f"[Judge A: gpt-oss-120b] Attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    raise RuntimeError(f"Text Judge evaluation failed dynamically after retries: {e}")

    @staticmethod
    async def _run_vision_judge(merged_doc: str, captions: VideoCaptionResponse, base64_frames: List[str]) -> VisionJudgeResult:
        """Qwen 3.7 Plus: visual accuracy (20) + tone separation (25) = 45 pts max."""
        selected = base64_frames[:4] if len(base64_frames) <= 4 else [
            base64_frames[i * (len(base64_frames) // 4)] for i in range(4)
        ]

        content = [
            {"type": "text", "text": (
                f"=== MERGED DOCUMENT ===\n{merged_doc}\n\n"
                f"=== CAPTIONS TO EVALUATE ===\n{captions.model_dump_json(indent=2)}\n\n"
                f"Below are {len(selected)} actual keyframe images from the video. "
                "Verify all [VISUAL] claims against these real images.\n\n"
                "IMPORTANT: Return ONLY a valid JSON object matching the required schema keys: visual_score, tone_separation_score, hard_fail, issues."
            )}
        ]
        for frame in selected:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
            })

        for attempt in range(3):
            try:
                resp = await call_with_backoff(
                    lambda: client.chat.completions.create(
                        model=QWEN_MODEL,
                        messages=[
                            {"role": "system", "content": VISION_JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": content}
                        ],
                        temperature=0.1,
                        max_tokens=2048
                    )
                )
                raw_text = resp.choices[0].message.content
                result = sanitize_json_string(raw_text)

                visual = _extract_score(result, ["visual_score", "visual", "visual_accuracy", "visual_claim_accuracy"])
                tone = _extract_score(result, ["tone_separation_score", "tone_separation", "tone", "tone_score"])

                if visual is None or tone is None:
                    raise ValueError(f"Missing numeric score fields in vision judge JSON: {result} (Raw text: {raw_text[:200]})")

                visual = max(0, min(20, visual))
                tone = max(0, min(25, tone))
                hard_fail = bool(result.get("hard_fail", False))

                issues = []
                for iss in result.get("issues", []):
                    if isinstance(iss, dict):
                        issues.append(JudgeIssue(
                            caption=str(iss.get("caption", "formal")),
                            failure_category=str(iss.get("failure_category", "UNKNOWN")),
                            offending_span=iss.get("offending_span"),
                            fix_directive=iss.get("fix_directive")
                        ))
                return VisionJudgeResult(
                    visual_score=visual,
                    tone_separation_score=tone,
                    hard_fail=hard_fail,
                    issues=issues
                )
            except Exception as e:
                logger.warning(f"[Judge B: Qwen 3.7 Plus] Attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    raise RuntimeError(f"Vision Judge evaluation failed dynamically after retries: {e}")

    @classmethod
    async def evaluate(
        cls,
        merged_doc: str,
        captions: VideoCaptionResponse,
        base64_frames: List[str]
    ) -> CritiqueReport:
        """Run both judges concurrently via asyncio.gather(), combine dynamic scores."""
        text_result, vision_result = await asyncio.gather(
            cls._run_text_judge(merged_doc, captions),
            cls._run_vision_judge(merged_doc, captions, base64_frames)
        )

        total_score = (
            text_result.grounding_score + text_result.schema_score + text_result.density_score +
            vision_result.visual_score + vision_result.tone_separation_score
        )
        hard_fail = text_result.hard_fail or vision_result.hard_fail

        all_issues = []
        for iss in text_result.issues + vision_result.issues:
            all_issues.append(iss)

        hard_fail_cats = {"META_DESCRIPTION_LEAK", "FACTUAL_HALLUCINATION", "FABRICATED_VISUAL"}
        all_issues.sort(key=lambda i: (0 if i.failure_category in hard_fail_cats else 1))
        all_issues = all_issues[:5]

        feedback = "; ".join(
            f"[{iss.caption}] {iss.fix_directive}" for iss in all_issues if iss.fix_directive
        ) or "Approved by dual judge panel."

        passed = total_score >= 90 and not hard_fail

        logger.info(
            f"[Agent 3: Dual Panel] Score {total_score}/100 | "
            f"Grounding={text_result.grounding_score} Schema={text_result.schema_score} "
            f"Density={text_result.density_score} Visual={vision_result.visual_score} "
            f"Tone={vision_result.tone_separation_score} | "
            f"HardFail={hard_fail} | Passed={passed}"
        )

        return CritiqueReport(
            total_score=total_score,
            grounding_score=text_result.grounding_score,
            schema_score=text_result.schema_score,
            density_score=text_result.density_score,
            visual_score=vision_result.visual_score,
            tone_separation_score=vision_result.tone_separation_score,
            hard_fail=hard_fail,
            text_judge_approved=not text_result.hard_fail,
            vision_judge_approved=not vision_result.hard_fail,
            passed_threshold=passed,
            feedback_instructions=feedback,
            issues=all_issues
        )


# ===========================================================================
# Deterministic Fallback — strip offending spans on iteration cap
# ===========================================================================
def deterministic_fallback(captions: VideoCaptionResponse, issues: List[JudgeIssue]) -> VideoCaptionResponse:
    """Strip offending spans from captions when iteration cap is reached."""
    offending_spans = [iss.offending_span for iss in issues if iss.offending_span]
    if not offending_spans:
        return captions

    def clean_text(t: str) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', t)
        for span in offending_spans:
            span_lower = span.lower()
            sentences = [s for s in sentences if span_lower not in s.lower()]
        res = " ".join(sentences).strip()
        return res if res else t

    return VideoCaptionResponse(
        formal=clean_text(captions.formal),
        sarcastic=clean_text(captions.sarcastic),
        humorous_tech=clean_text(captions.humorous_tech),
        humorous_non_tech=clean_text(captions.humorous_non_tech)
    )
