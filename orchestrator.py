import time
import uuid
import asyncio
import logging
from typing import Optional, Dict, Any, List
from schemas import (
    AgenticWorkflowResponse, AgentExecutionTimings, CritiqueReport
)
from agents import (
    PerceptionAgent, CreatorAgent, DualJudgePanel,
    lexical_prefilter, deterministic_fallback
)
from fireworks_client import GEMMA_MODEL, MINIMAX_VISION_MODEL, GPT_OSS_MODEL, QWEN_MODEL

logger = logging.getLogger("video_cap_pipeline")

# In-memory live job store for real-time telemetry polling
JOBS_STORE: Dict[str, Dict[str, Any]] = {}


def create_job() -> str:
    job_id = str(uuid.uuid4())
    JOBS_STORE[job_id] = {
        "job_id": job_id,
        "status": "RUNNING",
        "current_stage": "agent1",
        "stage_message": "Agent 1 Processing: Extracting audio & sampling visual keyframes...",
        "agent1_sec": 0.0,
        "agent2_sec": 0.0,
        "agent3_sec": 0.0,
        "total_pipeline_sec": 0.0,
        "iteration": 0,
        "result": None,
        "error": None
    }
    return job_id


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    return JOBS_STORE.get(job_id)


def update_job_status(job_id: Optional[str], **kwargs):
    if not job_id or job_id not in JOBS_STORE:
        return
    JOBS_STORE[job_id].update(kwargs)


async def run_agentic_workflow(
    transcript: str,
    has_audio: bool,
    base64_frames: List[str],
    scene_changes: int,
    duration: float,
    max_iterations: int = 3,
    passing_score: int = 90,
    job_id: Optional[str] = None
) -> AgenticWorkflowResponse:
    """
    Orchestrates the 3-Agent Dual-Model Ensemble with real-time perf_counter telemetry.
    Terminates immediately if score >= 90 on any iteration.
    """
    t_pipeline_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("[ValueStream AI] Starting 3-Agent Dual-Model Ensemble")
    logger.info(f"[ValueStream AI] Models: Creator={GEMMA_MODEL} | Perception={MINIMAX_VISION_MODEL}")
    logger.info(f"[ValueStream AI] Judges: TextJudge={GPT_OSS_MODEL} | VisionJudge={QWEN_MODEL}")
    logger.info("=" * 60)

    # --- AGENT 1: Perception & Merge Layer ---
    update_job_status(
        job_id,
        current_stage="agent1",
        stage_message="Agent 1 Processing: Dual-stream perception extracting transcript & visual facts..."
    )
    t1_start = time.perf_counter()
    packet = await PerceptionAgent.run(transcript, has_audio, base64_frames, scene_changes, duration)
    merged_doc = packet.merged_document
    agent1_sec = round(time.perf_counter() - t1_start, 2)
    update_job_status(job_id, agent1_sec=agent1_sec)

    best_draft = None
    best_score = -1
    best_critique = None
    prior_feedback = None
    critique_history = []
    loop_count = 0
    agent2_sec = 0.0
    agent3_sec = 0.0

    # --- AGENT 2 <-> AGENT 3 Orchestration Loop ---
    while loop_count < max_iterations:
        loop_count += 1
        logger.info(f"\n--- [ValueStream AI Loop: Iteration {loop_count}/{max_iterations}] ---")

        # --- AGENT 2: Creator / Refiner ---
        update_job_status(
            job_id,
            current_stage="agent2",
            iteration=loop_count,
            stage_message=f"Agent 2 Processing: Crafting/refining 4-tone captions (Iteration {loop_count})..."
        )
        t2_start = time.perf_counter()
        draft = await CreatorAgent.generate_or_refine(merged_doc, prior_feedback)
        agent2_sec += round(time.perf_counter() - t2_start, 2)
        update_job_status(job_id, agent2_sec=round(agent2_sec, 2))

        prefilter_issue = lexical_prefilter(draft)
        if prefilter_issue:
            total_score = 0
            hard_fail = True
            critique = CritiqueReport(
                total_score=0,
                hard_fail=True,
                text_judge_approved=False,
                vision_judge_approved=False,
                passed_threshold=False,
                feedback_instructions=prefilter_issue.fix_directive or "Meta-description leak.",
                issues=[prefilter_issue]
            )
        else:
            # --- AGENT 3: Dual Judge Panel (Concurrent via asyncio.gather) ---
            update_job_status(
                job_id,
                current_stage="agent3",
                iteration=loop_count,
                stage_message=f"Agent 3 Processing: Parallel Dual-Judge evaluating Factual Grounding & Tone Separation..."
            )
            t3_start = time.perf_counter()
            critique = await DualJudgePanel.evaluate(merged_doc, draft, base64_frames)
            agent3_sec += round(time.perf_counter() - t3_start, 2)
            update_job_status(job_id, agent3_sec=round(agent3_sec, 2))

            total_score = critique.total_score
            hard_fail = critique.hard_fail

        # Preserve highest-scoring non-hard-fail draft across all iterations
        if total_score > best_score and not hard_fail:
            best_draft, best_score, best_critique = draft, total_score, critique

        status = "APPROVED" if (total_score >= passing_score and not hard_fail) else ("HARD_FAIL" if hard_fail else "REFINING")
        entry = (
            f"Iter {loop_count}: Score {total_score}/100 [{status}] | "
            f"TextJudge({'PASS' if critique.text_judge_approved else 'FAIL'}) "
            f"VisionJudge({'PASS' if critique.vision_judge_approved else 'FAIL'})"
        )
        if critique.issues:
            cats = ", ".join(set(i.failure_category for i in critique.issues))
            entry += f" | Issues: {cats}"
        entry += f" -> {critique.feedback_instructions[:120]}"
        critique_history.append(entry)
        logger.info(f"[Loop Result] {entry}")

        # Target score >= 90: terminate immediately on any iteration
        if total_score >= passing_score and not hard_fail:
            logger.info(f"[ValueStream AI] Consensus APPROVED! Score {total_score} >= {passing_score} on Iteration {loop_count}")
            best_draft, best_score, best_critique = draft, total_score, critique
            break

        # Only proceed if score < 90 AND iteration < max_iterations
        prior_feedback = critique.issues[:5] if critique.issues else None

    if best_draft is None:
        logger.warning("[ValueStream AI] Applying deterministic fallback on last draft.")
        best_draft = deterministic_fallback(draft, critique.issues)
        best_score = max(total_score, 50)
        best_critique = critique

    total_pipeline_sec = round(time.perf_counter() - t_pipeline_start, 2)

    timings = AgentExecutionTimings(
        agent1_sec=round(agent1_sec, 2),
        agent2_sec=round(agent2_sec, 2),
        agent3_sec=round(agent3_sec, 2),
        total_pipeline_sec=total_pipeline_sec
    )

    response = AgenticWorkflowResponse(
        captions=best_draft,
        iterations_required=loop_count,
        final_score=best_score,
        critique_history=critique_history,
        perception_telemetry=packet,
        latest_critique=best_critique,
        execution_timings=timings,
        agent1_sec=round(agent1_sec, 2),
        agent2_sec=round(agent2_sec, 2),
        agent3_sec=round(agent3_sec, 2),
        total_pipeline_sec=total_pipeline_sec
    )

    update_job_status(
        job_id,
        status="COMPLETED",
        current_stage="completed",
        stage_message="Pipeline execution completed successfully.",
        total_pipeline_sec=total_pipeline_sec,
        result=response.model_dump()
    )

    return response
