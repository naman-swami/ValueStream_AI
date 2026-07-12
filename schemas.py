from pydantic import BaseModel, Field
from typing import Optional, List


class VideoCaptionResponse(BaseModel):
    """Strict output schema: four distinct caption tones."""
    formal: str = Field(..., description="Professional, objective, precise caption.")
    sarcastic: str = Field(..., description="Witty, dry humor, slightly mocking caption.")
    humorous_tech: str = Field(..., description="Inside jokes for programmers, tech jargon humor caption.")
    humorous_non_tech: str = Field(..., description="Broad, highly relatable everyday comedy caption.")


class MultimodalPerceptionPacket(BaseModel):
    """Agent 1 output: interleaved merged document + video telemetry."""
    merged_document: str = Field(..., description="Interleaved [SPEECH]/[VISUAL] merged document.")
    video_duration: float = Field(..., description="Total duration of the video in seconds.")
    video_type: str = Field(..., description="Classified type: 'visual_heavy', 'audio_heavy', or 'mixed'.")
    word_count: int = Field(..., description="Total word count of the audio transcript.")
    frame_count: int = Field(default=0, description="Number of sampled keyframes.")
    scene_change_count: int = Field(default=0, description="Number of visual scene changes detected.")

    @property
    def context(self):
        """Backwards-compatibility shim for UI code referencing context.visual_scene_descriptions."""
        class _Compat:
            def __init__(self, doc):
                self.visual_scene_descriptions = doc
                self.primary_content = doc
        return _Compat(self.merged_document)


class JudgeIssue(BaseModel):
    """Single issue flagged by a judge."""
    caption: str = Field(..., description="Which caption tone was flagged: formal, sarcastic, humorous_tech, humorous_non_tech")
    failure_category: str = Field(..., description="Category of failure.")
    offending_span: Optional[str] = Field(default=None, description="The specific text that failed.")
    fix_directive: Optional[str] = Field(default=None, description="1-sentence directive to correct Agent 2.")


class TextJudgeResult(BaseModel):
    """gpt-oss-120b text judge output: factual grounding + schema + value density (55 pts)."""
    grounding_score: int = Field(..., description="Factual grounding score 0-30.")
    schema_score: int = Field(..., description="JSON schema compliance score 0-10.")
    density_score: int = Field(..., description="Value density / hook strength 0-15.")
    hard_fail: bool = Field(default=False, description="True if any hard-fail condition detected.")
    issues: List[JudgeIssue] = Field(default_factory=list, description="Specific issues flagged.")


class VisionJudgeResult(BaseModel):
    """Qwen 3.7 Plus multimodal judge output: visual accuracy + tone separation (45 pts)."""
    visual_score: int = Field(..., description="Visual claim accuracy 0-20.")
    tone_separation_score: int = Field(..., description="Tone separation quality 0-25.")
    hard_fail: bool = Field(default=False, description="True if any hard-fail condition detected.")
    issues: List[JudgeIssue] = Field(default_factory=list, description="Specific issues flagged.")


class CritiqueReport(BaseModel):
    """Combined output from the dual judge panel."""
    total_score: int = Field(..., description="Combined score out of 100.")
    grounding_score: int = Field(default=0, description="Factual grounding 0-30 (gpt-oss-120b).")
    schema_score: int = Field(default=0, description="Schema compliance 0-10 (gpt-oss-120b).")
    density_score: int = Field(default=0, description="Value density 0-15 (gpt-oss-120b).")
    visual_score: int = Field(default=0, description="Visual accuracy 0-20 (Qwen 3.7 Plus).")
    tone_separation_score: int = Field(default=0, description="Tone separation 0-25 (Qwen 3.7 Plus).")
    hard_fail: bool = Field(default=False, description="True if either judge flagged a hard-fail.")
    text_judge_approved: bool = Field(default=True, description="gpt-oss-120b passed.")
    vision_judge_approved: bool = Field(default=True, description="Qwen 3.7 Plus passed.")
    passed_threshold: bool = Field(default=False, description="True if total_score >= 90 and not hard_fail.")
    feedback_instructions: str = Field(default="", description="Merged fix directives for Agent 2.")
    issues: List[JudgeIssue] = Field(default_factory=list, description="Merged issue list from both judges.")

    @property
    def judge_a_approved(self) -> bool:
        return self.text_judge_approved

    @property
    def judge_b_approved(self) -> bool:
        return self.vision_judge_approved

    @property
    def overall_score(self) -> int:
        return self.total_score

    @property
    def structured_critique(self):
        """Backwards-compat shim for UI code expecting structured_critique."""
        class _Compat:
            def __init__(self, report):
                self.approved = report.passed_threshold
                self.failure_category = report.issues[0].failure_category if report.issues else None
                self.offending_span = report.issues[0].offending_span if report.issues else None
                self.fix_directive = report.feedback_instructions
        return _Compat(self)


class AgentExecutionTimings(BaseModel):
    """Real-time server-side execution timings."""
    agent1_sec: float = Field(default=0.0, description="Exact execution seconds for Agent 1")
    agent2_sec: float = Field(default=0.0, description="Exact execution seconds for Agent 2")
    agent3_sec: float = Field(default=0.0, description="Exact execution seconds for Agent 3")
    total_pipeline_sec: float = Field(default=0.0, description="Total pipeline execution duration in seconds")


class AgenticWorkflowResponse(BaseModel):
    """Final payload from the multi-agent ensemble system."""
    captions: VideoCaptionResponse = Field(..., description="The final optimized 4-tone captions.")
    iterations_required: int = Field(..., description="Number of Creator <-> Judge loops executed.")
    final_score: int = Field(..., description="Final overall quality score out of 100.")
    critique_history: list[str] = Field(..., description="Chronological log of evaluations.")
    perception_telemetry: MultimodalPerceptionPacket = Field(..., description="Agent 1 telemetry.")
    latest_critique: Optional[CritiqueReport] = Field(default=None, description="Detailed live feedback from dual judge panel.")
    execution_timings: Optional[AgentExecutionTimings] = Field(default=None, description="Exact server-side execution timings.")
    agent1_sec: float = Field(default=0.0, description="Agent 1 execution duration in seconds.")
    agent2_sec: float = Field(default=0.0, description="Agent 2 execution duration in seconds.")
    agent3_sec: float = Field(default=0.0, description="Agent 3 execution duration in seconds.")
    total_pipeline_sec: float = Field(default=0.0, description="Total pipeline execution duration in seconds.")
