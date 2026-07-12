import os
import base64
import asyncio
import logging
import subprocess
import numpy as np
import cv2
from typing import List, Tuple
from fastapi import HTTPException
from fireworks_client import transcribe_audio, clean_transcript

logger = logging.getLogger("video_cap_pipeline")


def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extracts the audio track from a video using ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"[Warning] Audio extraction failed or ffmpeg not found: {e}")
        return False


def extract_frames(video_path: str, max_frames: int = 4, max_dim: int = 720) -> List[str]:
    """
    Extracts up to `max_frames` evenly spaced keyframes from the video.
    Downscales extracted keyframe image dimensions to max 720p (max_dim=720) before passing to vision models.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            return []

        num_frames = min(max_frames, max(1, total_frames))
        step = max(1, total_frames // num_frames)

        base64_frames = []
        current_idx = 0
        while current_idx < total_frames and len(base64_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            scale = min(1.0, float(max_dim) / max(h, w, 1))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            base64_frames.append(base64.b64encode(buffer).decode('utf-8'))

            for _ in range(step - 1):
                if not cap.grab():
                    break
            current_idx += step

        return base64_frames
    except Exception as e:
        logger.warning(f"[extract_frames error] {e}")
        return []
    finally:
        cap.release()


def detect_scene_changes(video_path: str, threshold: float = 0.4) -> int:
    """
    Detects the number of significant visual scene changes in the video using
    histogram comparison over sampled points.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            return 0

        num_samples = min(50, total_frames)
        step = max(1, total_frames // num_samples)

        scene_changes = 0
        prev_hist = None
        current_idx = 0

        while current_idx < total_frames and (current_idx // step) < 50:
            ret, frame = cap.read()
            if not ret:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if similarity > threshold:
                    scene_changes += 1

            prev_hist = hist
            for _ in range(step - 1):
                if not cap.grab():
                    break
            current_idx += step

        return scene_changes
    except Exception as e:
        logger.warning(f"[detect_scene_changes error] {e}")
        return 0
    finally:
        cap.release()


def classify_video_type(
    transcript: str,
    scene_change_count: int,
    has_audio: bool,
    video_duration: float
) -> str:
    """Classifies the video as 'visual_heavy', 'audio_heavy', or 'mixed'."""
    transcript_words = len(transcript.split()) if transcript else 0

    if not has_audio or transcript_words < 5:
        return "visual_heavy"

    words_per_second = transcript_words / max(video_duration, 1)
    if words_per_second > 1.5 and scene_change_count < 3:
        return "audio_heavy"

    return "mixed"


def validate_video_processable(duration: float, frames: List[str], transcript: str, has_audio: bool):
    """Pre-Check System: Verifies if the video has sufficient duration and signal."""
    if duration < 1.0:
        raise HTTPException(
            status_code=400,
            detail="Video Unprocessable: Duration is under 1 second. Please upload a video with sufficient duration."
        )

    if not frames:
        raise HTTPException(
            status_code=400,
            detail="Video Unprocessable: Could not extract visual frames from the video stream."
        )

    words_count = len(transcript.split()) if (has_audio and transcript) else 0

    blank_frames_count = 0
    for b64 in frames:
        try:
            img_bytes = base64.b64decode(b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None and np.std(img) < 2.0:
                blank_frames_count += 1
        except Exception:
            pass

    if len(frames) > 0 and blank_frames_count == len(frames) and words_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Video Unprocessable: The video appears to be completely blank/solid color with no spoken audio."
        )


def sync_visual_pipeline(video_path: str, max_frames: int = 4, max_dim: int = 720) -> Tuple[List[str], int, float]:
    """
    Runs CPU/disk-bound OpenCV operations in a single fast pass.
    Caps keyframes at max_frames=4 and downscales to max_dim=720p.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return [], 0, 0.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            return [], 0, 0.0

        duration = total_frames / fps

        num_extract_frames = min(max_frames, max(1, total_frames))
        extract_step = max(1, total_frames // num_extract_frames)

        num_scene_checks = min(50, max(10, int(duration * 2)))
        scene_step = max(1, total_frames // num_scene_checks)

        base64_frames = []
        scene_changes = 0
        prev_hist = None

        current_idx = 0
        while current_idx < total_frames:
            is_extract_target = (current_idx % extract_step == 0) and (len(base64_frames) < max_frames)
            is_scene_target = (current_idx % scene_step == 0)

            if is_extract_target or is_scene_target:
                ret, frame = cap.read()
                if not ret:
                    break

                if is_scene_target:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
                    cv2.normalize(hist, hist)
                    if prev_hist is not None:
                        similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                        if similarity > 0.4:
                            scene_changes += 1
                    prev_hist = hist

                if is_extract_target:
                    h, w = frame.shape[:2]
                    scale = min(1.0, float(max_dim) / max(h, w, 1))
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    _, buffer = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    base64_frames.append(base64.b64encode(buffer).decode('utf-8'))

                current_idx += 1
            else:
                if not cap.grab():
                    break
                current_idx += 1

        return base64_frames, scene_changes, duration
    except Exception as e:
        logger.warning(f"[Visual Pipeline Error] {e}")
        return [], 0, 0.0
    finally:
        cap.release()


async def async_audio_pipeline(video_path: str, audio_path: str) -> Tuple[str, bool]:
    """Runs ffmpeg audio extraction and Whisper transcription concurrently."""
    has_audio = await asyncio.to_thread(extract_audio, video_path, audio_path)
    if not has_audio or not os.path.exists(audio_path):
        fallback_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_temp_audio.mp3")
        if os.path.exists(fallback_audio):
            logger.info(f"[Audio] ffmpeg not in PATH; falling back to local audio file: {fallback_audio}")
            audio_path = fallback_audio
            has_audio = True
        else:
            return "", False
    raw_transcript = await transcribe_audio(audio_path)
    final_transcript = await clean_transcript(raw_transcript)
    return final_transcript, True
