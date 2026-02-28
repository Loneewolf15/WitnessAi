"""
WitnessAI — FFmpeg Utilities
Handles video clip extraction from frame buffers.
Falls back to imageio/OpenCV if FFmpeg is unavailable.
"""
import subprocess
import shutil
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def save_frames_to_clip(
    frames: List[Tuple[int, datetime, np.ndarray]],
    output_path: Path,
    fps: float = 15.0,
) -> bool:
    """
    Save a list of numpy frames to an MP4 video file.

    Tries FFmpeg first (via pipe), falls back to imageio if unavailable.

    Args:
        frames: List of (frame_number, timestamp, numpy_array) tuples.
        output_path: Destination .mp4 path.
        fps: Frames per second for the output video.

    Returns:
        True if successful, False otherwise.
    """
    if not frames:
        logger.warning("save_frames_to_clip called with empty frames list")
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_frames = [f for _, _, f in frames]
    h, w = raw_frames[0].shape[:2]

    if FFMPEG_AVAILABLE:
        return _save_with_ffmpeg(raw_frames, output_path, fps, w, h)
    else:
        return _save_with_imageio(raw_frames, output_path, fps)


def _save_with_ffmpeg(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> bool:
    """Pipe raw BGR frames into FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        str(output_path),
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        for frame in frames:
            if len(frame.shape) == 2:  # grayscale → BGR
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        _, stderr = proc.communicate(timeout=60)
        if proc.returncode != 0:
            logger.error(f"FFmpeg error: {stderr.decode()}")
            return False
        logger.info(f"Clip saved (FFmpeg): {output_path}")
        return True
    except Exception as e:
        logger.error(f"FFmpeg clip save failed: {e}")
        return False


def _save_with_imageio(
    frames: List[np.ndarray], output_path: Path, fps: float
) -> bool:
    """Fallback: use imageio to write frames."""
    try:
        import imageio  # type: ignore
        writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264")
        for frame in frames:
            # imageio expects RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame)
        writer.close()
        logger.info(f"Clip saved (imageio): {output_path}")
        return True
    except ImportError:
        logger.warning("imageio not available; saving raw frames as numpy archive")
        np.save(str(output_path.with_suffix(".npy")), np.stack(frames))
        return True
    except Exception as e:
        logger.error(f"imageio clip save failed: {e}")
        return False


def clip_duration_seconds(clip_path: Path) -> Optional[float]:
    """Return the duration of a video clip using ffprobe, or None."""
    if not FFMPEG_AVAILABLE:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(clip_path),
            ],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return None
