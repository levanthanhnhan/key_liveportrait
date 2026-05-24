from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}

@dataclass
class PipelineConfig:
    driving_video: Path
    source_images: Path
    workdir: Path
    output: Path
    backend: str = "liveportrait"
    liveportrait_repo: Path | None = None
    flag_crop_driving_video: bool = False
    driving_multiplier: float = 1.0
    animation_region: str = "all"
    grain_strength: float = 3.0
    motion_blur_alpha: float = 0.08
    brightness: float = 2.0
    contrast: float = 1.03
    gamma: float = 0.98
    saturation: float = 1.02
    sharpen_amount: float = 0.12
    disclosure_text: str = "AI-generated avatar"
    keep_raw: bool = False


def validate_config(cfg: PipelineConfig) -> list[Path]:
    if not cfg.driving_video.exists():
        raise FileNotFoundError(f"Driving video not found: {cfg.driving_video}")
    if cfg.driving_video.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Unsupported video format: {cfg.driving_video.suffix}")
    if not cfg.source_images.exists() or not cfg.source_images.is_dir():
        raise FileNotFoundError(f"Source image folder not found: {cfg.source_images}")
    images = sorted(p for p in cfg.source_images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"No valid source images found in: {cfg.source_images}")
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    return images


def image_quality_score(path: Path) -> float:
    img = cv2.imread(str(path))
    if img is None:
        return -1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean = float(gray.mean())
    exposure_penalty = abs(mean - 128.0) * 2.0
    size_bonus = min(img.shape[0] * img.shape[1] / 1_000_000, 3.0) * 30.0
    return sharpness - exposure_penalty + size_bonus


def pick_best_source_image(images: list[Path]) -> Path:
    scored = [(image_quality_score(p), p) for p in images]
    scored.sort(reverse=True, key=lambda x: x[0])
    if scored[0][0] < 0:
        raise RuntimeError("Could not read any source image.")
    return scored[0][1]


def newest_mp4(folder: Path, since_ts: float) -> Path | None:
    candidates = [p for p in folder.rglob("*.mp4") if p.is_file() and p.stat().st_mtime >= since_ts]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def run_liveportrait(cfg: PipelineConfig, source_image: Path, raw_output: Path) -> None:
    if cfg.liveportrait_repo is None:
        raise ValueError("--liveportrait_repo is required")
    repo = cfg.liveportrait_repo.resolve()
    inference_py = repo / "inference.py"
    if not inference_py.exists():
        raise FileNotFoundError(f"LivePortrait inference.py not found: {inference_py}")

    cmd = [
        "python", str(inference_py),
        "-s", str(source_image.resolve()),
        "-d", str(cfg.driving_video.resolve()),
        "--driving_multiplier", str(cfg.driving_multiplier),
        "--animation_region", cfg.animation_region,
    ]
    if cfg.flag_crop_driving_video:
        cmd.append("--flag_crop_driving_video")

    print("Running LivePortrait:", " ".join(cmd))
    before = time.time()
    result = subprocess.run(cmd, cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"LivePortrait failed with exit code {result.returncode}")

    generated = newest_mp4(repo / "animations", before)
    if generated is None:
        raise FileNotFoundError("No new mp4 found in LivePortrait animations folder.")
    shutil.copy2(generated, raw_output)
    print(f"Raw output copied from: {generated}")


def adjust_lighting(frame, brightness=0, contrast=1.0, gamma=1.0, saturation=1.0):
    img = frame.astype("float32") * contrast + brightness
    img = np.clip(img, 0, 255).astype("uint8")
    if gamma != 1.0:
        table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
        img = cv2.LUT(img, table)
    if saturation != 1.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[:, :, 1] *= saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        img = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return img


def enhance_skin_texture(frame, sharpen_amount=0.12):
    if sharpen_amount <= 0:
        return frame
    blur = cv2.GaussianBlur(frame, (0, 0), 1.2)
    sharp = cv2.addWeighted(frame, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)
    return np.clip(sharp, 0, 255).astype("uint8")


def add_camera_grain(frame, strength: float):
    if strength <= 0:
        return frame
    noise = np.random.normal(0, strength, frame.shape).astype(np.float32)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_subtle_motion_blur(frame, prev_frame, alpha: float):
    if prev_frame is None or alpha <= 0:
        return frame
    alpha = max(0.0, min(alpha, 0.25))
    return cv2.addWeighted(frame, 1.0 - alpha, prev_frame, alpha, 0)


def draw_disclosure(frame, text: str):
    if not text:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(w, h) / 1200)
    thickness = max(1, int(scale * 2))
    margin = int(18 * scale)
    pos = (margin, h - margin)
    cv2.putText(out, text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, pos, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def postprocess_video(input_video: Path, output_video: Path, cfg: PipelineConfig) -> None:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video}")
    prev = None
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = adjust_lighting(frame, cfg.brightness, cfg.contrast, cfg.gamma, cfg.saturation)
        frame = enhance_skin_texture(frame, cfg.sharpen_amount)
        frame = add_camera_grain(frame, cfg.grain_strength)
        frame = add_subtle_motion_blur(frame, prev, cfg.motion_blur_alpha)
        frame = draw_disclosure(frame, cfg.disclosure_text)
        writer.write(frame)
        prev = frame.copy()
        count += 1
    cap.release()
    writer.release()
    if count == 0:
        raise RuntimeError("No frames were written.")


def mux_audio_from_driving_video(video_no_audio: Path, driving_video: Path, output_video: Path):
    tmp = output_video.with_suffix(".muxed.mp4")
    cmd = ["ffmpeg", "-y", "-i", str(video_no_audio), "-i", str(driving_video), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp)]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode == 0 and tmp.exists():
            tmp.replace(output_video)
        else:
            print("Audio mux failed; keeping video without copied audio.")
            print(result.stdout)
            video_no_audio.replace(output_video)
    except FileNotFoundError:
        print("ffmpeg not found; keeping video without copied audio.")
        video_no_audio.replace(output_video)


def run_pipeline(cfg: PipelineConfig):
    images = validate_config(cfg)
    source = pick_best_source_image(images)
    print(f"Selected source image: {source}")
    raw_output = cfg.workdir / "raw_animation.mp4"
    processed = cfg.workdir / "processed_no_audio.mp4"
    run_liveportrait(cfg, source, raw_output)
    postprocess_video(raw_output, processed, cfg)
    mux_audio_from_driving_video(processed, cfg.driving_video, cfg.output)
    if not cfg.keep_raw:
        raw_output.unlink(missing_ok=True)
    print(f"Done: {cfg.output}")


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driving_video", required=True, type=Path)
    parser.add_argument("--source_images", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="liveportrait")
    parser.add_argument("--liveportrait_repo", required=True, type=Path)
    parser.add_argument("--flag_crop_driving_video", action="store_true")
    parser.add_argument("--driving_multiplier", default=1.0, type=float)
    parser.add_argument("--animation_region", default="all", choices=["exp", "pose", "lip", "eyes", "all"])
    parser.add_argument("--grain_strength", default=3.0, type=float)
    parser.add_argument("--motion_blur_alpha", default=0.08, type=float)
    parser.add_argument("--brightness", default=2.0, type=float)
    parser.add_argument("--contrast", default=1.03, type=float)
    parser.add_argument("--gamma", default=0.98, type=float)
    parser.add_argument("--saturation", default=1.02, type=float)
    parser.add_argument("--sharpen_amount", default=0.12, type=float)
    parser.add_argument("--disclosure_text", default="AI-generated avatar")
    parser.add_argument("--keep_raw", action="store_true")
    return PipelineConfig(**vars(parser.parse_args()))

if __name__ == "__main__":
    run_pipeline(parse_args())
