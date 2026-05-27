from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import random
import math
import imageio_ffmpeg


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
    animation_region: str = "all"

    grain_strength: float = 14.0
    motion_blur_alpha: float = 0.24
    brightness: float = -6
    contrast: float = 0.82
    gamma: float = 1.12
    saturation: float = 0.82
    sharpen_amount: float = 0.00
    driving_multiplier: float = 0.62

    disclosure_text: str = ""
    keep_raw: bool = False


def validate_config(cfg: PipelineConfig):
    if not cfg.driving_video.exists():
        raise FileNotFoundError(cfg.driving_video)

    if cfg.driving_video.suffix.lower() not in VIDEO_EXTS:
        raise ValueError("Unsupported video format")

    if not cfg.source_images.exists():
        raise FileNotFoundError(cfg.source_images)

    cfg.workdir.mkdir(parents=True, exist_ok=True)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)


def image_quality_score(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return -1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()

    return sharpness - abs(brightness - 128)


def pick_best_source_image(folder: Path):
    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ]

    if not images:
        raise RuntimeError("No images found")

    scored = [(image_quality_score(p), p) for p in images]
    scored.sort(reverse=True, key=lambda x: x[0])

    return scored[0][1]


def newest_mp4(folder: Path, ts: float):
    mp4s = [
        p for p in folder.rglob("*.mp4")
        if p.stat().st_mtime >= ts
    ]

    if not mp4s:
        return None

    return max(mp4s, key=lambda p: p.stat().st_mtime)


def run_liveportrait(cfg: PipelineConfig, source: Path, raw_output: Path):
    repo = cfg.liveportrait_repo.resolve()

    inference = repo / "inference.py"

    cmd = [
        sys.executable,
        str(inference),

        "-s",
        str(source.resolve()),

        "-d",
        str(cfg.driving_video.resolve()),

        "--driving_multiplier",
        str(cfg.driving_multiplier),

        "--animation_region",
        cfg.animation_region,
    ]

    if cfg.flag_crop_driving_video:
        cmd.append("--flag_crop_driving_video")

    print("Running LivePortrait:", " ".join(cmd))

    before = time.time()
    ffmpeg_bin_dir = (cfg.workdir / "ffmpeg_bin").resolve()
    ffmpeg_bin_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = ffmpeg_bin_dir / "ffmpeg.exe"
    if not ffmpeg_exe.exists():
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg_exe)

    child_env = os.environ.copy()
    child_env["PATH"] = str(ffmpeg_bin_dir) + os.pathsep + child_env.get("PATH", "")
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        cwd=str(repo),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"LivePortrait failed with exit code {result.returncode}"
        )

    animations_dir = repo / "animations"

    generated = newest_mp4(animations_dir, before)

    if generated is None:
        raise RuntimeError("No generated mp4 found")

    shutil.copy2(generated, raw_output)


def add_camera_grain(frame, strength):
    noise = np.random.normal(
        0,
        strength,
        frame.shape
    ).astype(np.float32)

    out = frame.astype(np.float32) + noise

    return np.clip(out, 0, 255).astype(np.uint8)


def add_motion_blur(frame, prev, alpha):
    if prev is None:
        return frame

    return cv2.addWeighted(
        frame,
        1.0 - alpha,
        prev,
        alpha,
        0
    )


def adjust_lighting(
    frame,
    brightness=0,
    contrast=1.0,
    gamma=1.0,
    saturation=1.0,
):
    img = frame.astype(np.float32)

    img = img * contrast + brightness

    img = np.clip(img, 0, 255).astype(np.uint8)

    if gamma != 1.0:
        table = np.array([
            ((i / 255.0) ** (1.0 / gamma)) * 255
            for i in range(256)
        ]).astype(np.uint8)

        img = cv2.LUT(img, table)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    hsv[:, :, 1] *= saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    img = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2BGR
    )

    return img


def enhance_skin_texture(frame, sharpen_amount=0.12):
    blur = cv2.GaussianBlur(frame, (0, 0), 1.2)

    sharp = cv2.addWeighted(
        frame,
        1.0 + sharpen_amount,
        blur,
        -sharpen_amount,
        0
    )

    return np.clip(sharp, 0, 255).astype(np.uint8)


def draw_disclosure(frame, text):
    if not text:
        return frame

    out = frame.copy()

    h, w = out.shape[:2]

    cv2.putText(
        out,
        text,
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return out


def postprocess_video(cfg: PipelineConfig, input_video: Path):
    cap = cv2.VideoCapture(str(input_video))

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(cfg.output),
        fourcc,
        fps,
        (width, height)
    )

    prev = None

    frame_idx = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = adjust_lighting(
            frame,
            brightness=cfg.brightness,
            contrast=cfg.contrast,
            gamma=cfg.gamma,
            saturation=cfg.saturation,
        )

        frame = enhance_skin_texture(
            frame,
            cfg.sharpen_amount
        )

        frame = apply_camera_imperfections(
            frame,
            frame_idx
        )

        frame = add_camera_grain(
            frame,
            cfg.grain_strength
        )

        frame = add_motion_blur(
            frame,
            prev,
            cfg.motion_blur_alpha
        )

        frame = draw_disclosure(
            frame,
            cfg.disclosure_text
        )

        writer.write(frame)

        prev = frame.copy()

    cap.release()
    writer.release()


def convert_to_browser_mp4(input_path: Path, output_path: Path):
    temp = output_path.with_suffix(".browser.mp4")

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        str(temp),
    ]

    subprocess.run(cmd, check=True)
    temp.replace(output_path)


def apply_rolling_shutter(frame, frame_idx):
    h, w = frame.shape[:2]
    out = np.zeros_like(frame)

    strength = 1.2 + 0.6 * math.sin(frame_idx * 0.13)

    for y in range(h):
        phase = frame_idx * 0.18 + y * 0.018
        shift = int(math.sin(phase) * strength)

        out[y] = np.roll(frame[y], shift, axis=0)

    return out


def apply_face_texture_instability(frame, frame_idx):
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)

    # vùng mặt tương đối ở giữa frame
    cx1, cx2 = int(w * 0.28), int(w * 0.72)
    cy1, cy2 = int(h * 0.16), int(h * 0.78)

    face = out[cy1:cy2, cx1:cx2]

    if face.size == 0:
        return frame

    fh, fw = face.shape[:2]

    noise = np.random.normal(
        0,
        random.uniform(1.2, 3.2),
        face.shape
    ).astype(np.float32)

    # mask mềm để không lộ viền
    mask = np.zeros((fh, fw), dtype=np.float32)
    cv2.ellipse(
        mask,
        (fw // 2, fh // 2),
        (int(fw * 0.42), int(fh * 0.46)),
        0,
        0,
        360,
        1,
        -1
    )
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    mask = np.expand_dims(mask, axis=2)

    # texture fluctuation rất nhẹ
    face = face + noise * mask

    # micro contrast drift trên vùng mặt
    contrast = 1.0 + random.uniform(-0.018, 0.018)
    face = 128 + (face - 128) * contrast

    out[cy1:cy2, cx1:cx2] = (
        face * mask + out[cy1:cy2, cx1:cx2] * (1 - mask)
    )

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_camera_imperfections(frame, frame_idx, fps=30):
    h, w = frame.shape[:2]

    out = frame.astype(np.float32)

    # =========================================================
    # 1. EXPOSURE FLICKER
    # =========================================================

    exposure = (
        1.0
        + 0.018 * math.sin(frame_idx * 0.37)
        + random.uniform(-0.012, 0.012)
    )

    out *= exposure

    # =========================================================
    # 2. COLOR TEMPERATURE DRIFT
    # =========================================================

    temp = 1.0 + 0.012 * math.sin(frame_idx * 0.11)

    out[:, :, 2] *= temp
    out[:, :, 0] *= (2.0 - temp)

    out = np.clip(out, 0, 255).astype(np.uint8)

    # =========================================================
    # 3. MICRO HANDHELD SHAKE
    # =========================================================

    dx = random.uniform(-0.7, 0.7)
    dy = random.uniform(-0.5, 0.5)

    M = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    out = cv2.warpAffine(
        out,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # =========================================================
    # 4. AUTOFOCUS BREATHING
    # =========================================================

    zoom = (
        1.0
        + 0.0025 * math.sin(frame_idx * 0.09)
        + random.uniform(-0.001, 0.001)
    )

    nw = int(w * zoom)
    nh = int(h * zoom)

    resized = cv2.resize(
        out,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR
    )

    x1 = max((nw - w) // 2, 0)
    y1 = max((nh - h) // 2, 0)

    out = resized[y1:y1+h, x1:x1+w]

    if out.shape[0] != h or out.shape[1] != w:
        out = cv2.resize(out, (w, h))

    # =========================================================
    # 5. DYNAMIC BLUR FLUCTUATION
    # =========================================================

    if frame_idx % random.randint(4, 9) == 0:
        blur_sigma = random.uniform(0.15, 0.45)

        out = cv2.GaussianBlur(
            out,
            (3, 3),
            blur_sigma
        )

    # =========================================================
    # 6. SENSOR NOISE (SHADOW-BASED)
    # =========================================================

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    gray = gray.astype(np.float32) / 255.0

    shadow_mask = 1.0 - gray

    shadow_mask = np.expand_dims(shadow_mask, axis=2)

    noise_strength = random.uniform(2.0, 5.5)

    noise = np.random.normal(
        0,
        noise_strength,
        out.shape
    ).astype(np.float32)

    out = out.astype(np.float32)

    out += noise * shadow_mask

    out = np.clip(out, 0, 255).astype(np.uint8)

    # =========================================================
    # 7. CHROMATIC ABERRATION
    # =========================================================

    b, g, r = cv2.split(out)

    shift = 1

    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)

    out = cv2.merge([b, g, r])

    out = apply_rolling_shutter(out, frame_idx)
    out = apply_face_texture_instability(out, frame_idx)

    # =========================================================
    # 8. JPEG COMPRESSION FLUCTUATION
    # =========================================================

    quality = random.randint(88, 96)

    _, encoded = cv2.imencode(
        ".jpg",
        out,
        [cv2.IMWRITE_JPEG_QUALITY, quality]
    )

    out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return out


def run_pipeline(cfg: PipelineConfig):
    validate_config(cfg)

    params = {
        "grain_strength": cfg.grain_strength,
        "motion_blur_alpha": cfg.motion_blur_alpha,
        "brightness": cfg.brightness,
        "contrast": cfg.contrast,
        "gamma": cfg.gamma,
        "saturation": cfg.saturation,
        "sharpen_amount": cfg.sharpen_amount,
        "driving_multiplier": cfg.driving_multiplier,
    }
    print("Pipeline params:")
    print(json.dumps(params, indent=2))

    source = pick_best_source_image(cfg.source_images)

    print("Selected source image:", source)

    raw_output = cfg.workdir / "raw.mp4"

    run_liveportrait(cfg, source, raw_output)

    postprocess_video(cfg, raw_output)

    convert_to_browser_mp4(cfg.output, cfg.output)

    print("Done:", cfg.output)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--driving_video", required=True, type=Path)
    parser.add_argument("--source_images", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--backend", default="liveportrait")
    parser.add_argument("--liveportrait_repo", required=True, type=Path)

    parser.add_argument("--flag_crop_driving_video", action="store_true")

    parser.add_argument(
        "--animation_region",
        default="all",
        choices=["exp", "pose", "lip", "eyes", "all"]
    )

    parser.add_argument("--grain_strength", default=14.0, type=float)
    parser.add_argument("--motion_blur_alpha", default=0.24, type=float)
    parser.add_argument("--brightness", default=-6, type=float)
    parser.add_argument("--contrast", default=0.82, type=float)
    parser.add_argument("--gamma", default=1.12, type=float)
    parser.add_argument("--saturation", default=0.82, type=float)
    parser.add_argument("--sharpen_amount", default=0.00, type=float)
    parser.add_argument("--driving_multiplier", default=0.62, type=float)

    parser.add_argument("--disclosure_text", default="")
    parser.add_argument("--keep_raw", action="store_true")

    return PipelineConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_pipeline(parse_args())
