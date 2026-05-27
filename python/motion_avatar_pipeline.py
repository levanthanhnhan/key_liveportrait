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

    grain_strength: float = 7.5       
    motion_blur_alpha: float = 0.20
    brightness: float = -4
    contrast: float = 0.85
    gamma: float = 1.10
    saturation: float = 0.85
    sharpen_amount: float = 0.18      
    driving_multiplier: float = 0.95  

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
    images = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise RuntimeError("No images found")
    scored = [(image_quality_score(p), p) for p in images]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]


def newest_mp4(folder: Path, ts: float):
    mp4s = [p for p in folder.rglob("*.mp4") if p.stat().st_mtime >= ts]
    if not mp4s:
        return None
    return max(mp4s, key=lambda p: p.stat().st_mtime)


def check_driving_video_safety(video_path: Path):
    print(">>> Đang phân tích góc quay của Driving Video...")
    try:
        import mediapipe as mp
        from mediapipe.python.solutions import face_mesh as mp_face_mesh_module
    except ImportError:
        print("[Cảnh báo] Chưa cài hoặc lỗi MediaPipe. Bỏ qua kiểm tra góc quay.")
        return True

    cap = cv2.VideoCapture(str(video_path))
    face_mesh = mp_face_mesh_module.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    unsafe_frames = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            nose_x = landmarks[1].x
            left_x = landmarks[234].x
            right_x = landmarks[454].x
            dist_left = abs(nose_x - left_x)
            dist_right = abs(right_x - nose_x)
            if dist_left > 0 and dist_right > 0:
                ratio = max(dist_left / dist_right, dist_right / dist_left)
                if ratio > 2.2:
                    unsafe_frames += 1

    cap.release()
    face_mesh.close()
    if unsafe_frames > 0:
        print(f"\n[CẢNH BÁO ĐỎ] Video gốc có góc xoay đầu lớn nguy hiểm ({unsafe_frames}/{total_frames} frames).")
    else:
        print("[OK] Góc quay driving video an toàn.")


def run_liveportrait(cfg: PipelineConfig, source: Path, raw_output: Path):
    repo = cfg.liveportrait_repo.resolve()
    inference = repo / "inference.py"
    cmd = [
        sys.executable, str(inference),
        "-s", str(source.resolve()),
        "-d", str(cfg.driving_video.resolve()),
        "--driving_multiplier", str(cfg.driving_multiplier),
        "--animation_region", cfg.animation_region,
    ]
    if cfg.flag_crop_driving_video:
        cmd.append("--flag_crop_driving_video")

    print("Running LivePortrait...", " ".join(cmd))
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

    result = subprocess.run(cmd, cwd=str(repo), env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LivePortrait lỗi: {result.stdout}")

    generated = newest_mp4(repo / "animations", before)
    if generated is None:
        raise RuntimeError("Không tìm thấy file video AI sinh ra.")
    shutil.copy2(generated, raw_output)


def pad_to_standard_smartphone_ratio(frame, target_w=720, target_h=1280):
    """Đưa kích thước dị dạng 674x714 về độ phân giải chuẩn 9:16 smartphone"""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_CUBIC)
    
    # Tạo khung nền mờ (Blurred Background Padding) giống hiệu ứng camera quay dọc
    canvas = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.GaussianBlur(canvas, (99, 99), 15)
    
    # Đè khung video chính vào giữa
    x_offset = (target_w - nw) // 2
    y_offset = (target_h - nh) // 2
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    return canvas


def add_camera_grain(frame, strength):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    v_channel = hsv[:, :, 2] / 255.0
    shadow_mask = np.power(1.0 - v_channel, 1.5)
    shadow_mask = np.expand_dims(shadow_mask, axis=2)
    noise = np.random.normal(0, strength, frame.shape).astype(np.float32)
    out = frame.astype(np.float32) + (noise * shadow_mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_lighting_and_pores(frame, sharpen_amount=0.18, frame_idx=0):
    # Tạo nhiễu khối biểu cảm động (Dynamic Lightcast Fluctuation)
    blur = cv2.GaussianBlur(frame, (0, 0), 1.5)
    sharp = cv2.addWeighted(frame, 1.0 + sharpen_amount, blur, -sharpen_amount, 0)
    
    h, w = frame.shape[:2]
    # Tạo khối thớ thịt và lỗ chân lông chuyển động siêu nhỏ (Micro-pore fluctuation)
    micro_noise = np.random.normal(0, 3.5, (h, w, 3)).astype(np.float32)
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mid_tone_mask = np.exp(-0.5 * ((gray - 128) / 40) ** 2)
    mid_tone_mask = np.expand_dims(mid_tone_mask, axis=2)
    
    # Áp biến thiên ánh sáng động lên mặt khi đổi góc quay
    light_drift = 1.0 + 0.015 * math.sin(frame_idx * 0.15)
    textured = sharp.astype(np.float32) * light_drift + (micro_noise * mid_tone_mask)
    return np.clip(textured, 0, 255).astype(np.uint8)


def apply_camera_imperfections_and_parallax(frame, frame_idx, prev_frame=None):
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)

    # 1. EXPOSURE FLICKER (Rung sáng ống kính vật lý)
    exposure = 1.0 + 0.015 * math.sin(frame_idx * 0.41) + random.uniform(-0.01, 0.01)
    out *= exposure
    out = np.clip(out, 0, 255).astype(np.uint8)

    # 2. TEMPORAL SMOOTHING (Khử giật dịch chuyển tóc tai giữa các frame liên tiếp)
    if prev_frame is not None:
        out = cv2.addWeighted(out, 0.85, prev_frame, 0.15, 0)

    # 3. HANDHELD SHAKE & LINEAR SKEW (Rolling shutter nghiêng thực tế)
    dx = random.uniform(-1.8, 1.8)
    dy = random.uniform(-1.0, 1.0)
    skew = (dx / w) * 0.35  

    # 4. PSEUDO-PARALLAX BACKGROUND DRIFT
    # Dịch chuyển nhẹ toàn bộ ma trận hình ảnh để đánh lừa cảm giác tĩnh của kệ sách
    M = np.float32([[1, skew, dx], [0, 1, dy]])
    out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 5. CHROMATIC ABERRATION (Quang sai rìa thấu kính)
    b, g, r = cv2.split(out)
    r = np.roll(r, 1, axis=1)
    b = np.roll(b, -1, axis=1)
    out = cv2.merge([b, g, r])

    # 6. JPEG COMPRESSION ARTIFACTS
    _, encoded = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, random.randint(91, 96)])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def postprocess_video(cfg: PipelineConfig, input_video: Path):
    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ép buộc đầu ra ghi tạm thời ra kích thước chuẩn smartphone
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_processed = cfg.workdir / "temp_processed.mp4"
    writer = cv2.VideoWriter(str(temp_processed), fourcc, fps, (width, height))

    prev = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok: break

        # Khử tỷ lệ vuông AI -> Đưa về chuẩn khung hình smartphone dọc
        frame = pad_to_standard_smartphone_ratio(frame, width, height)
        
        # Xử lý kết cấu bề mặt da/ánh sáng thay đổi theo góc quay
        frame = adjust_lighting_and_pores(frame, cfg.sharpen_amount, frame_idx)
        
        # Xử lý lỗi cơ học quang học và Temporal chống giật tóc
        frame = apply_camera_imperfections_and_parallax(frame, frame_idx, prev)
        
        # Thêm nhiễu hạt bám vùng tối thực tế
        frame = add_camera_grain(frame, cfg.grain_strength)

        writer.write(frame)
        prev = frame.copy()
        frame_idx += 1

    cap.release()
    writer.release()
    return temp_processed


def convert_to_camera_spoof_mp4(input_path: Path, output_path: Path):
    """
    HÀM QUAN TRỌNG NHẤT: Làm giả hoàn toàn dấu vết phần cứng thiết bị quay di động.
    - Xoá hoàn toàn vết tích Lavf/Lavc từ FFmpeg/OpenCV.
    - Ép profile mã hoá giống hệt luồng camera Apple iOS.
    - Bơm dải âm thanh microphone nền giả lập thực tế để phá bỏ cờ lệnh 'No Audio'.
    """
    print(">>> Đang chạy hệ thống ngụy trang Metadata & Âm thanh thực tế...")

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-i", str(input_path),
        # Lệnh sinh dải âm thanh Microphone noise floor siêu nhỏ (-55dB) tránh bị bộ quét phát hiện video câm
        "-f", "lavfi", "-i", "anoisesrc=color=white:amplitude=0.001",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level:v", "4.1",
        "-b:v", "14400k",
        "-minrate", "14400k",
        "-maxrate", "14400k",
        "-bufsize", "28800k",
        "-x264-params", "nal-hrd=cbr:force-cfr=1",
        "-c:a", "aac",
        "-b:a", "64k",
        "-shortest", # Ngắt track âm thanh khi video kết thúc
        
        # BỘ LỆNH ĐÈ METADATA PHẦN CỨNG IPHONE
        "-map_metadata", "-1", # Xoá sạch toàn bộ metadata cũ của FFmpeg/AI
        "-metadata", "make=Apple",
        "-metadata", "model=iPhone 13 Pro",
        "-metadata", "software=15.4.1",
        "-metadata:s:v", "handler_name=VideoHandler",
        "-metadata:s:v", "encoder=Apple iOS v15.4.1 CoreMedia",
        "-metadata:s:a", "handler_name=AudioHandler",
        
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        print("[FFmpeg stdout]")
        print(exc.stdout or "")
        print("[FFmpeg stderr]")
        print(exc.stderr or "")
        raise

    final_output = output_path
    print(f"\n[HOÀN THÀNH] Video sạch đã được xuất tại: {final_output}")


def run_pipeline(cfg: PipelineConfig):
    validate_config(cfg)
    check_driving_video_safety(cfg.driving_video)

    source = pick_best_source_image(cfg.source_images)
    raw_output = cfg.workdir / "raw.mp4"

    run_liveportrait(cfg, source, raw_output)
    temp_processed = postprocess_video(cfg, raw_output)
    
    # Đè mã hoá camera thật
    convert_to_camera_spoof_mp4(temp_processed, cfg.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driving_video", required=True, type=Path)
    parser.add_argument("--source_images", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="liveportrait")
    parser.add_argument("--liveportrait_repo", required=True, type=Path)
    parser.add_argument("--flag_crop_driving_video", action="store_true")
    parser.add_argument("--animation_region", default="all", choices=["exp", "pose", "lip", "eyes", "all"])
    parser.add_argument("--grain_strength", default=7.5, type=float)
    parser.add_argument("--motion_blur_alpha", default=0.20, type=float)
    parser.add_argument("--brightness", default=-4, type=float)
    parser.add_argument("--contrast", default=0.85, type=float)
    parser.add_argument("--gamma", default=1.10, type=float)
    parser.add_argument("--saturation", default=0.85, type=float)
    parser.add_argument("--sharpen_amount", default=0.18, type=float)
    parser.add_argument("--driving_multiplier", default=0.95, type=float)
    return PipelineConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run_pipeline(parse_args())
