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
    cogvideox_model: str = "THUDM/CogVideoX-5b-I2V"
    cogvideox_v2v_model: str = "THUDM/CogVideoX-2b"
    cogvideox_prompt: str = (
        "A natural handheld smartphone portrait video, subtle head movement, "
        "realistic lighting, stable identity, detailed face, cinematic realism"
    )
    cogvideox_negative_prompt: str = (
        "distorted face, identity change, extra limbs, warped eyes, low quality, "
        "blurry, flicker, artifacts"
    )
    cogvideox_steps: int = 15
    cogvideox_guidance_scale: float = 6.0
    cogvideox_strength: float = 0.35
    cogvideox_fps: int = 8
    cogvideox_num_frames: int = 17
    cogvideox_width: int = 480
    cogvideox_height: int = 720
    cogvideox_seed: int | None = None
    cogvideox_dtype: str = "bfloat16"
    cogvideox_device: str = "cuda"

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
    supported_backends = {"liveportrait", "cogvideox", "liveportrait_cogvideox"}
    if cfg.backend not in supported_backends:
        raise ValueError(f"Unsupported backend: {cfg.backend}. Use one of: {', '.join(sorted(supported_backends))}")

    if not cfg.driving_video.exists():
        raise FileNotFoundError(cfg.driving_video)

    if cfg.driving_video.suffix.lower() not in VIDEO_EXTS:
        raise ValueError("Unsupported video format")

    if not cfg.source_images.exists():
        raise FileNotFoundError(cfg.source_images)

    if cfg.backend in {"liveportrait", "liveportrait_cogvideox"}:
        if cfg.liveportrait_repo is None:
            raise ValueError("--liveportrait_repo is required for LivePortrait backends")
        if not (cfg.liveportrait_repo / "inference.py").exists():
            raise FileNotFoundError(cfg.liveportrait_repo / "inference.py")

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


def _cogvideox_torch_dtype(torch_module, dtype_name: str):
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "float32":
        return torch_module.float32
    return torch_module.bfloat16


def _cogvideox_generator(torch_module, cfg: PipelineConfig):
    if cfg.cogvideox_seed is None:
        return None
    return torch_module.Generator(device=cfg.cogvideox_device).manual_seed(cfg.cogvideox_seed)


def _prepare_cogvideox_pipe(pipe, cfg: PipelineConfig):
    offloaded = False
    if cfg.cogvideox_device == "cuda" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        offloaded = True
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    return offloaded


def _resize_to_cogvideox_frame(frame, width: int, height: int):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw = max(8, int(w * scale) // 8 * 8)
    nh = max(8, int(h * scale) // 8 * 8)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def load_cogvideox_video_frames(video_path: Path, cfg: PipelineConfig):
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Could not read frames from {video_path}")

    frame_count = max(1, cfg.cogvideox_num_frames)
    frame_count = frame_count if (frame_count - 1) % 4 == 0 else ((frame_count - 1) // 4 * 4 + 1)
    frame_count = min(frame_count, total)
    indices = np.linspace(0, total - 1, frame_count, dtype=np.int32)
    wanted = set(int(i) for i in indices)

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frame = _resize_to_cogvideox_frame(frame, cfg.cogvideox_width, cfg.cogvideox_height)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        idx += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"Could not prepare CogVideoX frames from {video_path}")

    print(
        f"Prepared {len(frames)} frames for CogVideoX at "
        f"{cfg.cogvideox_width}x{cfg.cogvideox_height}."
    )
    return frames


def run_cogvideox_image_to_video(cfg: PipelineConfig, source: Path, raw_output: Path):
    print("Running CogVideoX image-to-video...")
    try:
        import torch
        from diffusers import CogVideoXImageToVideoPipeline
        from diffusers.utils import export_to_video, load_image
    except ImportError as exc:
        raise RuntimeError(
            "CogVideoX requires torch, diffusers, transformers, accelerate, sentencepiece and pillow. "
            "Install them in the Python environment before using --backend cogvideox."
        ) from exc

    dtype = _cogvideox_torch_dtype(torch, cfg.cogvideox_dtype)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(cfg.cogvideox_model, torch_dtype=dtype)
    offloaded = _prepare_cogvideox_pipe(pipe, cfg)

    if not offloaded:
        pipe.to(cfg.cogvideox_device)

    image = load_image(str(source.resolve()))
    generator = _cogvideox_generator(torch, cfg)
    result = pipe(
        image=image,
        prompt=cfg.cogvideox_prompt,
        negative_prompt=cfg.cogvideox_negative_prompt or None,
        height=cfg.cogvideox_height,
        width=cfg.cogvideox_width,
        num_inference_steps=cfg.cogvideox_steps,
        guidance_scale=cfg.cogvideox_guidance_scale,
        generator=generator,
        use_dynamic_cfg=True,
    )
    export_to_video(result.frames[0], str(raw_output), fps=cfg.cogvideox_fps)


def run_cogvideox_video_to_video(cfg: PipelineConfig, input_video: Path, raw_output: Path):
    print("Running CogVideoX video-to-video refinement...")
    try:
        import torch
        from diffusers import CogVideoXDPMScheduler, CogVideoXVideoToVideoPipeline
        from diffusers.utils import export_to_video
    except ImportError as exc:
        raise RuntimeError(
            "CogVideoX video-to-video requires torch, diffusers, transformers, accelerate, sentencepiece and pillow. "
            "Install them in the Python environment before using --backend liveportrait_cogvideox."
        ) from exc

    dtype = _cogvideox_torch_dtype(torch, cfg.cogvideox_dtype)
    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(cfg.cogvideox_v2v_model, torch_dtype=dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    offloaded = _prepare_cogvideox_pipe(pipe, cfg)

    if not offloaded:
        pipe.to(cfg.cogvideox_device)

    video = load_cogvideox_video_frames(input_video, cfg)
    generator = _cogvideox_generator(torch, cfg)
    result = pipe(
        video=video,
        prompt=cfg.cogvideox_prompt,
        negative_prompt=cfg.cogvideox_negative_prompt or None,
        height=cfg.cogvideox_height,
        width=cfg.cogvideox_width,
        strength=cfg.cogvideox_strength,
        guidance_scale=cfg.cogvideox_guidance_scale,
        num_inference_steps=cfg.cogvideox_steps,
        generator=generator,
        use_dynamic_cfg=True,
    )
    export_to_video(result.frames[0], str(raw_output), fps=cfg.cogvideox_fps)


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

    source = pick_best_source_image(cfg.source_images)
    raw_output = cfg.workdir / "raw.mp4"

    if cfg.backend == "liveportrait":
        check_driving_video_safety(cfg.driving_video)
        run_liveportrait(cfg, source, raw_output)
    elif cfg.backend == "cogvideox":
        run_cogvideox_image_to_video(cfg, source, raw_output)
    elif cfg.backend == "liveportrait_cogvideox":
        check_driving_video_safety(cfg.driving_video)
        liveportrait_output = cfg.workdir / "liveportrait_raw.mp4"
        run_liveportrait(cfg, source, liveportrait_output)
        run_cogvideox_video_to_video(cfg, liveportrait_output, raw_output)
    else:
        raise ValueError(f"Unsupported backend: {cfg.backend}")

    temp_processed = postprocess_video(cfg, raw_output)
    
    # Đè mã hoá camera thật
    convert_to_camera_spoof_mp4(temp_processed, cfg.output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driving_video", required=True, type=Path)
    parser.add_argument("--source_images", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="liveportrait", choices=["liveportrait", "cogvideox", "liveportrait_cogvideox"])
    parser.add_argument("--liveportrait_repo", type=Path)
    parser.add_argument("--cogvideox_model", default="THUDM/CogVideoX-5b-I2V")
    parser.add_argument("--cogvideox_v2v_model", default=PipelineConfig.cogvideox_v2v_model)
    parser.add_argument("--cogvideox_prompt", default=PipelineConfig.cogvideox_prompt)
    parser.add_argument("--cogvideox_negative_prompt", default=PipelineConfig.cogvideox_negative_prompt)
    parser.add_argument("--cogvideox_steps", default=PipelineConfig.cogvideox_steps, type=int)
    parser.add_argument("--cogvideox_guidance_scale", default=PipelineConfig.cogvideox_guidance_scale, type=float)
    parser.add_argument("--cogvideox_strength", default=PipelineConfig.cogvideox_strength, type=float)
    parser.add_argument("--cogvideox_fps", default=PipelineConfig.cogvideox_fps, type=int)
    parser.add_argument("--cogvideox_num_frames", default=PipelineConfig.cogvideox_num_frames, type=int)
    parser.add_argument("--cogvideox_width", default=PipelineConfig.cogvideox_width, type=int)
    parser.add_argument("--cogvideox_height", default=PipelineConfig.cogvideox_height, type=int)
    parser.add_argument("--cogvideox_seed", default=None, type=int)
    parser.add_argument("--cogvideox_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cogvideox_device", default="cuda")
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
