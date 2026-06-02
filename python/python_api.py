from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
import subprocess
import uuid
import json
import os

from pathlib import Path

app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = Path(os.environ.get("LOCAL_JOBS_DIR", PROJECT_ROOT / "local_jobs")).resolve()
LIVEPORTRAIT_REPO = Path(
    os.environ.get("LIVEPORTRAIT_REPO", PROJECT_ROOT / "LivePortrait")
).resolve()
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")
DEFAULT_BACKEND = os.environ.get("PIPELINE_BACKEND", "liveportrait_cogvideox")

BASE.mkdir(parents=True, exist_ok=True)


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    video: UploadFile = File(...),
    images: list[UploadFile] = File(...),

    backend: str = Form(DEFAULT_BACKEND),
    grain_strength: float = Form(14.0),
    motion_blur_alpha: float = Form(0.24),
    brightness: float = Form(-6),
    contrast: float = Form(0.82),
    gamma: float = Form(1.12),
    saturation: float = Form(0.82),
    sharpen_amount: float = Form(0.00),
    driving_multiplier: float = Form(0.62),
    animation_region: str = Form("all"),
    cogvideox_model: str = Form("THUDM/CogVideoX-5b-I2V"),
    cogvideox_v2v_model: str = Form("THUDM/CogVideoX-2b"),
    cogvideox_prompt: str = Form("A natural handheld smartphone portrait video, subtle head movement, realistic lighting, stable identity, detailed face, cinematic realism"),
    cogvideox_negative_prompt: str = Form("distorted face, identity change, extra limbs, warped eyes, low quality, blurry, flicker, artifacts"),
    cogvideox_steps: int = Form(15),
    cogvideox_guidance_scale: float = Form(6.0),
    cogvideox_strength: float = Form(0.35),
    cogvideox_fps: int = Form(8),
    cogvideox_num_frames: int = Form(17),
    cogvideox_width: int = Form(480),
    cogvideox_height: int = Form(720),
    cogvideox_seed: int | None = Form(None),
    cogvideox_dtype: str = Form("bfloat16"),
    cogvideox_device: str = Form("cuda"),
):
    params = {
        "backend": backend,
        "grain_strength": grain_strength,
        "motion_blur_alpha": motion_blur_alpha,
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "saturation": saturation,
        "sharpen_amount": sharpen_amount,
        "driving_multiplier": driving_multiplier,
        "animation_region": animation_region,
        "cogvideox_model": cogvideox_model,
        "cogvideox_v2v_model": cogvideox_v2v_model,
        "cogvideox_steps": cogvideox_steps,
        "cogvideox_guidance_scale": cogvideox_guidance_scale,
        "cogvideox_strength": cogvideox_strength,
        "cogvideox_fps": cogvideox_fps,
        "cogvideox_num_frames": cogvideox_num_frames,
        "cogvideox_width": cogvideox_width,
        "cogvideox_height": cogvideox_height,
        "cogvideox_seed": cogvideox_seed,
        "cogvideox_dtype": cogvideox_dtype,
        "cogvideox_device": cogvideox_device,
    }

    print(json.dumps(params, indent=2))

    job_id = str(uuid.uuid4())

    job_dir = BASE / job_id

    image_dir = job_dir / "images"

    image_dir.mkdir(parents=True)

    video_path = job_dir / video.filename

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    for img in images:
        img_path = image_dir / img.filename

        with open(img_path, "wb") as f:
            shutil.copyfileobj(img.file, f)

    output_path = job_dir / "output.mp4"

    cmd = [
        PYTHON_BIN,

        str(PROJECT_ROOT / "python" / "motion_avatar_pipeline.py"),

        "--backend",
        backend,

        "--liveportrait_repo",
        str(LIVEPORTRAIT_REPO),

        "--driving_video",
        str(video_path),

        "--source_images",
        str(image_dir),

        "--workdir",
        str(job_dir),

        "--output",
        str(output_path),

        "--flag_crop_driving_video",

        "--grain_strength",
        str(grain_strength),

        "--motion_blur_alpha",
        str(motion_blur_alpha),

        "--brightness",
        str(brightness),

        "--contrast",
        str(contrast),

        "--gamma",
        str(gamma),

        "--saturation",
        str(saturation),

        "--sharpen_amount",
        str(sharpen_amount),

        "--driving_multiplier",
        str(driving_multiplier),

        "--animation_region",
        animation_region,

        "--cogvideox_model",
        cogvideox_model,

        "--cogvideox_v2v_model",
        cogvideox_v2v_model,

        "--cogvideox_prompt",
        cogvideox_prompt,

        "--cogvideox_negative_prompt",
        cogvideox_negative_prompt,

        "--cogvideox_steps",
        str(cogvideox_steps),

        "--cogvideox_guidance_scale",
        str(cogvideox_guidance_scale),

        "--cogvideox_strength",
        str(cogvideox_strength),

        "--cogvideox_fps",
        str(cogvideox_fps),

        "--cogvideox_num_frames",
        str(cogvideox_num_frames),

        "--cogvideox_width",
        str(cogvideox_width),

        "--cogvideox_height",
        str(cogvideox_height),

        "--cogvideox_dtype",
        cogvideox_dtype,

        "--cogvideox_device",
        cogvideox_device,
    ]

    if cogvideox_seed is not None:
        cmd.extend(["--cogvideox_seed", str(cogvideox_seed)])

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        return {
            "error": "Motion avatar pipeline failed",
            "code": result.returncode,
            "log": result.stdout,
        }

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="output.mp4"
    )
