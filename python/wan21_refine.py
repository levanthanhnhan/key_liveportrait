from __future__ import annotations

import argparse
import functools
import os
from pathlib import Path


DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, underexposed, blurry, low quality, jpeg artifacts, distorted face, "
    "deformed, disfigured, bad anatomy, extra limbs, flicker, subtitles, text, watermark"
)


def log_step(message: str):
    print(f"\n>>> {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Refine a LivePortrait video with Wan2.1 video-to-video.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--prompt", default="realistic portrait video, natural face motion, stable identity")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--height", default=480, type=int)
    parser.add_argument("--width", default=720, type=int)
    parser.add_argument("--num_frames", default=81, type=int)
    parser.add_argument("--fps", default=16, type=int)
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--guidance_scale", default=5.0, type=float)
    parser.add_argument("--strength", default=0.35, type=float)
    parser.add_argument("--flow_shift", default=3.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--progress_every", default=1, type=int)
    return parser.parse_args()


def _fit_to_size(image, width: int, height: int):
    from PIL import Image

    image = image.convert("RGB")
    src_w, src_h = image.size
    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _prepare_video(path: Path, width: int, height: int, num_frames: int):
    from diffusers.utils import load_video

    frames = load_video(str(path))
    if not frames:
        raise RuntimeError(f"Cannot read video frames: {path}")

    frames = frames[:num_frames]
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))

    return [_fit_to_size(frame, width, height) for frame in frames]


def _patch_torch_rms_norm(torch):
    if hasattr(torch.nn, "RMSNorm"):
        return

    class RMSNorm(torch.nn.Module):
        def __init__(self, normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = eps if eps is not None else torch.finfo(dtype or torch.float32).eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = torch.nn.Parameter(torch.ones(self.normalized_shape, device=device, dtype=dtype))
            else:
                self.register_parameter("weight", None)

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps).to(input_dtype)
            if self.weight is not None:
                hidden_states = hidden_states * self.weight
            return hidden_states

    torch.nn.RMSNorm = RMSNorm


def _patch_torch_attention(torch):
    original = torch.nn.functional.scaled_dot_product_attention
    if getattr(original, "_wan21_compat_patched", False):
        return

    @functools.wraps(original)
    def scaled_dot_product_attention(*args, **kwargs):
        kwargs.pop("enable_gqa", None)
        return original(*args, **kwargs)

    scaled_dot_product_attention._wan21_compat_patched = True
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention


def _effective_denoise_steps(steps: int, strength: float):
    return max(1, min(steps, int(steps * strength)))


def _make_progress_callback(total_steps: int, every: int):
    every = max(1, every)

    def callback(_pipeline, step: int, _timestep, callback_kwargs):
        current = step + 1
        if current == 1 or current == total_steps or current % every == 0:
            percent = int(round((current / total_steps) * 100))
            log_step(f"Wan2.1 progress: {current}/{total_steps} ({percent}%)")
        return callback_kwargs

    return callback


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_root = Path(os.environ.get("HF_HOME", Path.cwd() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))

    log_step("START Wan2.1 load libraries")
    import torch
    _patch_torch_rms_norm(torch)
    _patch_torch_attention(torch)
    from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    log_step(f"START Wan2.1 load model: {args.model_id}")
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVideoToVideoPipeline.from_pretrained(
        args.model_id,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
    pipe.set_progress_bar_config(disable=True)

    if torch.cuda.is_available():
        if args.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        pipe.to("cpu")

    log_step("END Wan2.1 load model")

    log_step("START Wan2.1 prepare input video")
    video = _prepare_video(args.input, args.width, args.height, args.num_frames)
    log_step(f"END Wan2.1 prepare input video: {len(video)} frames")

    generator = None
    if args.seed:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(args.seed)

    log_step("START Wan2.1 inference")
    total_steps = _effective_denoise_steps(args.steps, args.strength)
    log_step(f"Wan2.1 progress: 0/{total_steps} (0%)")
    output = pipe(
        video=video,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        generator=generator,
        callback_on_step_end=_make_progress_callback(total_steps, args.progress_every),
        callback_on_step_end_tensor_inputs=["latents"],
    ).frames[0]
    log_step("END Wan2.1 inference")

    log_step(f"START Wan2.1 export video: {args.output}")
    export_to_video(output, str(args.output), fps=args.fps)
    log_step(f"END Wan2.1 export video: {args.output}")


if __name__ == "__main__":
    main()
