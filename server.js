import express from "express";
import cors from "cors";
import multer from "multer";
import { nanoid } from "nanoid";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const PIPELINE_PATH = process.env.PIPELINE_PATH || path.join(__dirname, "python", "motion_avatar_pipeline.py");
const LIVEPORTRAIT_REPO = process.env.LIVEPORTRAIT_REPO || path.join(__dirname, "external", "LivePortrait");

const uploadDir = path.join(__dirname, "uploads");
const outputDir = path.join(__dirname, "outputs");
const workDir = path.join(__dirname, "work");

for (const dir of [uploadDir, outputDir, workDir]) {
  fs.mkdirSync(dir, { recursive: true });
}

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));
app.use("/outputs", express.static(outputDir));

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const jobId = req.jobId || nanoid(10);
    req.jobId = jobId;
    const jobDir = path.join(uploadDir, jobId);
    fs.mkdirSync(path.join(jobDir, "images"), { recursive: true });
    fs.mkdirSync(path.join(jobDir, "video"), { recursive: true });

    if (file.fieldname === "images") cb(null, path.join(jobDir, "images"));
    else if (file.fieldname === "drivingVideo") cb(null, path.join(jobDir, "video"));
    else cb(new Error("Unknown upload field"));
  },
  filename: (req, file, cb) => {
    const safeName = file.originalname.replace(/[^a-zA-Z0-9._-]/g, "_");
    cb(null, `${Date.now()}-${safeName}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 1024 * 1024 * 700,
    files: 40,
  },
});

function clampNumber(value, min, max, fallback) {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  return Math.min(max, Math.max(min, num));
}

function runPythonJob({ jobId, imagesDir, drivingVideoPath, params }) {
  return new Promise((resolve, reject) => {
    const outputPath = path.join(outputDir, `${jobId}.mp4`);
    const jobWorkDir = path.join(workDir, jobId);
    fs.mkdirSync(jobWorkDir, { recursive: true });

    const args = [
      PIPELINE_PATH,
      "--backend", "liveportrait",
      "--liveportrait_repo", LIVEPORTRAIT_REPO,
      "--driving_video", drivingVideoPath,
      "--source_images", imagesDir,
      "--workdir", jobWorkDir,
      "--output", outputPath,
      "--driving_multiplier", String(params.drivingMultiplier),
      "--animation_region", params.animationRegion,
      "--grain_strength", String(params.grainStrength),
      "--motion_blur_alpha", String(params.motionBlurAlpha),
      "--brightness", String(params.brightness),
      "--contrast", String(params.contrast),
      "--gamma", String(params.gamma),
      "--saturation", String(params.saturation),
      "--sharpen_amount", String(params.sharpenAmount),
      "--disclosure_text", params.disclosureText,
    ];

    if (params.cropDrivingVideo) args.push("--flag_crop_driving_video");

    const child = spawn(PYTHON_BIN, args, {
      cwd: __dirname,
      env: process.env,
    });

    let logs = "";
    child.stdout.on("data", (data) => {
      logs += data.toString();
    });
    child.stderr.on("data", (data) => {
      logs += data.toString();
    });

    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python exited with code ${code}\n${logs}`));
        return;
      }
      resolve({ outputPath, outputUrl: `/outputs/${jobId}.mp4`, logs });
    });
  });
}

app.post(
  "/api/generate",
  upload.fields([
    { name: "images", maxCount: 30 },
    { name: "drivingVideo", maxCount: 1 },
  ]),
  async (req, res) => {
    try {
      const jobId = req.jobId;
      const images = req.files?.images || [];
      const videos = req.files?.drivingVideo || [];

      if (!images.length) {
        return res.status(400).json({ error: "Please upload at least one face image." });
      }
      if (!videos.length) {
        return res.status(400).json({ error: "Please upload a driving video." });
      }

      const params = {
        drivingMultiplier: clampNumber(req.body.drivingMultiplier, 0.1, 2.0, 1.0),
        grainStrength: clampNumber(req.body.grainStrength, 0, 12, 3.0),
        motionBlurAlpha: clampNumber(req.body.motionBlurAlpha, 0, 0.25, 0.08),
        brightness: clampNumber(req.body.brightness, -30, 30, 2),
        contrast: clampNumber(req.body.contrast, 0.8, 1.3, 1.03),
        gamma: clampNumber(req.body.gamma, 0.7, 1.4, 0.98),
        saturation: clampNumber(req.body.saturation, 0.7, 1.3, 1.02),
        sharpenAmount: clampNumber(req.body.sharpenAmount, 0, 0.5, 0.12),
        animationRegion: ["exp", "pose", "lip", "eyes", "all"].includes(req.body.animationRegion)
          ? req.body.animationRegion
          : "all",
        cropDrivingVideo: req.body.cropDrivingVideo === "true",
        disclosureText: req.body.disclosureText || "AI-generated avatar",
      };

      const imagesDir = path.dirname(images[0].path);
      const drivingVideoPath = videos[0].path;
      const result = await runPythonJob({ jobId, imagesDir, drivingVideoPath, params });

      res.json({
        ok: true,
        jobId,
        outputUrl: result.outputUrl,
        logs: result.logs.slice(-6000),
      });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: err.message || "Generation failed" });
    }
  }
);

app.listen(PORT, () => {
  console.log(`Avatar Motion Web running at http://localhost:${PORT}`);
});
