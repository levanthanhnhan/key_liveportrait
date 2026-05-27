import express from "express";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { spawn } from "child_process";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_BIN = process.env.PYTHON_BIN || "python";
const LIVEPORTRAIT_REPO = path.resolve(
  process.env.LIVEPORTRAIT_REPO || path.join(process.cwd(), "LivePortrait")
);

app.use(cors());
app.use(express.static("public"));
app.use(express.json());

const uploadRoot = path.join(process.cwd(), "uploads");
const jobsRoot = path.join(process.cwd(), "local_jobs");
fs.mkdirSync(uploadRoot, { recursive: true });
fs.mkdirSync(jobsRoot, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadRoot),
  filename: (req, file, cb) => {
    const safe = file.originalname.replace(/[^a-zA-Z0-9._-]/g, "_");
    cb(null, `${Date.now()}-${safe}`);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 250 * 1024 * 1024,
    files: 30
  }
});

function cleanupFiles(files = []) {
  for (const file of files) {
    if (!file?.path) continue;
    fs.unlink(file.path, () => {});
  }
}

function cleanupDirectory(dir) {
  if (!dir) return;
  const resolved = path.resolve(dir);
  if (!resolved.startsWith(jobsRoot)) return;
  fs.rm(resolved, { recursive: true, force: true }, () => {});
}

function runPipeline(args) {
  return new Promise((resolve) => {
    const child = spawn(PYTHON_BIN, args, {
      cwd: process.cwd(),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8"
      }
    });

    let log = "";
    const append = (data) => {
      const text = data.toString();
      process.stdout.write(text);
      log += text;
      if (log.length > 20000) {
        log = log.slice(-20000);
      }
    };

    child.stdout.on("data", append);
    child.stderr.on("data", append);
    child.on("error", (err) => resolve({ code: 1, log: err.message }));
    child.on("close", (code) => resolve({ code, log }));
  });
}

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    mode: "local-node-to-liveportrait",
    pythonBin: PYTHON_BIN,
    liveportraitRepo: LIVEPORTRAIT_REPO,
    liveportraitConfigured: fs.existsSync(path.join(LIVEPORTRAIT_REPO, "inference.py"))
  });
});

app.post(
  "/api/generate",
  upload.fields([
    { name: "video", maxCount: 1 },
    { name: "images", maxCount: 20 }
  ]),
  async (req, res) => {
    const uploaded = [
      ...(req.files?.video || []),
      ...(req.files?.images || [])
    ];

    try {
      if (!fs.existsSync(path.join(LIVEPORTRAIT_REPO, "inference.py"))) {
        return res.status(500).json({
          error: "Missing LivePortrait local repo.",
          details: `Set LIVEPORTRAIT_REPO in .env or place LivePortrait at ${LIVEPORTRAIT_REPO}`
        });
      }

      const video = req.files?.video?.[0];
      const images = req.files?.images || [];

      if (!video) {
        return res.status(400).json({ error: "Missing driving video." });
      }
      if (!images.length) {
        return res.status(400).json({ error: "Please upload at least one source image." });
      }

      const allowedFields = [
        "grain_strength",
        "motion_blur_alpha",
        "brightness",
        "contrast",
        "gamma",
        "saturation",
        "sharpen_amount",
        "driving_multiplier",
        "animation_region"
      ];

      const jobId = crypto.randomUUID();
      const jobDir = path.join(jobsRoot, jobId);
      const imageDir = path.join(jobDir, "images");
      fs.mkdirSync(imageDir, { recursive: true });

      const videoPath = path.join(jobDir, video.filename);
      fs.renameSync(video.path, videoPath);
      video.path = videoPath;

      for (const img of images) {
        const target = path.join(imageDir, img.filename);
        fs.renameSync(img.path, target);
        img.path = target;
      }

      const outputPath = path.join(jobDir, "output.mp4");
      const pipelineArgs = [
        path.join("python", "motion_avatar_pipeline.py"),
        "--backend", "liveportrait",
        "--liveportrait_repo", LIVEPORTRAIT_REPO,
        "--driving_video", videoPath,
        "--source_images", imageDir,
        "--workdir", jobDir,
        "--output", outputPath,
        "--flag_crop_driving_video"
      ];

      for (const key of allowedFields) {
        if (req.body[key] !== undefined && req.body[key] !== "") {
          pipelineArgs.push(`--${key}`, String(req.body[key]));
        }
      }

      const result = await runPipeline(pipelineArgs);

      if (result.code !== 0 || !fs.existsSync(outputPath)) {
        cleanupDirectory(jobDir);
        return res.status(500).json({
          error: "Local LivePortrait pipeline failed.",
          code: result.code,
          details: result.log
        });
      }

      res.sendFile(outputPath, {
        headers: {
          "Content-Type": "video/mp4",
          "Content-Disposition": "inline; filename=output.mp4"
        }
      }, (err) => {
        cleanupDirectory(jobDir);
        if (err) console.error(err);
      });
    } catch (err) {
      console.error(err);
      res.status(500).json({
        error: "Generate failed.",
        details: err.message
      });
    } finally {
      cleanupFiles(uploaded);
    }
  }
);

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`PYTHON_BIN=${PYTHON_BIN}`);
  console.log(`LIVEPORTRAIT_REPO=${LIVEPORTRAIT_REPO}`);
});
