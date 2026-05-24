import express from "express";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import path from "path";
import FormData from "form-data";
import fetch from "node-fetch";
import dotenv from "dotenv";

// Render reads environment variables from its dashboard.
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_API_URL = process.env.PYTHON_API_URL;

app.use(cors());
app.use(express.static("public"));
app.use(express.json());

const uploadRoot = path.join(process.cwd(), "uploads");
fs.mkdirSync(uploadRoot, { recursive: true });

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

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    mode: "render-node-proxy-to-colab",
    pythonApiConfigured: Boolean(PYTHON_API_URL)
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
      if (!PYTHON_API_URL) {
        return res.status(500).json({
          error: "Missing PYTHON_API_URL. Set it in Render Environment Variables, for example https://xxxx.ngrok-free.app"
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

      const form = new FormData();
      form.append("video", fs.createReadStream(video.path), video.originalname);

      for (const img of images) {
        form.append("images", fs.createReadStream(img.path), img.originalname);
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

      for (const key of allowedFields) {
        if (req.body[key] !== undefined && req.body[key] !== "") {
          form.append(key, req.body[key]);
        }
      }

      const colabUrl = `${PYTHON_API_URL.replace(/\/$/, "")}/generate`;
      const response = await fetch(colabUrl, {
        method: "POST",
        body: form,
        headers: form.getHeaders()
      });

      if (!response.ok) {
        const text = await response.text();
        return res.status(response.status).json({
          error: "Colab API failed.",
          details: text.slice(0, 5000)
        });
      }

      res.setHeader("Content-Type", "video/mp4");
      res.setHeader("Content-Disposition", "inline; filename=output.mp4");
      response.body.pipe(res);
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
  console.log(`PYTHON_API_URL=${PYTHON_API_URL || "NOT_SET"}`);
});
