const $ = (id) => document.getElementById(id);

const sliders = [
  ["grain", "grainValue"], ["blur", "blurValue"], ["brightness", "brightnessValue"],
  ["contrast", "contrastValue"], ["gamma", "gammaValue"], ["saturation", "saturationValue"],
  ["sharpen", "sharpenValue"], ["driving", "drivingValue"]
];

for (const [inputId, labelId] of sliders) {
  const input = $(inputId);
  const label = $(labelId);
  input.addEventListener("input", () => label.textContent = input.value);
}

function log(message) {
  const el = $("log");
  el.textContent += `${new Date().toLocaleTimeString()} - ${message}\n`;
  el.scrollTop = el.scrollHeight;
}

$("generateBtn").addEventListener("click", async () => {
  const images = $("images").files;
  const video = $("video").files[0];
  const btn = $("generateBtn");
  const status = $("status");

  if (!images.length || !video) {
    alert("Hãy upload ít nhất 1 ảnh và 1 video motion.");
    return;
  }

  const form = new FormData();
  form.append("video", video);
  for (const img of images) form.append("images", img);

  form.append("grain_strength", $("grain").value);
  form.append("motion_blur_alpha", $("blur").value);
  form.append("brightness", $("brightness").value);
  form.append("contrast", $("contrast").value);
  form.append("gamma", $("gamma").value);
  form.append("saturation", $("saturation").value);
  form.append("sharpen_amount", $("sharpen").value);
  form.append("driving_multiplier", $("driving").value);
  form.append("animation_region", "all");

  btn.disabled = true;
  status.textContent = "Generating...";
  $("log").textContent = "";
  log("Uploading files to Render, then forwarding to Colab GPU API...");

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: form
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text);
    }

    const blob = await response.blob();
    const videoBlob = new Blob([blob], { type: "video/mp4" });
    const videoUrl = URL.createObjectURL(videoBlob);
    const videoEl = $("outputVideo");
    const link = $("downloadLink");

    videoEl.pause();
    videoEl.removeAttribute("src");
    videoEl.load();

    videoEl.src = videoUrl;
    videoEl.controls = true;
    videoEl.playsInline = true;
    videoEl.preload = "auto";
    videoEl.load();

    link.href = videoUrl;
    link.classList.remove("hidden");

    status.textContent = "Done";
    log("Output video received.");
  } catch (err) {
    status.textContent = "Error";
    log(err.message || String(err));
  } finally {
    btn.disabled = false;
  }
});

$("images").addEventListener("change", (e) => {
  const preview = $("imagePreview");
  preview.innerHTML = "";

  for (const file of e.target.files) {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    preview.appendChild(img);
  }
});

$("video").addEventListener("change", (e) => {
  const file = e.target.files[0];
  const preview = $("videoPreview");
  preview.innerHTML = "";

  if (!file) return;

  const video = document.createElement("video");
  video.src = URL.createObjectURL(file);
  video.controls = true;
  video.playsInline = true;
  video.muted = true;

  preview.appendChild(video);
});
