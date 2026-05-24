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
    const response = await fetch("/api/generate", { method: "POST", body: form });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const videoEl = $("outputVideo");
    videoEl.src = url;
    videoEl.load();
    videoEl.play().catch(() => {});

    const link = $("downloadLink");
    link.href = url;
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
