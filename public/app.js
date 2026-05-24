const form = document.querySelector("#generatorForm");
const imageInput = document.querySelector("#images");
const videoInput = document.querySelector("#drivingVideo");
const imagePreview = document.querySelector("#imagePreview");
const drivingPreview = document.querySelector("#drivingPreview");
const outputVideo = document.querySelector("#outputVideo");
const downloadLink = document.querySelector("#downloadLink");
const statusBox = document.querySelector("#status");
const logsBox = document.querySelector("#logs");
const generateBtn = document.querySelector("#generateBtn");

const sliderNames = [
  "drivingMultiplier",
  "grainStrength",
  "motionBlurAlpha",
  "brightness",
  "contrast",
  "gamma",
  "saturation",
  "sharpenAmount",
];

for (const name of sliderNames) {
  const input = form.elements[name];
  const value = document.querySelector(`#${name}Value`);
  const sync = () => value.textContent = input.value;
  input.addEventListener("input", sync);
  sync();
}

imageInput.addEventListener("change", () => {
  imagePreview.innerHTML = "";
  for (const file of [...imageInput.files].slice(0, 12)) {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    imagePreview.appendChild(img);
  }
});

videoInput.addEventListener("change", () => {
  const file = videoInput.files[0];
  if (!file) return;
  drivingPreview.src = URL.createObjectURL(file);
  drivingPreview.hidden = false;
});

function setStatus(text, mode) {
  statusBox.textContent = text;
  statusBox.className = `status ${mode}`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  generateBtn.disabled = true;
  outputVideo.hidden = true;
  downloadLink.hidden = true;
  logsBox.textContent = "";
  setStatus("Đang xử lý...", "running");

  const formData = new FormData(form);
  formData.set("cropDrivingVideo", form.elements.cropDrivingVideo.checked ? "true" : "false");

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Generate failed");
    }

    outputVideo.src = `${data.outputUrl}?t=${Date.now()}`;
    outputVideo.hidden = false;
    downloadLink.href = data.outputUrl;
    downloadLink.download = `avatar-${data.jobId}.mp4`;
    downloadLink.hidden = false;
    downloadLink.textContent = "Tải video output";
    logsBox.textContent = data.logs || "Done";
    setStatus("Hoàn tất", "done");
  } catch (error) {
    logsBox.textContent = error.message;
    setStatus("Lỗi khi generate", "error");
  } finally {
    generateBtn.disabled = false;
  }
});
