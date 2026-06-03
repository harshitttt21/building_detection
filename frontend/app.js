const imageInput = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const minArea = document.getElementById('minArea');
const maxArea = document.getElementById('maxArea');
const minAreaValue = document.getElementById('minAreaValue');
const maxAreaValue = document.getElementById('maxAreaValue');
const processBtn = document.getElementById('processBtn');
const statusNode = document.getElementById('status');
const originalPreview = document.getElementById('originalPreview');
const annotatedPreview = document.getElementById('annotatedPreview');
const maskPreview = document.getElementById('maskPreview');
const buildingCount = document.getElementById('buildingCount');
const processingTime = document.getElementById('processingTime');
const downloadLink = document.getElementById('downloadLink');

let selectedFile = null;
let selectedObjectUrl = null;
let throttleTimer = null;

function setStatus(message) {
  statusNode.textContent = message;
}

function updateSliderLabels() {
  minAreaValue.textContent = minArea.value;
  maxAreaValue.textContent = maxArea.value;
}

function resetResults() {
  buildingCount.textContent = '0';
  processingTime.textContent = '0 ms';
  annotatedPreview.removeAttribute('src');
  maskPreview.removeAttribute('src');
  downloadLink.href = '#';
}

function previewFile(file) {
  if (selectedObjectUrl) {
    URL.revokeObjectURL(selectedObjectUrl);
  }
  selectedObjectUrl = URL.createObjectURL(file);
  originalPreview.src = selectedObjectUrl;
}

function scheduleProcessing() {
  if (!selectedFile) {
    return;
  }
  window.clearTimeout(throttleTimer);
  throttleTimer = window.setTimeout(() => processImage(), 300);
}

async function processImage() {
  if (!selectedFile) {
    setStatus('Upload an image first.');
    return;
  }

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('min_area', minArea.value);
  formData.append('max_area', maxArea.value);
  formData.append('model_type', 'vit_b');

  processBtn.disabled = true;
  setStatus('Processing image with YOLO...');

  try {
    const response = await fetch('/detect', {
      method: 'POST',
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || 'Detection failed.');
    }

    buildingCount.textContent = payload.building_count;
    processingTime.textContent = `${payload.processing_time_ms} ms`;
    annotatedPreview.src = `${payload.annotated_image_url}?t=${Date.now()}`;
    maskPreview.src = `${payload.mask_image_url}?t=${Date.now()}`;
    downloadLink.href = payload.annotated_image_url;
    downloadLink.download = payload.annotated_image_url.split('/').pop();
    setStatus(`Done. ${payload.filtered_mask_count} masks survived filtering.`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    processBtn.disabled = false;
  }
}

imageInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }
  selectedFile = file;
  previewFile(file);
  resetResults();
  setStatus('Image ready. Processing...');
  processImage();
});

processBtn.addEventListener('click', () => processImage());
minArea.addEventListener('input', () => {
  updateSliderLabels();
  scheduleProcessing();
});
maxArea.addEventListener('input', () => {
  updateSliderLabels();
  scheduleProcessing();
});

dropZone.addEventListener('dragover', (event) => {
  event.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropZone.classList.remove('dragover');
  const [file] = event.dataTransfer.files;
  if (!file) {
    return;
  }
  imageInput.files = event.dataTransfer.files;
  selectedFile = file;
  previewFile(file);
  resetResults();
  setStatus('Image ready. Processing...');
  processImage();
});

updateSliderLabels();
