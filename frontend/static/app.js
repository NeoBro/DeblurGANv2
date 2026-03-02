const form = document.getElementById('upload-form');
const imageInput = document.getElementById('image-input');
const weightsPath = document.getElementById('weights-path');
const optDeblur = document.getElementById('opt-deblur');
const optUpscale = document.getElementById('opt-upscale');
const optUpscaleAi = document.getElementById('opt-upscale-ai');
const optCpuFallback = document.getElementById('opt-cpu-fallback');
const optDetailView = document.getElementById('opt-detail-view');
const upscaleScale = document.getElementById('upscale-scale');
const upscaleMode = document.getElementById('upscale-mode');
const upscaleModelPath = document.getElementById('upscale-model-path');
const statusEl = document.getElementById('status');
const uploadIndicator = document.getElementById('upload-indicator');
const runBtn = document.getElementById('run-btn');
const comparePanel = document.getElementById('compare-panel');
const imgBefore = document.getElementById('img-before');
const imgAfter = document.getElementById('img-after');
const beforeWrap = document.getElementById('before-wrap');
const slider = document.getElementById('slider');
const downloadLink = document.getElementById('download-link');
const detailPanel = document.getElementById('detail-panel');
const detailMeta = document.getElementById('detail-meta');
const zoomBefore = document.getElementById('zoom-before');
const zoomAfter = document.getElementById('zoom-after');
const focusBox = document.getElementById('focus-box');

let lastPreviewUrl = null;
let lastOutputUrl = null;
let currentFocusRegion = null;
let currentSourceSize = null;

slider.addEventListener('input', () => {
  beforeWrap.style.width = `${slider.value}%`;
});

function syncProcessingControls() {
  const upscaleEnabled = optUpscale.checked;
  const aiEnabled = upscaleEnabled && optUpscaleAi.checked;

  upscaleScale.disabled = !upscaleEnabled;
  upscaleMode.disabled = !upscaleEnabled;
  upscaleModelPath.disabled = !aiEnabled;
  optCpuFallback.disabled = !aiEnabled;

  if (!upscaleEnabled) {
    upscaleScale.value = '1';
  } else if (upscaleScale.value === '1') {
    upscaleScale.value = '2';
  }
  upscaleMode.value = aiEnabled ? 'ai' : 'lanczos';
}

optUpscale.addEventListener('change', syncProcessingControls);
optUpscaleAi.addEventListener('change', syncProcessingControls);
syncProcessingControls();

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (!file) {
    uploadIndicator.textContent = 'No image uploaded';
    uploadIndicator.classList.remove('ready');
    return;
  }
  uploadIndicator.textContent = `Uploaded: ${file.name} (${Math.round(file.size / 1024)} KB)`;
  uploadIndicator.classList.add('ready');
});

function waitForImageLoad(imgEl) {
  return new Promise((resolve, reject) => {
    if (imgEl.complete && imgEl.naturalWidth > 0) {
      resolve();
      return;
    }
    imgEl.onload = () => resolve();
    imgEl.onerror = () => reject(new Error('Failed to load image for detail view.'));
  });
}

function toCanvas(imgEl) {
  const canvas = document.createElement('canvas');
  canvas.width = imgEl.naturalWidth;
  canvas.height = imgEl.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(imgEl, 0, 0);
  return canvas;
}

function toCanvasSized(imgEl, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(imgEl, 0, 0, width, height);
  return canvas;
}

function findMostChangedRegion(beforeCanvas, afterCanvas) {
  const width = Math.min(beforeCanvas.width, afterCanvas.width);
  const height = Math.min(beforeCanvas.height, afterCanvas.height);
  const probeW = Math.max(32, Math.floor(width * 0.15));
  const probeH = Math.max(32, Math.floor(height * 0.15));
  const stepX = Math.max(8, Math.floor(width / 80));
  const stepY = Math.max(8, Math.floor(height / 80));
  const yWeight = 0.65;

  const beforeCtx = beforeCanvas.getContext('2d', { willReadFrequently: true });
  const afterCtx = afterCanvas.getContext('2d', { willReadFrequently: true });
  const beforeData = beforeCtx.getImageData(0, 0, width, height).data;
  const afterData = afterCtx.getImageData(0, 0, width, height).data;

  let bestScore = -1;
  let best = { x: Math.floor((width - probeW) / 2), y: Math.floor((height - probeH) / 2), w: probeW, h: probeH };

  for (let y = 0; y <= height - probeH; y += stepY) {
    for (let x = 0; x <= width - probeW; x += stepX) {
      let sum = 0;
      let count = 0;

      for (let py = y; py < y + probeH; py += 2) {
        const rowStart = py * width;
        for (let px = x; px < x + probeW; px += 2) {
          const idx = (rowStart + px) * 4;
          const dr = Math.abs(beforeData[idx] - afterData[idx]);
          const dg = Math.abs(beforeData[idx + 1] - afterData[idx + 1]);
          const db = Math.abs(beforeData[idx + 2] - afterData[idx + 2]);
          sum += (dr + dg + db) / 3;
          count += 1;
        }
      }

      const avgDiff = count > 0 ? sum / count : 0;
      const centerY = y + probeH / 2;
      const centerBias = 1 - Math.abs(centerY / height - 0.5) * yWeight;
      const score = avgDiff * Math.max(0.2, centerBias);
      if (score > bestScore) {
        bestScore = score;
        best = { x, y, w: probeW, h: probeH };
      }
    }
  }

  return best;
}

function renderZoom(sourceCanvas, targetCanvas, region) {
  const targetCtx = targetCanvas.getContext('2d');
  targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  targetCtx.imageSmoothingEnabled = false;
  targetCtx.drawImage(
    sourceCanvas,
    region.x, region.y, region.w, region.h,
    0, 0, targetCanvas.width, targetCanvas.height
  );
}

function renderFocusBox(region, sourceWidth, sourceHeight) {
  const viewW = comparePanel.querySelector('.compare-frame').clientWidth;
  const viewH = comparePanel.querySelector('.compare-frame').clientHeight;
  const fitScale = Math.min(viewW / sourceWidth, viewH / sourceHeight);
  const drawW = sourceWidth * fitScale;
  const drawH = sourceHeight * fitScale;
  const offsetX = (viewW - drawW) / 2;
  const offsetY = (viewH - drawH) / 2;

  focusBox.style.left = `${offsetX + region.x * fitScale}px`;
  focusBox.style.top = `${offsetY + region.y * fitScale}px`;
  focusBox.style.width = `${region.w * fitScale}px`;
  focusBox.style.height = `${region.h * fitScale}px`;
  focusBox.hidden = false;
}

async function updateDetailView() {
  await Promise.all([waitForImageLoad(imgBefore), waitForImageLoad(imgAfter)]);
  const targetWidth = Math.min(imgBefore.naturalWidth, imgAfter.naturalWidth);
  const targetHeight = Math.min(imgBefore.naturalHeight, imgAfter.naturalHeight);
  const beforeCanvas = toCanvasSized(imgBefore, targetWidth, targetHeight);
  const afterCanvas = toCanvasSized(imgAfter, targetWidth, targetHeight);
  const region = findMostChangedRegion(beforeCanvas, afterCanvas);

  renderZoom(beforeCanvas, zoomBefore, region);
  renderZoom(afterCanvas, zoomAfter, region);
  detailMeta.textContent = `Auto-selected region: x=${region.x}, y=${region.y}, size=${region.w}x${region.h}`;
  currentFocusRegion = region;
  currentSourceSize = { width: targetWidth, height: targetHeight };
  renderFocusBox(region, targetWidth, targetHeight);
  detailPanel.hidden = false;
}

window.addEventListener('resize', () => {
  if (!currentFocusRegion || !currentSourceSize || comparePanel.hidden) return;
  renderFocusBox(currentFocusRegion, currentSourceSize.width, currentSourceSize.height);
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    statusEl.textContent = 'Select an image first.';
    return;
  }

  if (lastPreviewUrl) URL.revokeObjectURL(lastPreviewUrl);
  const localPreview = URL.createObjectURL(file);
  lastPreviewUrl = localPreview;
  imgBefore.src = localPreview;
  comparePanel.hidden = true;
  detailPanel.hidden = true;
  focusBox.hidden = true;
  currentFocusRegion = null;
  currentSourceSize = null;

  const body = new FormData();
  body.append('image', file);
  body.append('weights_path', weightsPath.value.trim() || 'fpn_inception.h5');
  body.append('do_deblur', optDeblur.checked ? '1' : '0');
  body.append('do_upscale', optUpscale.checked ? '1' : '0');
  body.append('allow_cpu_fallback', optCpuFallback.checked ? '1' : '0');
  body.append('upscale_scale', upscaleScale.value);
  body.append('upscale_mode', upscaleMode.value);
  body.append('upscale_model_path', upscaleModelPath.value.trim());

  runBtn.disabled = true;
  statusEl.textContent = 'Running deblur model...';

  try {
    const resp = await fetch('/api/deblur', { method: 'POST', body });
    if (!resp.ok) {
      let errMsg = `Request failed: ${resp.status}`;
      const contentType = resp.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        const payload = await resp.json();
        errMsg = payload.hint ? `${payload.error} (${payload.hint})` : (payload.error || errMsg);
      } else {
        const text = await resp.text();
        if (text) errMsg = text;
      }
      throw new Error(errMsg);
    }

    const blob = await resp.blob();
    if (lastOutputUrl) URL.revokeObjectURL(lastOutputUrl);
    const outUrl = URL.createObjectURL(blob);
    lastOutputUrl = outUrl;
    imgAfter.src = outUrl;
    downloadLink.href = outUrl;
    comparePanel.hidden = false;
    if (optDetailView.checked) {
      await updateDetailView();
    } else {
      detailPanel.hidden = true;
      focusBox.hidden = true;
    }
    const upscaleApplied = resp.headers.get('X-Upscale-Applied') || 'off';
    const deblurApplied = resp.headers.get('X-Deblur-Applied') || 'deblur';
    const upscaleNote = resp.headers.get('X-Upscale-Note');
    const stepLabel = `${deblurApplied}, ${upscaleApplied}`;
    statusEl.textContent = upscaleNote
      ? `Done (${stepLabel}; ${upscaleNote})`
      : `Done (${stepLabel})`;
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    runBtn.disabled = false;
  }
});
