const form = document.getElementById('upload-form');
const imageInput = document.getElementById('image-input');
const weightsPath = document.getElementById('weights-path');
const statusEl = document.getElementById('status');
const runBtn = document.getElementById('run-btn');
const comparePanel = document.getElementById('compare-panel');
const imgBefore = document.getElementById('img-before');
const imgAfter = document.getElementById('img-after');
const beforeWrap = document.getElementById('before-wrap');
const slider = document.getElementById('slider');
const downloadLink = document.getElementById('download-link');

slider.addEventListener('input', () => {
  beforeWrap.style.width = `${slider.value}%`;
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = imageInput.files[0];
  if (!file) {
    statusEl.textContent = 'Select an image first.';
    return;
  }

  const localPreview = URL.createObjectURL(file);
  imgBefore.src = localPreview;

  const body = new FormData();
  body.append('image', file);
  body.append('weights_path', weightsPath.value.trim() || 'fpn_inception.h5');

  runBtn.disabled = true;
  statusEl.textContent = 'Running deblur model...';

  try {
    const resp = await fetch('/api/deblur', { method: 'POST', body });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text || `Request failed: ${resp.status}`);
    }

    const blob = await resp.blob();
    const outUrl = URL.createObjectURL(blob);
    imgAfter.src = outUrl;
    downloadLink.href = outUrl;
    comparePanel.hidden = false;
    statusEl.textContent = 'Done.';
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  } finally {
    runBtn.disabled = false;
  }
});
