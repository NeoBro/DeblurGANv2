import io
import os
import sys
import types
from functools import lru_cache

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request

from predict import Predictor

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
DEFAULT_WEIGHTS = os.environ.get('DEBLUR_WEIGHTS_PATH', 'fpn_inception.h5')
DEFAULT_UPSCALE_MODE = os.environ.get('DEBLUR_UPSCALE_MODE', 'lanczos')
DEFAULT_UPSCALE_MODEL_PATH = os.environ.get('DEBLUR_UPSCALE_MODEL_PATH', '').strip()
DEFAULT_AI_TILE_SIZE = int(os.environ.get('DEBLUR_AI_TILE_SIZE', '256'))


def parse_bool(raw_value: str, default: bool = False) -> bool:
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in ('1', 'true', 'yes', 'on')


@lru_cache(maxsize=4)
def get_predictor(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'weights file not found: {weights_path}')
    return Predictor(weights_path=weights_path)


def parse_upscale_scale(raw_value: str) -> int:
    try:
        scale = int(raw_value)
    except Exception:
        return 1
    return scale if scale in (1, 2, 4) else 1


def infer_default_upscale_model_path(scale: int) -> str:
    candidates = []
    if scale == 4:
        candidates = [
            'models/upscale/RealESRGAN_x4plus.pth',
            'models/upscale/realesr-general-x4v3.pth',
        ]
    elif scale == 2:
        candidates = [
            'models/upscale/RealESRGAN_x2plus.pth',
            'models/upscale/RealESRGAN_x4plus.pth',
            'models/upscale/realesr-general-x4v3.pth',
        ]

    for rel_path in candidates:
        abs_path = os.path.abspath(os.path.expanduser(rel_path))
        if os.path.exists(abs_path):
            return abs_path
    return ''


def resize_lanczos(image: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return image
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


@lru_cache(maxsize=8)
def get_realesrgan_upsampler(model_scale: int, model_path: str, device_name: str = 'auto', tile_size: int = 0):
    # BasicSR expects this legacy torchvision module path.
    if 'torchvision.transforms.functional_tensor' not in sys.modules:
        try:
            import torchvision.transforms.functional_tensor  # noqa: F401
        except ModuleNotFoundError:
            from torchvision.transforms import functional as tv_functional

            shim = types.ModuleType('torchvision.transforms.functional_tensor')
            shim.rgb_to_grayscale = tv_functional.rgb_to_grayscale
            sys.modules['torchvision.transforms.functional_tensor'] = shim

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if model_scale == 2:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    elif model_scale == 4:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    else:
        raise ValueError(f'Unsupported AI model scale: {model_scale}')

    kwargs = {
        'scale': model_scale,
        'model_path': model_path,
        'model': model,
        'tile': tile_size,
        'tile_pad': 10,
        'pre_pad': 0,
        'half': False,
    }
    if device_name == 'cpu':
        kwargs['device'] = torch.device('cpu')
    return RealESRGANer(**kwargs)


def upscale_output(image: np.ndarray, scale: int, mode: str, model_path: str, allow_cpu_fallback: bool = False):
    if scale <= 1:
        return image, 'off', ''

    if mode == 'ai':
        try:
            if not model_path:
                raise FileNotFoundError('Missing Real-ESRGAN model path (.pth).')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Real-ESRGAN model not found: {model_path}')
            model_name = os.path.basename(model_path).lower()
            model_scale = 2 if 'x2' in model_name else 4
            try:
                upsampler = get_realesrgan_upsampler(model_scale=model_scale, model_path=model_path, device_name='auto', tile_size=0)
                output, _ = upsampler.enhance(image, outscale=scale)
                return output, f'ai-realesrgan-gpu-x{scale}', ''
            except RuntimeError as exc:
                if 'CUDA' not in str(exc).upper():
                    raise
                torch.cuda.empty_cache()
                try:
                    upsampler = get_realesrgan_upsampler(
                        model_scale=model_scale,
                        model_path=model_path,
                        device_name='auto',
                        tile_size=DEFAULT_AI_TILE_SIZE,
                    )
                    output, _ = upsampler.enhance(image, outscale=scale)
                    return (
                        output,
                        f'ai-realesrgan-gpu-tiled-x{scale}',
                        f'AI retried on GPU using tile={DEFAULT_AI_TILE_SIZE} due to CUDA memory pressure.',
                    )
                except RuntimeError as tiled_exc:
                    if (not allow_cpu_fallback) or ('CUDA' not in str(tiled_exc).upper()):
                        raise
                    upsampler = get_realesrgan_upsampler(
                        model_scale=model_scale,
                        model_path=model_path,
                        device_name='cpu',
                        tile_size=DEFAULT_AI_TILE_SIZE,
                    )
                    output, _ = upsampler.enhance(image, outscale=scale)
                    return output, f'ai-realesrgan-cpu-x{scale}', 'AI retried on CPU after GPU memory pressure.'
        except Exception as exc:
            fallback = resize_lanczos(image, scale)
            return fallback, f'lanczos-x{scale}', f'AI upscale unavailable: {exc}'

    return resize_lanczos(image, scale), f'lanczos-x{scale}', ''


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/api/deblur')
def deblur():
    image_file = request.files.get('image')
    weights_path = (request.form.get('weights_path') or DEFAULT_WEIGHTS).strip()
    weights_path = os.path.abspath(os.path.expanduser(weights_path))
    do_deblur = parse_bool(request.form.get('do_deblur'), default=True)
    do_upscale = parse_bool(request.form.get('do_upscale'), default=False)
    allow_cpu_fallback = parse_bool(request.form.get('allow_cpu_fallback'), default=False)
    upscale_scale = parse_upscale_scale((request.form.get('upscale_scale') or '1').strip())
    upscale_mode = (request.form.get('upscale_mode') or DEFAULT_UPSCALE_MODE).strip().lower()
    if upscale_mode not in ('ai', 'lanczos'):
        upscale_mode = 'lanczos'
    upscale_model_path = (request.form.get('upscale_model_path') or DEFAULT_UPSCALE_MODEL_PATH).strip()
    if upscale_model_path:
        upscale_model_path = os.path.abspath(os.path.expanduser(upscale_model_path))
    elif upscale_mode == 'ai':
        upscale_model_path = infer_default_upscale_model_path(upscale_scale)
    if not do_upscale:
        upscale_scale = 1

    if image_file is None:
        return jsonify({'error': 'Missing image file'}), 400

    raw = np.frombuffer(image_file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({'error': 'Could not decode image'}), 400

    if do_deblur:
        try:
            predictor = get_predictor(weights_path)
        except Exception as exc:
            return jsonify({
                'error': str(exc),
                'hint': 'Set a valid .h5 path in the Weights Path field (or DEBLUR_WEIGHTS_PATH env var).'
            }), 400

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred_rgb = predictor(rgb, mask=None)
        pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)
        deblur_applied = 'deblur'
    else:
        pred_bgr = bgr.copy()
        deblur_applied = 'skip-deblur'

    pred_bgr, applied_upscale, upscale_note = upscale_output(
        pred_bgr,
        scale=upscale_scale,
        mode=upscale_mode,
        model_path=upscale_model_path,
        allow_cpu_fallback=allow_cpu_fallback,
    )

    ok, encoded = cv2.imencode('.png', pred_bgr)
    if not ok:
        return jsonify({'error': 'Failed to encode output image'}), 500

    response = Response(io.BytesIO(encoded.tobytes()).getvalue(), mimetype='image/png')
    response.headers['X-Deblur-Applied'] = deblur_applied
    response.headers['X-Upscale-Applied'] = applied_upscale
    if upscale_note:
        response.headers['X-Upscale-Note'] = upscale_note
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
