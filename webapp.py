import io
import os
from functools import lru_cache

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from predict import Predictor

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')


@lru_cache(maxsize=4)
def get_predictor(weights_path: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f'weights file not found: {weights_path}')
    return Predictor(weights_path=weights_path)


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/api/deblur')
def deblur():
    image_file = request.files.get('image')
    weights_path = request.form.get('weights_path', 'fpn_inception.h5')

    if image_file is None:
        return jsonify({'error': 'Missing image file'}), 400

    raw = np.frombuffer(image_file.read(), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({'error': 'Could not decode image'}), 400

    try:
        predictor = get_predictor(weights_path)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pred_rgb = predictor(rgb, mask=None)
    pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

    ok, encoded = cv2.imencode('.png', pred_bgr)
    if not ok:
        return jsonify({'error': 'Failed to encode output image'}), 500

    return Response(io.BytesIO(encoded.tobytes()).getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
