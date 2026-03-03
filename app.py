"""Flask UI for SmartEdge image sharpening with evaluation metrics."""

import base64

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

from sharpen import smartedge_sharpen
from metrics import evaluate
from evaluate import EXPECTED, _arrow

app = Flask(__name__)


def _read_upload(file_storage):
    """Read an uploaded file into a BGR float32 numpy array."""
    buf = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img.astype(np.float32)


def _to_base64_png(img_bgr):
    """Encode a BGR float32 image as a base64 PNG string."""
    _, buf = cv2.imencode('.png', np.clip(img_bgr, 0, 255).astype(np.uint8))
    return base64.b64encode(buf).decode('ascii')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/sharpen', methods=['POST'])
def sharpen():
    f = request.files.get('image')
    if f is None or f.filename == '':
        return jsonify(error="Please upload an input image.")

    bgr = _read_upload(f)
    if bgr is None:
        return jsonify(error="Could not read image. Try a different file.")

    ref_bgr = None
    ref_file = request.files.get('reference')
    if ref_file is not None and ref_file.filename != '':
        ref_bgr = _read_upload(ref_file)

    w = int(request.form.get('w', 3))
    alpha_mode = request.form.get('alpha', '1')

    before = evaluate(bgr, reference=ref_bgr)
    sharpened = smartedge_sharpen(bgr, w=w, alpha_mode=alpha_mode)
    after = evaluate(sharpened, reference=ref_bgr)

    metric_keys = ["Pm", "mu", "Lm", "NIQE"]
    if ref_bgr is not None:
        metric_keys += ["PSNR", "SSIM"]

    rows = []
    for key in metric_keys:
        if key not in before:
            continue
        b = before[key]
        a = after[key]
        change, result = _arrow(b, a, EXPECTED[key], key=key)
        rows.append([key, f"{b:.4f}", f"{a:.4f}", change, EXPECTED[key], result])

    return jsonify(
        original=_to_base64_png(bgr),
        sharpened=_to_base64_png(sharpened),
        metrics=rows,
    )


if __name__ == '__main__':
    app.run(debug=False, port=7860)
