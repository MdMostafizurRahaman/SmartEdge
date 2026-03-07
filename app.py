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
    use_adaptive = request.form.get('adaptive', 'false').lower() == 'true'
    block_size = int(request.form.get('block_size', 64))

    before = evaluate(bgr, reference=ref_bgr)
    sharpened_global = smartedge_sharpen(bgr, w=w, alpha_mode=alpha_mode)
    after_global = evaluate(sharpened_global, reference=ref_bgr)

    metric_keys = ["Pm", "mu", "Lm", "NIQE"]
    if ref_bgr is not None:
        metric_keys += ["PSNR", "SSIM"]

    def _make_rows(before_dict, after_dict):
        rows = []
        for key in metric_keys:
            if key not in before_dict:
                continue
            b = before_dict[key]
            a = after_dict[key]
            change, result = _arrow(b, a, EXPECTED[key], key=key)
            rows.append([key, f"{b:.4f}", f"{a:.4f}", change, EXPECTED[key], result])
        return rows

    response = dict(
        original=_to_base64_png(bgr),
        sharpened=_to_base64_png(sharpened_global),
        metrics=_make_rows(before, after_global),
    )

    if use_adaptive:
        sharpened_adaptive = smartedge_sharpen(
            bgr, w=w, alpha_mode=alpha_mode,
            adaptive=True, block_size=block_size,
        )
        after_adaptive = evaluate(sharpened_adaptive, reference=ref_bgr)
        response['sharpened_adaptive'] = _to_base64_png(sharpened_adaptive)
        response['metrics_adaptive'] = _make_rows(before, after_adaptive)

    return jsonify(**response)


if __name__ == '__main__':
    app.run(debug=False, port=7860)
