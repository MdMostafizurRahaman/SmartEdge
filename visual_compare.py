import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import cv2
from input import load_image
from sharpen import smartedge_sharpen
from metrics import evaluate
from prewitt_gradient import sobel_gradient, prewitt_gradient, scharr_gradient

img, _ = load_image('Input/diagonal_rich.png')

kernels = [
    ('Original (Blurred)', None),
    ('Prewitt', prewitt_gradient),
    ('Sobel',   sobel_gradient),
    ('Scharr',  scharr_gradient),
]

panels = []
metric_rows = []

for name, fn in kernels:
    if fn is None:
        out = img.copy()
    else:
        out = smartedge_sharpen(img, gradient_fn=fn)

    arr = np.clip(out, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    # label bar
    bar = np.ones((42, 512, 3), dtype=np.uint8) * 30
    cv2.putText(bar, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # metrics info strip
    info = np.ones((52, 512, 3), dtype=np.uint8) * 15
    if fn is not None:
        m = evaluate(out)
        metric_rows.append((name, m))
        txt = f"Pm={m['Pm']:.1f}  mu={m['mu']:.4f}  NIQE={m['NIQE']:.4f}"
        cv2.putText(info, txt, (5, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
    else:
        cv2.putText(info, 'Input — Gaussian blurred (sigma=1.2)', (5, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    panels.append(np.vstack([bar, arr, info]))

comparison = np.hstack(panels)
cv2.imwrite('Output/kernel_comparison.png', comparison)
print('Saved: Output/kernel_comparison.png')
print()

names = ['Prewitt', 'Sobel', 'Scharr']
all_m = {r[0]: r[1] for r in metric_rows}
print(f"{'Metric':<8}  {'Prewitt':>10}  {'Sobel':>10}  {'Scharr':>10}  Best")
print('-' * 58)
for key, better in [('Pm', 'high'), ('mu', 'low'), ('NIQE', 'low')]:
    vals = [all_m[n][key] for n in names]
    bi = vals.index(min(vals)) if better == 'low' else vals.index(max(vals))
    row = f'{key:<8}'
    for i, v in enumerate(vals):
        mark = '***' if i == bi else '   '
        row += f'  {v:>10.4f} {mark}'
    row += f'  {names[bi]}'
    print(row)
