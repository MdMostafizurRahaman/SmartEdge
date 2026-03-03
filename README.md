# SmartEdge

Automatic image sharpening filter based on Prewitt gradient and Retinex-inspired contrast — no manual tuning required.

## What It Does

Makes blurry or poorly-lit images sharper by enhancing edges while preserving natural appearance.

- **Automatic** — adapts to each image, no parameter tuning needed
- **Gentle** — less aggressive than standard Laplacian sharpening
- **Versatile** — works on medical images, low-light photos, aerial imagery, compressed images

Based on the IEEE paper: [A New Image Sharpening Filter Based on Gradient and Retinex-Inspired Contrast](https://ieeexplore.ieee.org/document/11192240)

## How It Works

1. Computes **Prewitt gradient** magnitude (edge strength)
2. Computes **Retinex-inspired contrast** (local luminance ratios)
3. Compares them to find edges with improvable visibility
4. Builds an adaptive **Laplacian-like kernel** from the ratio
5. Applies the kernel to sharpen edges without affecting uniform regions

## Tech Stack

Python, OpenCV, SciPy, NumPy, Flask

## Quick Start

### Web UI

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860` in your browser. Upload an image, pick parameters, click **Sharpen**.

### Command Line

```bash
python main.py --input image.jpg --output sharp.jpg
```

Options:
- `--w 3|5|7` — window size for contrast computation (default: 3)
- `--alpha 1|auto` — scaling factor, `auto` estimates from image content (default: 1)
- `--epsilon 0.05` — threshold for edge detection

### Evaluate Metrics

```bash
python evaluate.py --input image.jpg
python evaluate.py --input blurred.jpg --reference original.jpg
```

## Evaluation Metrics

The filter is validated using 6 metrics from the paper (Section III-C):

| Metric | Measures | After Sharpening |
|--------|----------|-----------------|
| **Pm** | Edge strength (Prewitt gradient magnitude) | Should increase |
| **mu** | Edge thickness (Crete et al. blur metric) | Should decrease |
| **Lm** | Mean luminance | Should stay stable |
| **NIQE** | Image naturalness | Should stay stable |
| **PSNR** | Similarity to reference (full-reference) | Should increase |
| **SSIM** | Structural similarity (full-reference) | Should increase |

PSNR and SSIM require a reference (undistorted) image. The other 4 are no-reference metrics.

## Project Structure

```
SmartEdge/
├── app.py                 # Flask web UI
├── main.py                # CLI entry point
├── evaluate.py            # Metric evaluation script
├── sharpen.py             # Full sharpening pipeline
├── prewitt_gradient.py    # Prewitt gradient magnitude
├── contrast.py            # Retinex-inspired contrast (phi)
├── c_estimation.py        # Rho map, R set cleanup, c estimate
├── kernel.py              # Laplacian-like kernel from c
├── filters.py             # Median/max filters, convolution
├── metrics.py             # Pm, mu, Lm, NIQE, PSNR, SSIM
├── input.py               # Image load/save (cv2/PIL)
├── templates/
│   └── index.html         # Web UI template
├── requirements.txt       # Python dependencies
└── README.md
```

## Deploy on Render

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your repo and set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Deploy (free tier available)
