# Translation Overlay

An always-on-top overlay that captures a screen region with OCR every few seconds
and shows the translation in a floating window — designed for language learners.

## Features

- Auto or Manual capture mode
- Selectable screen region (DPI-aware, supports 4K/HiDPI)
- Input language selector: English / French / Spanish
- Word-by-word lookup panel with cached translations
- Adaptive polling — slows down when screen is idle
- Hotkey `Ctrl+Shift+H` to show/hide

## Requirements

- Python 3.9+
- Windows (uses `PIL.ImageGrab` for screen capture)
- `tkinter` — included with Python

## Setup

```bat
:: Create and activate a virtual environment
uv venv
.\.venv\Scripts\activate

:: Install dependencies
uv pip install rapidocr-onnxruntime Pillow numpy deep-translator keyboard
```

> **Note:** Do NOT install `paddlepaddle` or `paddleocr`.
> This project uses `rapidocr-onnxruntime` which bundles its own ONNX models
> and does not require PaddlePaddle.



**Prerequisites:**

| Requirement | How to check |
|---|---|
| NVIDIA GPU | `nvidia-smi` |
| CUDA 11.8 **or** 12.x | `nvcc --version` or `nvidia-smi` top-right corner |
| cuDNN (matching CUDA version) | installed alongside CUDA |

> `onnxruntime-gpu` 1.19+ supports CUDA 11.8 and 12.x.
> If `nvidia-smi` shows CUDA 11.x (< 11.8), install `onnxruntime-gpu==1.16.*` instead.

Once `onnxruntime-gpu` is installed, ONNX Runtime automatically selects the
`CUDAExecutionProvider` when a compatible GPU is found — no code changes needed.

To verify GPU is being used, run Python and check:

```python
import onnxruntime
print(onnxruntime.get_available_providers())
# Should include 'CUDAExecutionProvider'
```

## Run

```bat
python translation_overlay.py
```

## Dependencies

| Package | Purpose |
|---|---|
| `rapidocr-onnxruntime` | OCR engine (PaddleOCR models via ONNX, no PaddlePaddle needed) |
| `Pillow` | Screen capture (`ImageGrab`) and image resizing |
| `numpy` | Image array conversion for OCR input |
| `deep-translator` | Google Translate API wrapper |
| `keyboard` | Global hotkey (`Ctrl+Shift+H`) — optional |

## Configuration

Edit the constants at the top of `translation_overlay.py`:

| Constant | Default | Description |
|---|---|---|
| `TRANSLATE_TO` | `zh-CN` | Target language code |
| `OCR_INTERVAL` | `2.0` | Seconds between captures (auto mode) |
| `MAX_OCR_WIDTH` | `1280` | Image is downscaled to this width before OCR |
| `ALPHA` | `0.85` | Window transparency |
