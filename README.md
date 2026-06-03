# Satellite Building Counter with SAM

This project detects and counts buildings in satellite images using Meta's Segment Anything Model (SAM) with a segmentation-first pipeline.

## What it does

- Accepts uploaded satellite images.
- Generates SAM masks automatically.
- Filters masks using area and shape heuristics.
- Combines masks into a binary building map.
- Counts buildings with OpenCV connected components.
- Returns an annotated image, an optional mask image, and the total count.

## Folder structure

- `backend/` Flask API and SAM pipeline.
- `frontend/` Browser UI.
- `static/` Generated output images.
- `uploads/` Uploaded originals.

## Setup

1. Create and activate a Python virtual environment.
2. Use the same interpreter for every command. On this machine, `segment_anything` is installed for Python 3.11, so use `py -3.11` for setup and running.
3. Install PyTorch for your platform first. For Windows CPU, a common starting point is:

```bash
py -3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

4. Install the project dependencies:

```bash
py -3.11 -m pip install -r requirements.txt
```

5. Download a pretrained SAM checkpoint from Meta and place it here:

```text
backend/model/checkpoints/sam_vit_b_01ec64.pth
```

You can also use `sam_vit_h_4b8939.pth` by updating `checkpoint_path` or the backend code.

## Run the app

Start the Flask backend:

```bash
py -3.11 backend/app.py
```

Open the app in your browser:

```text
http://127.0.0.1:5000
```

## API

`POST /detect`

Form fields:

- `image` - image file upload.
- `min_area` - minimum mask area.
- `max_area` - maximum mask area.
- `model_type` - `vit_b` or `vit_h`.

Response includes:

- `building_count`
- `annotated_image_url`
- `mask_image_url`
- `processing_time_ms`

## Notes

- This is a pretrained inference-only pipeline. No training is required.
- The shape filtering is intentionally conservative to reduce false positives.
- For better accuracy on your dataset, tune the area thresholds and the shape filters in `backend/utils/image_processing.py`.
