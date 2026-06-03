from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ModuleNotFoundError as exc:  # pragma: no cover - import-time environment guard
    raise ModuleNotFoundError(
        "segment_anything is not installed in the active Python environment. "
        "Run the app with Python 3.11 (for example: py -3.11 backend/app.py) and install dependencies with "
        "py -3.11 -m pip install -r requirements.txt."
    ) from exc

from backend.utils.image_processing import FilterConfig, combine_masks, count_buildings, draw_output, filter_masks, generate_masks


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "sam_vit_b_01ec64.pth"
MAX_SAM_DIMENSION = 1024


def _resize_for_sam(image_bgr: np.ndarray, max_dimension: int = MAX_SAM_DIMENSION) -> tuple[np.ndarray, float]:
    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dimension:
        return image_bgr, 1.0

    scale = max_dimension / float(longest_side)
    resized = cv2.resize(image_bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = target_shape
    if mask.shape[:2] == (target_height, target_width):
        return mask.astype(np.uint8)

    resized = cv2.resize(mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


@lru_cache(maxsize=2)
def load_model(model_type: str | None = None, checkpoint_path: str | None = None, device: str | None = None) -> SamAutomaticMaskGenerator:
    chosen_model = (model_type or "vit_b").lower()
    chosen_checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cpu = chosen_device == "cpu"

    if chosen_model not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {chosen_model}")
    if not chosen_checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {chosen_checkpoint}. Download sam_vit_b_01ec64.pth or sam_vit_h_4b8939.pth and place it there."
        )

    sam = sam_model_registry[chosen_model](checkpoint=str(chosen_checkpoint))
    sam.to(device=chosen_device)
    sam.eval()

    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=16 if is_cpu else 32,
        pred_iou_thresh=0.86 if is_cpu else 0.88,
        stability_score_thresh=0.88 if is_cpu else 0.92,
        box_nms_thresh=0.7,
        crop_n_layers=0 if is_cpu else 1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=80,
    )


def process_image(
    image_bytes: bytes,
    *,
    min_area: int,
    max_area: int,
    model_type: str | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Unable to decode the uploaded image.")

    resized_bgr, scale = _resize_for_sam(image_bgr)
    image_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    mask_generator = load_model(model_type=model_type, checkpoint_path=checkpoint_path)

    masks = generate_masks(mask_generator, image_rgb)
    original_height, original_width = image_bgr.shape[:2]
    target_shape = (original_height, original_width)

    if scale != 1.0:
        for mask in masks:
            segmentation = mask.get("segmentation")
            if segmentation is None:
                continue
            mask["segmentation"] = _resize_mask(segmentation, target_shape)

    filtered_masks = filter_masks(
        masks,
        FilterConfig(
            min_area=min_area,
            max_area=max_area,
        ),
    )

    combined_mask = combine_masks(filtered_masks, image_bgr.shape)
    count, refined_mask = count_buildings(combined_mask)
    annotated_bgr = draw_output(image_bgr, refined_mask, count)

    return {
        "count": count,
        "annotated_bgr": annotated_bgr,
        "mask_binary": refined_mask,
        "filtered_mask_count": len(filtered_masks),
    }
