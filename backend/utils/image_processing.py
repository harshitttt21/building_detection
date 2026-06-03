from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class FilterConfig:
    min_area: int = 300
    max_area: int = 200000
    min_aspect_ratio: float = 0.25
    max_aspect_ratio: float = 4.0
    min_extent: float = 0.25
    min_compactness: float = 0.18


def generate_masks(mask_generator: Any, image_rgb: np.ndarray) -> list[dict[str, Any]]:
    return mask_generator.generate(image_rgb)


def _mask_metrics(mask: np.ndarray) -> tuple[int, float, float, float]:
    binary = mask.astype(np.uint8)
    area = int(binary.sum())
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return area, 0.0, 0.0, 0.0

    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if contour_area <= 0:
        return area, 0.0, 0.0, 0.0

    x, y, width, height = cv2.boundingRect(contour)
    aspect_ratio = width / max(height, 1)
    rect_area = float(width * height) if width > 0 and height > 0 else 0.0
    extent = contour_area / rect_area if rect_area > 0 else 0.0
    perimeter = float(cv2.arcLength(contour, True))
    compactness = (4.0 * np.pi * contour_area) / (perimeter * perimeter) if perimeter > 0 else 0.0
    return area, aspect_ratio, extent, compactness


def filter_masks(masks: list[dict[str, Any]], config: FilterConfig) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []

    for mask in masks:
        segmentation = mask.get("segmentation")
        if segmentation is None:
            continue

        area, aspect_ratio, extent, compactness = _mask_metrics(segmentation)
        if area < config.min_area or area > config.max_area:
            continue
        if not (config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio):
            continue
        if extent < config.min_extent:
            continue
        if compactness < config.min_compactness:
            continue

        mask_copy = dict(mask)
        mask_copy["computed_area"] = area
        mask_copy["aspect_ratio"] = aspect_ratio
        mask_copy["extent"] = extent
        mask_copy["compactness"] = compactness
        filtered.append(mask_copy)

    return filtered


def combine_masks(masks: list[dict[str, Any]], image_shape: tuple[int, int, int]) -> np.ndarray:
    combined = np.zeros(image_shape[:2], dtype=np.uint8)
    for mask in masks:
        segmentation = mask.get("segmentation")
        if segmentation is None:
            continue
        combined = np.maximum(combined, segmentation.astype(np.uint8))

    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    return combined


def count_buildings(binary_mask: np.ndarray, min_component_area: int = 200) -> tuple[int, np.ndarray]:
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    refined = np.zeros_like(binary_mask)
    count = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue
        refined[labels == label] = 255
        count += 1

    return count, refined


def draw_output(image_bgr: np.ndarray, binary_mask: np.ndarray, count: int) -> np.ndarray:
    annotated = image_bgr.copy()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
    overlay = annotated.copy()
    overlay[binary_mask > 0] = (0, 180, 0)
    annotated = cv2.addWeighted(annotated, 0.75, overlay, 0.25, 0)

    cv2.rectangle(annotated, (12, 12), (330, 72), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"Buildings: {count}",
        (24, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated
