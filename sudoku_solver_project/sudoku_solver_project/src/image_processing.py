from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import (
    CELL_INNER_MARGIN,
    DRAW_COLOR_BGR,
    DRAW_FONT_SCALE,
    DRAW_THICKNESS,
    GRID_DETECTION_MIN_AREA,
    WARP_SIZE,
)


class ImageProcessingError(Exception):
    pass


@dataclass
class BoardExtraction:
    original_image: np.ndarray
    warped_image: np.ndarray
    transform_matrix: np.ndarray
    inverse_transform_matrix: np.ndarray
    grid_points: np.ndarray



def read_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ImageProcessingError(f'Не удалось открыть изображение: {image_path}')
    return image



def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype='float32')
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1)

    rect[0] = points[np.argmin(point_sum)]
    rect[2] = points[np.argmax(point_sum)]
    rect[1] = points[np.argmin(point_diff)]
    rect[3] = points[np.argmax(point_diff)]
    return rect



def find_grid_points(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_polygon = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < GRID_DETECTION_MIN_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(polygon) == 4 and area > best_area:
            best_polygon = polygon
            best_area = area

    if best_polygon is None:
        raise ImageProcessingError('Не удалось найти внешнюю рамку судоку.')

    return order_points(best_polygon.reshape(4, 2).astype('float32'))



def extract_board(image_path: str | Path) -> BoardExtraction:
    original = read_image(image_path)
    points = find_grid_points(original)

    destination = np.array(
        [[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]],
        dtype='float32',
    )

    matrix = cv2.getPerspectiveTransform(points, destination)
    inverse_matrix = cv2.getPerspectiveTransform(destination, points)
    warped = cv2.warpPerspective(original, matrix, (WARP_SIZE, WARP_SIZE))

    return BoardExtraction(
        original_image=original,
        warped_image=warped,
        transform_matrix=matrix,
        inverse_transform_matrix=inverse_matrix,
        grid_points=points,
    )



def iterate_cells(warped_image: np.ndarray) -> Iterable[tuple[int, int, np.ndarray]]:
    cell_size = warped_image.shape[0] // 9
    for row in range(9):
        for col in range(9):
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size
            yield row, col, warped_image[y1:y2, x1:x2].copy()



def crop_inner_region(cell_image: np.ndarray, margin: int = CELL_INNER_MARGIN) -> np.ndarray:
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    return gray[margin:-margin, margin:-margin]



def draw_solution_on_warped(
    warped_image: np.ndarray,
    original_board: list[list[int]],
    solved_board: list[list[int]],
) -> np.ndarray:
    output = warped_image.copy()
    cell_size = output.shape[0] // 9

    for row in range(9):
        for col in range(9):
            if original_board[row][col] != 0:
                continue

            text = str(solved_board[row][col])
            x = int(col * cell_size + cell_size * 0.32)
            y = int(row * cell_size + cell_size * 0.72)

            cv2.putText(
                output,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                DRAW_FONT_SCALE,
                DRAW_COLOR_BGR,
                DRAW_THICKNESS,
                cv2.LINE_AA,
            )

    return output



def overlay_solution_on_original(
    extraction: BoardExtraction,
    original_board: list[list[int]],
    solved_board: list[list[int]],
) -> np.ndarray:
    warped_with_solution = draw_solution_on_warped(
        extraction.warped_image,
        original_board,
        solved_board,
    )

    original_h, original_w = extraction.original_image.shape[:2]
    projected = cv2.warpPerspective(
        warped_with_solution,
        extraction.inverse_transform_matrix,
        (original_w, original_h),
    )

    original_grid = cv2.warpPerspective(
        extraction.warped_image,
        extraction.inverse_transform_matrix,
        (original_w, original_h),
    )

    difference = cv2.absdiff(projected, original_grid)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_difference, 10, 255, cv2.THRESH_BINARY)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    output = extraction.original_image.copy()
    output = np.where(mask_bgr > 0, projected, output)
    return output.astype(np.uint8)



def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ImageProcessingError(f'Не удалось сохранить изображение: {path}')
