from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = PROJECT_ROOT / 'assets' / 'sample_input.png'
OUTPUT_DIR = PROJECT_ROOT / 'assets' / 'templates'
WARP_SIZE = 900
KNOWN_DIGITS = {
    (0, 6): 4, (0, 7): 1,
    (1, 1): 8, (1, 8): 3,
    (2, 4): 5, (2, 5): 6,
    (3, 1): 9, (3, 2): 4, (3, 5): 3,
    (4, 2): 2, (4, 6): 3, (4, 8): 6,
    (5, 2): 7, (5, 5): 9,
    (6, 2): 5, (6, 3): 4, (6, 6): 6,
    (7, 1): 4, (7, 4): 2,
    (8, 1): 2, (8, 4): 6, (8, 5): 1, (8, 8): 9,
}


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype='float32')
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def extract_glyph(cell_image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    inner = gray[12:88, 12:88]
    _, threshold = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if area < 80 or height < 18 or width < 5:
            continue
        boxes.append((x, y, width, height))

    if not boxes:
        return None

    x1 = min(item[0] for item in boxes)
    y1 = min(item[1] for item in boxes)
    x2 = max(item[0] + item[2] for item in boxes)
    y2 = max(item[1] + item[3] for item in boxes)

    glyph = threshold[y1:y2, x1:x2]
    height, width = glyph.shape
    side = max(height, width) + 20
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - height) // 2
    x_offset = (side - width) // 2
    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = glyph
    normalized = cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_AREA)
    _, normalized = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)
    return normalized


def main() -> None:
    image = cv2.imread(str(SAMPLE_PATH))
    if image is None:
        raise FileNotFoundError(f'Не найден sample input: {SAMPLE_PATH}')

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
    contour = max(
        (cnt for cnt in contours if len(cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)) == 4),
        key=cv2.contourArea,
    )
    polygon = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True).reshape(4, 2).astype('float32')
    matrix = cv2.getPerspectiveTransform(
        order_points(polygon),
        np.array([[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]], dtype='float32'),
    )
    warped = cv2.warpPerspective(image, matrix, (WARP_SIZE, WARP_SIZE))

    cell_size = WARP_SIZE // 9
    grouped: dict[int, list[np.ndarray]] = {digit: [] for digit in range(1, 10)}
    for (row, col), digit in KNOWN_DIGITS.items():
        cell = warped[row * cell_size:(row + 1) * cell_size, col * cell_size:(col + 1) * cell_size]
        glyph = extract_glyph(cell)
        if glyph is not None:
            grouped[digit].append(glyph)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for digit, glyphs in grouped.items():
        if not glyphs:
            continue
        average = np.mean(np.stack(glyphs, axis=0), axis=0).astype('uint8')
        _, average = cv2.threshold(average, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(OUTPUT_DIR / f'{digit}.png'), average)

    print(f'Шаблоны сохранены в: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
