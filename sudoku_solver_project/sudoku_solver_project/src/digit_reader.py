from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import (
    NORMALIZED_GLYPH_SIZE,
    OCR_CORR_THRESHOLD,
    OCR_MARGIN_THRESHOLD,
    OCR_MSE_THRESHOLD,
    TEMPLATES_DIR,
)
from .image_processing import crop_inner_region, iterate_cells

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


@dataclass
class DigitPrediction:
    value: int
    confidence: float
    method: str


class DigitReader:
    def __init__(self, templates_dir: str | Path | None = None) -> None:
        self.templates_dir = Path(templates_dir or TEMPLATES_DIR)
        self.templates = self._load_templates()
        if not self.templates:
            raise RuntimeError(f'Не найдены шаблоны цифр в папке: {self.templates_dir}')

    def _load_templates(self) -> dict[int, np.ndarray]:
        templates: dict[int, np.ndarray] = {}
        for digit in range(1, 10):
            path = self.templates_dir / f'{digit}.png'
            template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue
            _, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
            templates[digit] = template
        return templates

    @staticmethod
    def _normalize_glyph(binary_crop: np.ndarray) -> np.ndarray:
        height, width = binary_crop.shape
        side = max(height, width) + 20
        canvas = np.zeros((side, side), dtype=np.uint8)
        y_offset = (side - height) // 2
        x_offset = (side - width) // 2
        canvas[y_offset:y_offset + height, x_offset:x_offset + width] = binary_crop
        normalized = cv2.resize(
            canvas,
            (NORMALIZED_GLYPH_SIZE, NORMALIZED_GLYPH_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        _, normalized = cv2.threshold(normalized, 127, 255, cv2.THRESH_BINARY)
        return normalized

    def extract_glyph(self, cell_image: np.ndarray) -> Optional[np.ndarray]:
        inner = crop_inner_region(cell_image)
        _, threshold = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if area < 80 or height < 18 or width < 5:
                continue
            candidates.append((x, y, width, height))

        if not candidates:
            return None

        x1 = min(item[0] for item in candidates)
        y1 = min(item[1] for item in candidates)
        x2 = max(item[0] + item[2] for item in candidates)
        y2 = max(item[1] + item[3] for item in candidates)

        glyph = threshold[y1:y2, x1:x2]
        if glyph.size == 0:
            return None

        return self._normalize_glyph(glyph)

    def predict_from_template(self, glyph: np.ndarray) -> DigitPrediction:
        glyph_float = glyph.astype(np.float32) / 255.0

        best_digit = 0
        best_mse = 10.0
        second_best_mse = 10.0
        best_corr = -1.0

        for digit, template in self.templates.items():
            template_float = template.astype(np.float32) / 255.0
            mse = float(np.mean((glyph_float - template_float) ** 2))
            corr = float(np.corrcoef(glyph_float.flatten(), template_float.flatten())[0, 1])

            if mse < best_mse:
                second_best_mse = best_mse
                best_mse = mse
                best_corr = corr
                best_digit = digit
            elif mse < second_best_mse:
                second_best_mse = mse

        confidence = max(0.0, min(1.0, 1.0 - best_mse))
        margin = second_best_mse - best_mse

        if (
            best_mse <= OCR_MSE_THRESHOLD
            and margin >= OCR_MARGIN_THRESHOLD
            and best_corr >= OCR_CORR_THRESHOLD
        ):
            return DigitPrediction(best_digit, confidence, 'template')

        return DigitPrediction(0, confidence, 'template_low_confidence')

    def predict_with_tesseract(self, glyph: np.ndarray) -> DigitPrediction:
        if pytesseract is None:
            return DigitPrediction(0, 0.0, 'tesseract_unavailable')

        enlarged = cv2.resize(glyph, (128, 128), interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(
            enlarged,
            config='--psm 10 -c tessedit_char_whitelist=123456789',
        ).strip()

        if text.isdigit() and text in '123456789':
            return DigitPrediction(int(text), 0.6, 'tesseract')

        return DigitPrediction(0, 0.0, 'tesseract_failed')

    def predict_digit(self, cell_image: np.ndarray) -> DigitPrediction:
        glyph = self.extract_glyph(cell_image)
        if glyph is None:
            return DigitPrediction(0, 1.0, 'empty')

        template_prediction = self.predict_from_template(glyph)
        if template_prediction.value != 0:
            return template_prediction

        tesseract_prediction = self.predict_with_tesseract(glyph)
        if tesseract_prediction.value != 0:
            return tesseract_prediction

        return template_prediction

    def read_board(self, warped_image: np.ndarray) -> tuple[list[list[int]], list[list[DigitPrediction]]]:
        board = [[0 for _ in range(9)] for _ in range(9)]
        predictions = [[DigitPrediction(0, 0.0, 'unread') for _ in range(9)] for _ in range(9)]

        for row, col, cell in iterate_cells(warped_image):
            prediction = self.predict_digit(cell)
            board[row][col] = prediction.value
            predictions[row][col] = prediction

        return board, predictions
