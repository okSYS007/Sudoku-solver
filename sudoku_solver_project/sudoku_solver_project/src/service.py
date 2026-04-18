from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .digit_reader import DigitPrediction, DigitReader
from .image_processing import BoardExtraction, extract_board, overlay_solution_on_original, save_image
from .solver import SudokuError, solve


@dataclass
class SolveResult:
    extraction: BoardExtraction
    original_board: list[list[int]]
    solved_board: list[list[int]]
    predictions: list[list[DigitPrediction]]


class SudokuService:
    def __init__(self) -> None:
        self.reader = DigitReader()

    def solve_from_image(self, image_path: str | Path) -> SolveResult:
        extraction = extract_board(image_path)
        original_board, predictions = self.reader.read_board(extraction.warped_image)
        solved_board = solve(original_board)
        return SolveResult(
            extraction=extraction,
            original_board=original_board,
            solved_board=solved_board,
            predictions=predictions,
        )

    def save_overlay_result(self, result: SolveResult, output_path: str | Path) -> None:
        overlaid = overlay_solution_on_original(
            result.extraction,
            result.original_board,
            result.solved_board,
        )
        save_image(output_path, overlaid)


__all__ = ['SudokuError', 'SudokuService', 'SolveResult']
