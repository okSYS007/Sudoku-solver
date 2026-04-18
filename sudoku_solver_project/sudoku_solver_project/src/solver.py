from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Tuple

Board = List[List[int]]


class SudokuError(Exception):
    pass


def validate_board(board: Board) -> None:
    if len(board) != 9 or any(len(row) != 9 for row in board):
        raise SudokuError('Поле должно быть размером 9x9.')

    for r in range(9):
        for c in range(9):
            value = board[r][c]
            if not 0 <= value <= 9:
                raise SudokuError(f'Недопустимое значение {value} в клетке ({r + 1}, {c + 1}).')
            if value == 0:
                continue

            board[r][c] = 0
            if not is_valid(board, r, c, value):
                board[r][c] = value
                raise SudokuError(
                    f'Исходное поле некорректно: цифра {value} конфликтует в клетке ({r + 1}, {c + 1}).'
                )
            board[r][c] = value


def find_empty(board: Board) -> Optional[Tuple[int, int]]:
    best_cell: Optional[Tuple[int, int]] = None
    best_count = 10

    for row in range(9):
        for col in range(9):
            if board[row][col] != 0:
                continue

            candidates = get_candidates(board, row, col)
            if len(candidates) < best_count:
                best_count = len(candidates)
                best_cell = (row, col)

            if best_count == 1:
                return best_cell

    return best_cell


def get_candidates(board: Board, row: int, col: int) -> list[int]:
    return [num for num in range(1, 10) if is_valid(board, row, col, num)]


def is_valid(board: Board, row: int, col: int, num: int) -> bool:
    if any(board[row][index] == num for index in range(9)):
        return False

    if any(board[index][col] == num for index in range(9)):
        return False

    start_row = (row // 3) * 3
    start_col = (col // 3) * 3

    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if board[r][c] == num:
                return False

    return True


def solve_in_place(board: Board) -> bool:
    empty = find_empty(board)
    if empty is None:
        return True

    row, col = empty
    for num in get_candidates(board, row, col):
        board[row][col] = num
        if solve_in_place(board):
            return True
        board[row][col] = 0

    return False


def solve(board: Board) -> Board:
    validate_board(board)
    solved = deepcopy(board)
    if not solve_in_place(solved):
        raise SudokuError('Не удалось найти решение для этого судоку.')
    return solved
