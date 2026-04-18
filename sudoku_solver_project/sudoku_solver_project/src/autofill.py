from __future__ import annotations

import time

try:
    import pyautogui
except Exception:  # pragma: no cover
    pyautogui = None


class AutoFillError(Exception):
    pass



def fill_sudoku_on_screen(
    original_board: list[list[int]],
    solved_board: list[list[int]],
    top_left_x: int,
    top_left_y: int,
    grid_size: int,
    delay_before_start: float = 3.0,
    key_interval: float = 0.02,
) -> None:
    if pyautogui is None:
        raise AutoFillError('Модуль pyautogui не установлен. Установи зависимости из requirements.txt.')

    cell_size = grid_size / 9.0
    time.sleep(delay_before_start)

    for row in range(9):
        for col in range(9):
            if original_board[row][col] != 0:
                continue

            x = int(top_left_x + col * cell_size + cell_size / 2)
            y = int(top_left_y + row * cell_size + cell_size / 2)
            pyautogui.click(x, y)
            pyautogui.write(str(solved_board[row][col]), interval=key_interval)
