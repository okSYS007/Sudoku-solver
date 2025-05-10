import cv2
import os
import numpy as np

# === Настройки ===
CELL_SIZE = 90
TEMPLATE_PATH = "numbersBD"  # Папка с шаблонами: 1.png ... 9.png
IMAGE_PATH = "image.png"     # Поле судоку

# === Загрузка изображения поля ===
image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

# Нарезка поля на 81 ячейку
cells = []
for i in range(9):
    row = []
    for j in range(9):
        y1, y2 = i * CELL_SIZE, (i + 1) * CELL_SIZE
        x1, x2 = j * CELL_SIZE, (j + 1) * CELL_SIZE
        cell = image[y1:y2, x1:x2]
        row.append(cell)
    cells.append(row)

# === Загрузка шаблонов 1-9 ===
templates = {}
for i in range(1, 10):
    path = os.path.join(TEMPLATE_PATH, f"{i}.png")
    tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if tmpl is not None:
        templates[i] = tmpl

# === Сравнение через matchTemplate ===
def match_digit(cell):
    best_digit = ""
    best_score = 0.65  # Чем ближе к 1, тем точнее
    for digit, tmpl in templates.items():
        cell_resized = cv2.resize(cell, tmpl.shape[::-1])
        res = cv2.matchTemplate(cell_resized, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_digit = digit
    return best_digit

# === Распознавание всех ячеек ===
sudoku_grid = []
for row in cells:
    sudoku_row = []
    for cell in row:
        digit = match_digit(cell)
        sudoku_row.append(digit)
    sudoku_grid.append(sudoku_row)

# === Вывод результата ===
for row in sudoku_grid:
    print(row)
