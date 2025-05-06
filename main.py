import cv2
import os
from skimage.metrics import structural_similarity as ssim

# === Настройки ===
CELL_SIZE = 90  # Размер ячейки
TEMPLATE_PATH = "numbersBD"  # Папка с шаблонами 1.png - 9.png

# === Загрузка изображения поля ===
image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Проверка размера поля
height, width = image.shape
if height < 810 or width < 810:
    raise ValueError("Поле должно быть не менее 810x810 пикселей")

# === Нарезка на 81 ячейку ===
cells = []
for i in range(9):
    row = []
    for j in range(9):
        y1, y2 = i * CELL_SIZE, (i + 1) * CELL_SIZE
        x1, x2 = j * CELL_SIZE, (j + 1) * CELL_SIZE
        cell = image[y1:y2, x1:x2]
        row.append(cell)
    cells.append(row)

# === Загрузка шаблонов цифр ===
templates = {}
for i in range(1, 10):
    tmpl = cv2.imread(os.path.join(TEMPLATE_PATH, f"{i}.png"), cv2.IMREAD_GRAYSCALE)
    if tmpl is not None:
        templates[i] = cv2.resize(tmpl, (CELL_SIZE, CELL_SIZE))

# === Сравнение ячейки с шаблонами ===
def match_digit(cell):
    cell_resized = cv2.resize(cell, (CELL_SIZE, CELL_SIZE))
    best_digit = ""
    best_score = 0.7
    for digit, tmpl in templates.items():
        score = ssim(cell_resized, tmpl)
        if score > best_score:
            best_score = score
            best_digit = digit
    return best_digit

# === Распознавание всех цифр ===
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
