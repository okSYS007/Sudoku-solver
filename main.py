import cv2
import numpy as np
import pytesseract

# если Windows и tesseract не в PATH:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left
    return rect

def four_point_transform(image, pts, size=900):
    rect = order_points(pts)
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (size, size))
    return warped, M

def find_sudoku_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > max_area:
            biggest = approx
            max_area = area

    if biggest is None:
        raise ValueError("Не удалось найти поле судоку")

    pts = biggest.reshape(4, 2).astype(np.float32)
    warped, M = four_point_transform(image, pts)
    return warped

def preprocess_cell(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # убираем отступы от границ клетки
    h, w = gray.shape
    margin = int(min(h, w) * 0.12)
    gray = gray[margin:h-margin, margin:w-margin]

    # бинаризация
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # шум
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    return th

def recognize_digit(cell_img):
    th = preprocess_cell(cell_img)

    # если пикселей мало — клетка пустая
    non_zero = cv2.countNonZero(th)
    area = th.shape[0] * th.shape[1]
    fill_ratio = non_zero / area

    if fill_ratio < 0.03:
        return 0

    # увеличим для OCR
    resized = cv2.resize(th, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    config = r'--psm 10 -c tessedit_char_whitelist=123456789'
    text = pytesseract.image_to_string(resized, config=config).strip()

    if text.isdigit() and text in "123456789":
        return int(text)

    return 0

def extract_board_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Не удалось открыть изображение")

    warped = find_sudoku_grid(image)
    size = warped.shape[0]
    cell_size = size // 9

    board = []
    fixed_mask = []

    for r in range(9):
        row = []
        mask_row = []
        for c in range(9):
            y1 = r * cell_size
            y2 = (r + 1) * cell_size
            x1 = c * cell_size
            x2 = (c + 1) * cell_size

            cell = warped[y1:y2, x1:x2]
            digit = recognize_digit(cell)
            row.append(digit)
            mask_row.append(digit != 0)

        board.append(row)
        fixed_mask.append(mask_row)

    return warped, board, fixed_mask