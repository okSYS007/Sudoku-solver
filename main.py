from PIL import Image
import pytesseract
import numpy as np
import cv2

# Load the image
image_path = "image.png"
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize image to ensure uniformity
resized = cv2.resize(gray, (450, 450))

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)

# Define cell size
cell_size = 50  # 450 / 9 = 50

# Extract digits from each cell
sudoku_grid = []

for row in range(9):
    row_data = []
    for col in range(9):
        x, y = col * cell_size, row * cell_size
        cell = thresh[y:y + cell_size, x:x + cell_size]

        # Slight padding to avoid grid lines
        margin = 5
        digit_cell = cell[margin:-margin, margin:-margin]

        # Resize digit cell for better OCR accuracy
        digit_cell_resized = cv2.resize(digit_cell, (100, 100))
        
        # Run OCR
        text = pytesseract.image_to_string(digit_cell_resized, config='--psm 10 digits')
        digit = int(text.strip()) if text.strip().isdigit() else 0
        row_data.append(digit)
    sudoku_grid.append(row_data)

sudoku_grid