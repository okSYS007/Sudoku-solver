from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from .autofill import AutoFillError, fill_sudoku_on_screen
from .service import SolveResult, SudokuError, SudokuService


class SudokuApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('Sudoku Solver')
        self.geometry('1180x760')
        self.minsize(1100, 700)

        self.service = SudokuService()
        self.current_image_path: Path | None = None
        self.current_result: SolveResult | None = None
        self._preview_refs: list[ImageTk.PhotoImage] = []

        self._build_ui()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill='both', expand=True)

        controls = ttk.Frame(outer)
        controls.pack(fill='x', pady=(0, 12))

        ttk.Button(controls, text='Открыть изображение', command=self.open_image).pack(side='left')
        ttk.Button(controls, text='Решить', command=self.solve_current_image).pack(side='left', padx=8)
        ttk.Button(controls, text='Сохранить результат', command=self.save_result).pack(side='left')
        ttk.Button(controls, text='Автозаполнение на сайте', command=self.autofill_to_site).pack(side='left', padx=8)

        self.status_var = tk.StringVar(value='Открой скриншот судоку и нажми «Решить».')
        ttk.Label(controls, textvariable=self.status_var).pack(side='left', padx=16)

        content = ttk.Panedwindow(outer, orient='horizontal')
        content.pack(fill='both', expand=True)

        left = ttk.Frame(content)
        right = ttk.Frame(content)
        content.add(left, weight=3)
        content.add(right, weight=2)

        previews = ttk.Frame(left)
        previews.pack(fill='both', expand=True)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        previews.rowconfigure(1, weight=1)

        ttk.Label(previews, text='Исходное изображение').grid(row=0, column=0, sticky='w')
        ttk.Label(previews, text='Результат').grid(row=0, column=1, sticky='w')

        self.original_preview = ttk.Label(previews, anchor='center')
        self.original_preview.grid(row=1, column=0, sticky='nsew', padx=(0, 8))

        self.result_preview = ttk.Label(previews, anchor='center')
        self.result_preview.grid(row=1, column=1, sticky='nsew', padx=(8, 0))

        ttk.Label(right, text='Распознанное поле').pack(anchor='w')
        self.original_board_text = tk.Text(right, height=12, width=30, font=('Consolas', 14))
        self.original_board_text.pack(fill='x', pady=(4, 12))

        ttk.Label(right, text='Решение').pack(anchor='w')
        self.solved_board_text = tk.Text(right, height=12, width=30, font=('Consolas', 14))
        self.solved_board_text.pack(fill='x', pady=(4, 12))

        ttk.Label(right, text='Подсказка').pack(anchor='w')
        hint = (
            'Для автозаполнения сайт должен быть уже открыт. '\
            'Программа спросит координаты левого верхнего угла сетки и размер поля в пикселях.'
        )
        ttk.Label(right, text=hint, wraplength=320, justify='left').pack(anchor='w')

    @staticmethod
    def _board_to_text(board: list[list[int]]) -> str:
        lines = []
        for row_index, row in enumerate(board):
            chunks = []
            for col_index, value in enumerate(row):
                chunks.append(str(value) if value != 0 else '.')
                if col_index in (2, 5):
                    chunks.append('|')
            lines.append(' '.join(chunks))
            if row_index in (2, 5):
                lines.append('-' * 25)
        return '\n'.join(lines)

    def _show_preview(self, label: ttk.Label, image_bgr, max_size: tuple[int, int] = (420, 620)) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(max_size)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        self._preview_refs.append(photo)
        if len(self._preview_refs) > 4:
            self._preview_refs = self._preview_refs[-4:]

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title='Выбери скриншот судоку',
            filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp')],
        )
        if not path:
            return

        self.current_image_path = Path(path)
        self.current_result = None
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror('Ошибка', 'Не удалось открыть изображение.')
            return

        self._show_preview(self.original_preview, image)
        self.result_preview.configure(image='')
        self.original_board_text.delete('1.0', tk.END)
        self.solved_board_text.delete('1.0', tk.END)
        self.status_var.set(f'Выбран файл: {self.current_image_path.name}')

    def solve_current_image(self) -> None:
        if self.current_image_path is None:
            messagebox.showinfo('Нет изображения', 'Сначала выбери скриншот.')
            return

        try:
            result = self.service.solve_from_image(self.current_image_path)
        except SudokuError as error:
            messagebox.showerror('Ошибка решения', str(error))
            return
        except Exception as error:
            messagebox.showerror('Ошибка', str(error))
            return

        self.current_result = result
        self.original_board_text.delete('1.0', tk.END)
        self.original_board_text.insert('1.0', self._board_to_text(result.original_board))

        self.solved_board_text.delete('1.0', tk.END)
        self.solved_board_text.insert('1.0', self._board_to_text(result.solved_board))

        overlaid = cv2.imread(str(self.current_image_path))
        if overlaid is not None:
            from .image_processing import overlay_solution_on_original
            overlaid = overlay_solution_on_original(result.extraction, result.original_board, result.solved_board)
            self._show_preview(self.result_preview, overlaid)

        self.status_var.set('Судоку решено. Можно сохранить результат или включить автозаполнение.')

    def save_result(self) -> None:
        if self.current_result is None:
            messagebox.showinfo('Нет результата', 'Сначала реши судоку.')
            return

        default_name = 'solved_overlay.png'
        path = filedialog.asksaveasfilename(
            title='Куда сохранить результат',
            defaultextension='.png',
            initialfile=default_name,
            filetypes=[('PNG image', '*.png')],
        )
        if not path:
            return

        self.service.save_overlay_result(self.current_result, path)
        self.status_var.set(f'Файл сохранен: {Path(path).name}')
        messagebox.showinfo('Готово', f'Результат сохранен:\n{path}')

    def autofill_to_site(self) -> None:
        if self.current_result is None:
            messagebox.showinfo('Нет решения', 'Сначала реши судоку.')
            return

        top_left_x = simpledialog.askinteger('Автозаполнение', 'X левого верхнего угла поля:')
        if top_left_x is None:
            return
        top_left_y = simpledialog.askinteger('Автозаполнение', 'Y левого верхнего угла поля:')
        if top_left_y is None:
            return
        grid_size = simpledialog.askinteger('Автозаполнение', 'Размер сетки по ширине в пикселях:')
        if grid_size is None:
            return

        try:
            fill_sudoku_on_screen(
                self.current_result.original_board,
                self.current_result.solved_board,
                top_left_x,
                top_left_y,
                grid_size,
            )
        except AutoFillError as error:
            messagebox.showerror('Ошибка автозаполнения', str(error))
            return

        messagebox.showinfo('Готово', 'Автозаполнение выполнено.')


def run_gui() -> None:
    app = SudokuApp()
    app.mainloop()
