from __future__ import annotations

import argparse
from pathlib import Path

from src.autofill import AutoFillError, fill_sudoku_on_screen
from src.config import OUTPUT_DIR
from src.gui import run_gui
from src.service import SudokuError, SudokuService


def format_board(board: list[list[int]]) -> str:
    return '\n'.join(' '.join(str(value) for value in row) for row in board)



def solve_command(args: argparse.Namespace) -> int:
    service = SudokuService()

    try:
        result = service.solve_from_image(args.image)
    except SudokuError as error:
        print(f'Ошибка решения: {error}')
        return 1
    except Exception as error:
        print(f'Ошибка: {error}')
        return 1

    print('Распознанное поле:')
    print(format_board(result.original_board))
    print('\nРешение:')
    print(format_board(result.solved_board))

    output_path = Path(args.output or (OUTPUT_DIR / 'solved_overlay.png'))
    service.save_overlay_result(result, output_path)
    print(f'\nРезультат сохранен: {output_path}')

    if args.autofill:
        if args.top_left_x is None or args.top_left_y is None or args.grid_size is None:
            print('Для автозаполнения нужно передать --top-left-x, --top-left-y и --grid-size.')
            return 1

        try:
            fill_sudoku_on_screen(
                result.original_board,
                result.solved_board,
                args.top_left_x,
                args.top_left_y,
                args.grid_size,
            )
        except AutoFillError as error:
            print(f'Ошибка автозаполнения: {error}')
            return 1

        print('Автозаполнение выполнено.')

    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Sudoku solver по скриншоту.')
    subparsers = parser.add_subparsers(dest='command')

    gui_parser = subparsers.add_parser('gui', help='Запустить графическое приложение.')
    gui_parser.set_defaults(func=lambda _: (run_gui(), 0)[1])

    solve_parser = subparsers.add_parser('solve', help='Решить судоку по изображению.')
    solve_parser.add_argument('--image', required=True, help='Путь к изображению.')
    solve_parser.add_argument('--output', help='Куда сохранить итоговое изображение.')
    solve_parser.add_argument('--autofill', action='store_true', help='После решения заполнить поле на сайте.')
    solve_parser.add_argument('--top-left-x', type=int)
    solve_parser.add_argument('--top-left-y', type=int)
    solve_parser.add_argument('--grid-size', type=int)
    solve_parser.set_defaults(func=solve_command)

    return parser



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        run_gui()
        return 0

    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
