"""Скачивает датасет goodbooks-10k с Kaggle и кладёт csv-файлы в папку data/.

Запуск: uv run python download_data.py
"""

import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).resolve().parent / "data"
FILES = ["ratings.csv", "books.csv", "tags.csv", "book_tags.csv"]


def main() -> None:
    src = Path(kagglehub.dataset_download("zygmunt/goodbooks-10k"))
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        shutil.copy(src / name, DATA_DIR / name)
        print(f"{name}: {(DATA_DIR / name).stat().st_size / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
