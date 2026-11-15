"""Общий код для ноутбуков: загрузка данных, разбиение на train/test,
построение разреженных матриц и получение top-K рекомендаций."""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

SEED = 42
TEST_FRAC = 0.2
RELEVANCE_THRESHOLD = 4  # оценка >= 4 считается релевантной
TOP_K = 10

# В датасете user_id идут подряд от 1 до 53424, а book_id от 1 до 10000,
# поэтому индекс строки/столбца в матрицах равен id - 1.
N_USERS = 53424
N_BOOKS = 10000


def load_ratings() -> pd.DataFrame:
    """Оценки без повторов пары (user_id, book_id)."""
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    ratings = ratings.drop_duplicates(["user_id", "book_id"], keep="last")
    assert ratings["user_id"].max() == N_USERS and ratings["book_id"].max() == N_BOOKS
    return ratings.reset_index(drop=True)


def load_books() -> pd.DataFrame:
    """Таблица книг. В исходном файле столбец `id` соответствует book_id
    из ratings.csv, а в `book_id` лежит идентификатор Goodreads. Переименовываем,
    чтобы не путаться."""
    books = pd.read_csv(DATA_DIR / "books.csv")
    books = books.rename(columns={"id": "book_id", "book_id": "goodreads_book_id"})
    return books.sort_values("book_id").reset_index(drop=True)


def split_ratings(ratings: pd.DataFrame, test_frac: float = TEST_FRAC, seed: int = SEED):
    """Случайное разбиение внутри каждого пользователя: test_frac его оценок
    уходит в тест. У пользователей с 1-2 оценками в тест ничего не попадает."""
    test = ratings.groupby("user_id").sample(frac=test_frac, random_state=seed)
    train = ratings.drop(test.index)
    return train, test


def to_sparse(df: pd.DataFrame, values=None) -> sp.csr_matrix:
    """Разреженная матрица пользователь-книга. По умолчанию в значениях стоят оценки."""
    if values is None:
        values = df["rating"].to_numpy()
    rows = df["user_id"].to_numpy() - 1
    cols = df["book_id"].to_numpy() - 1
    return sp.csr_matrix((values.astype(np.float32), (rows, cols)), shape=(N_USERS, N_BOOKS))


def get_eval_users(test: pd.DataFrame) -> np.ndarray:
    """Пользователи, у которых в тесте есть хотя бы одна релевантная книга."""
    relevant = test[test["rating"] >= RELEVANCE_THRESHOLD]
    return np.sort(relevant["user_id"].unique())


def keep_top_k(sim: sp.spmatrix, k: int) -> sp.csr_matrix:
    """Оставляет в каждой строке разреженной матрицы k наибольших значений."""
    sim = sim.tocoo()
    order = np.lexsort((-sim.data, sim.row))
    row, col, data = sim.row[order], sim.col[order], sim.data[order]
    rank = np.arange(len(row)) - np.searchsorted(row, row, side="left")
    mask = rank < k
    return sp.csr_matrix((data[mask], (row[mask], col[mask])), shape=sim.shape)


def recommend_top_k(score_fn, seen: sp.csr_matrix, user_ids, k: int = TOP_K, batch_size: int = 2000):
    """Top-k книг для набора пользователей.

    score_fn(user_idx) должна вернуть плотную матрицу оценок релевантности
    размера (len(user_idx), N_BOOKS). Книги, которые пользователь уже оценил
    (ненулевые элементы seen), из выдачи исключаются. Считаем батчами, чтобы
    не держать в памяти матрицу на всех пользователей сразу.

    Возвращает массив book_id размера (len(user_ids), k), отсортированный по
    убыванию оценки релевантности.
    """
    user_idx = np.asarray(user_ids) - 1
    recs = np.empty((len(user_idx), k), dtype=np.int32)
    for start in range(0, len(user_idx), batch_size):
        batch = user_idx[start:start + batch_size]
        scores = np.array(score_fn(batch), dtype=np.float32)
        scores[seen[batch].nonzero()] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
        recs[start:start + batch_size] = np.take_along_axis(top, order, axis=1)
    return recs + 1


def save_recs(name: str, user_ids, recs) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    np.savez(ARTIFACTS_DIR / f"recs_{name}.npz", user_ids=user_ids, recs=recs)


def load_recs(name: str):
    data = np.load(ARTIFACTS_DIR / f"recs_{name}.npz")
    return data["user_ids"], data["recs"]
