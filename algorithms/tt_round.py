# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if max_rank is not None:
        if not isinstance(max_rank, int) or max_rank <= 0:
            raise ValueError("max_rank must be positive integer or None")

    if eps < 0:
        raise ValueError("eps must be non-negative")

    if tt.order == 1:
        return tt.copy()

    rounded = right_canonicalize(tt, backend)
    cores = [core.copy() for core in rounded.cores]

    first_norm = backend.norm(cores[0])

    if first_norm > 1e-30:
        delta = eps * first_norm / math.sqrt(tt.order - 1)
    else:
        delta = 0.0

    for k in range(tt.order - 1):
        core = cores[k]
        r_left, n, r_right = core.shape

        matrix = backend.reshape(core, (r_left * n, r_right))

        U, S, Vt = backend.svd(matrix, full_matrices=False)

        rank = _compute_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        cores[k] = backend.reshape(U_trunc, (r_left, n, rank))

        rest = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)

        next_core = cores[k + 1]
        _, next_n, next_r = next_core.shape

        new_next_core = backend.zeros((rank, next_n, next_r))

        for a in range(rank):
            for i in range(next_n):
                for b in range(next_r):
                    value = 0.0

                    for c in range(r_right):
                        value += rest[(a, c)] * next_core[(c, i, b)]

                    new_next_core[(a, i, b)] = value

        cores[k + 1] = new_next_core

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    if S.ndim != 1:
        raise ValueError("S must be vector")

    count = S.shape[0]

    if count == 0:
        return 1

    border = max(1e-12, 1e-8 * abs(S[0]))

    numerical_rank = 0
    for i in range(count):
        if abs(S[i]) > border:
            numerical_rank = i + 1

    if numerical_rank == 0:
        numerical_rank = 1

    rank = numerical_rank

    if delta > 0:
        tail_sum = 0.0
        rank = numerical_rank

        while rank > 1:
            tail_sum += S[rank - 1] * S[rank - 1]

            if tail_sum <= delta * delta:
                rank -= 1
            else:
                break

    if max_rank is not None:
        rank = min(rank, max_rank)

    return max(1, rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    rows, cols = matrix.shape

    if rank < 1 or rank > cols:
        raise ValueError("wrong rank")

    result = backend.zeros((rows, rank))

    for i in range(rows):
        for j in range(rank):
            result[(i, j)] = matrix[(i, j)]

    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    rows, cols = matrix.shape

    if rank < 1 or rank > rows:
        raise ValueError("wrong rank")

    result = backend.zeros((rank, cols))

    for i in range(rank):
        for j in range(cols):
            result[(i, j)] = matrix[(i, j)]

    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if vector.ndim != 1:
        raise ValueError("vector must be 1-dimensional")

    if rank < 1 or rank > vector.shape[0]:
        raise ValueError("wrong rank")

    result = backend.zeros((rank,))

    for i in range(rank):
        result[(i,)] = vector[(i,)]

    return result


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    if diag_vec.ndim != 1:
        raise ValueError("diag_vec must be vector")

    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    if diag_vec.shape[0] != rank or matrix.shape[0] != rank:
        raise ValueError("wrong shapes")

    cols = matrix.shape[1]
    result = backend.zeros((rank, cols))

    for i in range(rank):
        for j in range(cols):
            result[(i, j)] = diag_vec[(i,)] * matrix[(i, j)]

    return result