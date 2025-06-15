from statistics import StatisticsError


def gidx(i: int, j: int, Nx) -> int:
    if not all(isinstance(x, int) for x in (i, j, Nx)): raise TypeError("Indices and Nx must be integers.")
    if i < 0 or j < 0: raise ValueError("Indices must be non-negative.")
    return i + j * Nx

def harmonic_mean_2point(k1: int|float, k2: int|float) -> float:
    if k1<0.0 or k2<0.0: raise StatisticsError("harmonic mean does not support negative values")
    return 0.0 if k1 + k2 == 0.0 else 2.0 * k1 * k2 / (k1 + k2)