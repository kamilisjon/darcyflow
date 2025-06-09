def gidx(i: int, j: int, Nx) -> int:
    if not all(isinstance(x, int) for x in (i, j, Nx)): raise TypeError("Indices and Nx must be integers.")
    if i < 0 or j < 0: raise ValueError("Indices must be non-negative.")
    return i + j * Nx