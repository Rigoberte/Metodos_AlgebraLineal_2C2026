import numpy as np

def error(x, y):
    x_float64 = np.asarray(x, dtype=np.float64)
    y_float64 = np.asarray(y, dtype=np.float64)

    return np.abs(x_float64 - y_float64)

def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x,y), 0, atol=atol)

def calculaLU(A):
    if A is None:
        return None, None, 0

    if np.ndim(A) != 2:
        return None, None, 0

    m, n = np.shape(A)

    if m != n:
        return None, None, 0

    Ac = np.array(A, dtype=float)

    cant_op = 0

    for k in range(n - 1):

        if sonIguales(Ac[k, k], 0):
            return None, None, 0

        for i in range(k + 1, n):

            m_ik = Ac[i, k] / Ac[k, k]
            cant_op += 1

            Ac[i, k] = m_ik

            for j in range(k + 1, n):
                Ac[i, j] = Ac[i, j] - m_ik * Ac[k, j]

                cant_op += 2

    if n > 0 and sonIguales(Ac[n - 1, n - 1], 0):
        return None, None, 0

    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            if i > j:
                L[i, j] = Ac[i, j]
            else:
                U[i, j] = Ac[i, j]

    return L, U, cant_op

def res_tri(L, b, inferior=True):
    if L is None or b is None:
        return None

    if np.ndim(L) != 2 or np.ndim(b) != 1:
        return None

    n, m = np.shape(L)

    if n != m or np.shape(b)[0] != n:
        return None

    x = np.zeros(n)

    if inferior:

        # Sustitucion hacia adelante
        for i in range(n):

            if sonIguales(L[i, i], 0):
                return None

            suma = 0.0

            for j in range(i):
                suma = suma + L[i, j] * x[j]

            x[i] = (b[i] - suma) / L[i, i]

    else:

        # Sustitucion hacia atras
        for i in range(n - 1, -1, -1):

            if sonIguales(L[i, i], 0):
                return None

            suma = 0.0

            for j in range(i + 1, n):
                suma = suma + L[i, j] * x[j]

            x[i] = (b[i] - suma) / L[i, i]

    return x

def inversa(A):
    L, U, _ = calculaLU(A)

    if L is None:
        return None

    n = np.shape(A)[0]

    Ainv = np.zeros((n, n))

    for j in range(n):

        e = np.zeros(n)
        e[j] = 1.0

        y = res_tri(L, e, inferior=True)

        if y is None:
            return None

        x = res_tri(U, y, inferior=False)

        if x is None:
            return None

        Ainv[:, j] = x

    return Ainv


def calculaLDV(A):
    L, U, _ = calculaLU(A)

    if L is None:
        return None, None, None

    n = np.shape(A)[0]

    D = np.zeros((n, n))
    V = np.zeros((n, n))

    for i in range(n):

        if sonIguales(U[i, i], 0):
            return None, None, None

        D[i, i] = U[i, i]

        for j in range(i, n):
            V[i, j] = U[i, j] / U[i, i]

    return L, D, V


def esSDP(A, atol=1e-10):
    if A is None:
        return False

    if np.ndim(A) != 2:
        return False

    n, m = np.shape(A)

    if n != m:
        return False

    for i in range(n):
        for j in range(i + 1, n):

            if not sonIguales(A[i, j], A[j, i], atol=atol):
                return False

    _, D, _ = calculaLDV(A)

    if D is None:
        return False

    for i in range(n):

        if D[i, i] <= atol:
            return False

    return True