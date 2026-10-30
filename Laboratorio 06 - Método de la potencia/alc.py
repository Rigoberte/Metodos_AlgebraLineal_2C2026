import numpy as np

def error(x, y):
    x_float64 = np.asarray(x, dtype=np.float64)
    y_float64 = np.asarray(y, dtype=np.float64)

    return np.abs(x_float64 - y_float64)


def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x,y), 0, atol=atol)

def norma2(v):
    n = np.shape(v)[0]

    suma = 0

    for i in range(n):
        suma = suma + v[i] * v[i]

    return suma ** 0.5


def productoInterno(v, w):
    n = np.shape(v)[0]

    suma = 0

    for i in range(n):
        suma = suma + v[i] * w[i]

    return suma

def productoMatrizVector(A, v):
    """
    Calcula A*v sin utilizar @ ni np.matmul.
    """

    m, n = np.shape(A)

    if np.shape(v)[0] != n:
        return None

    resultado = np.zeros(m)

    for i in range(m):

        suma = 0

        for j in range(n):
            suma = suma + A[i, j] * v[j]

        resultado[i] = suma

    return resultado


def productoMatrices(A, B):
    """
    Calcula A*B sin utilizar @ ni np.matmul.
    """

    m, n = np.shape(A)
    n2, p = np.shape(B)

    if n != n2:
        return None

    C = np.zeros((m, p))

    for i in range(m):
        for j in range(p):

            suma = 0

            for k in range(n):
                suma = suma + A[i, k] * B[k, j]

            C[i, j] = suma

    return C

def _fA(A, v, tol=1e-15):
    """
    Aplica:

        w' = A v
        w  = w' / ||w'||_2

    Si ||w'|| es cero, devuelve el vector cero.
    """

    w = productoMatrizVector(A, v)

    norma = norma2(w)

    if sonIguales(norma, 0, atol=tol):
        return np.zeros(np.shape(v)[0])

    for i in range(np.shape(w)[0]):
        w[i] = w[i] / norma

    return w

def metpot2k(A, tol=1e-15, K=1000):
    """
    Metodo de la potencia aplicando f_A dos veces
    en cada iteracion.

    A debe ser una matriz cuadrada.

    Retorna:
        v      autovector asociado al autovalor dominante
        lambda autovalor dominante
        k      cantidad de iteraciones realizadas
    """

    if A is None:
        return None

    if np.ndim(A) != 2:
        return None

    n, m = np.shape(A)

    if n != m:
        return None

    # Vector inicial aleatorio
    v = np.random.random(n)

    # Lo normalizamos para poder comparar productos internos.
    norma_v = norma2(v)

    if sonIguales(norma_v, 0, atol=tol):
        return None

    for i in range(n):
        v[i] = v[i] / norma_v

    # v_tilde = f_A^2(v)
    v_tilde = _fA(A, v, tol)
    v_tilde = _fA(A, v_tilde, tol)

    e = productoInterno(v_tilde, v)

    k = 0

    while (
        not sonIguales(e, 1, atol=tol)
        and k < K
    ):

        v = np.copy(v_tilde)

        v_tilde = _fA(A, v, tol)
        v_tilde = _fA(A, v_tilde, tol)

        e = productoInterno(v_tilde, v)

        k = k + 1

    # Estimacion del autovalor mediante cociente de Rayleigh:
    Av = productoMatrizVector(A, v_tilde)

    lamb = productoInterno(v_tilde, Av)

    return v_tilde, lamb, k

def _householderAutovector(v, tol=1e-15):
    """
    Construye el reflector H de forma que H*v = e1.
    """

    n = np.shape(v)[0]

    e1 = np.zeros(n)
    e1[0] = 1

    u = np.zeros(n)

    for i in range(n):
        u[i] = e1[i] - v[i]

    norma_u = norma2(u)

    if sonIguales(norma_u, 0, atol=tol):
        return np.eye(n)

    H = np.eye(n)

    denominador = norma_u * norma_u

    for i in range(n):
        for j in range(n):

            H[i, j] = (
                H[i, j]
                - 2 * u[i] * u[j] / denominador
            )

    return H

def diagRH(A, tol=1e-15, K=1000):
    """
    Diagonaliza una matriz simetrica A usando:

        - metodo de la potencia
        - reflectores de Householder
        - recursion

    Retorna S, D tales que:

        A = S D S^T

    Si A no es simetrica retorna None.
    """

    if A is None:
        return None

    if np.ndim(A) != 2:
        return None

    n, m = np.shape(A)

    if n != m:
        return None

    for i in range(n):
        for j in range(i + 1, n):

            if not sonIguales(
                A[i, j],
                A[j, i],
                atol=tol
            ):
                return None

    # Caso base n = 1
    if n == 1:

        S = np.eye(1)

        D = np.zeros((1, 1))
        D[0, 0] = A[0, 0]

        return S, D

    v1, lambda1, _ = metpot2k(
        A,
        tol=tol,
        K=K
    )

    # Reflector de v1
    H = _householderAutovector(
        v1,
        tol=tol
    )

    HA = productoMatrices(H, A)
    B = productoMatrices(HA, H)

    for i in range(n):
        for j in range(n):

            if sonIguales(B[i, j], 0, atol=tol):
                B[i, j] = 0

    # Caso n = 2
    if n == 2:

        D = np.zeros((2, 2))

        D[0, 0] = lambda1
        D[1, 1] = B[1, 1]

        S = np.copy(H)

        return S, D

    A_tilde = np.copy(
        B[1:n, 1:n]
    )

    S_tilde, D_tilde = diagRH(
        A_tilde,
        tol=tol,
        K=K
    )

    D = np.zeros((n, n))

    D[0, 0] = lambda1

    for i in range(n - 1):
        for j in range(n - 1):

            D[i + 1, j + 1] = D_tilde[i, j]

    P = np.zeros((n, n))

    P[0, 0] = 1

    for i in range(n - 1):
        for j in range(n - 1):

            P[i + 1, j + 1] = S_tilde[i, j]

    S = productoMatrices(H, P)

    return S, D