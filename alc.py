import numpy as np

def error(x, y):
    x_float64 = np.asarray(x, dtype=np.float64)
    y_float64 = np.asarray(y, dtype=np.float64)

    return np.abs(x_float64 - y_float64)

def error_relativo(x, y):
    """
    Error relativo de aproximar x usando y.
    """
    x_float64 = np.asarray(x, dtype=np.float64)
    y_float64 = np.asarray(y, dtype=np.float64)

    if x_float64 == 0:
        return np.inf if y_float64 != 0 else 0.0

    resultado =  np.abs(x_float64 - y_float64) / np.abs(x_float64)

    if resultado.ndim == 0:
        return resultado.item()

    return resultado

def matricesIguales(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    
    if A.shape != B.shape:
        return False

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            a = A[i, j]
            b = B[i, j]
            if not np.isclose(a, b, rtol=1e-5, atol=1e-8):
                return False
    return True

def rota(theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return R

def escala(s):
    n = len(s)
    S = np.zeros((n, n))

    for i in range(n):
        S[i, i] = s[i]

    return S

def rota_y_escala(theta, s):
    return escala(s) @ rota(theta)

def afin(theta, s, b):

    rot_y_esc = rota_y_escala(theta, s)

    A = np.zeros((3, 3))

    A[0:2, 0:2] = rot_y_esc
    A[0:2, 2] = b
    A[2, 2] = 1

    return A

def trans_afin(v, theta, s, b):
    A = afin(theta, s, b)

    v_hom = np.array([v[0], v[1], 1])

    v_trans = A @ v_hom

    return v_trans[0:2]