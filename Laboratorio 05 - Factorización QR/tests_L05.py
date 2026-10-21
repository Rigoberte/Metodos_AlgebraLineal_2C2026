import alc
import numpy as np


### Funciones L05-QR

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt
    (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    return alc.QR_con_GS(A, tol, retorna_nops)


def QR_con_HH(A, tol=1e-12, extras=False):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None

    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso
        de factorizacion.

    Devuelve:
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    """
    return alc.QR_con_HH(A, tol, extras)


def calculaQR(A, metodo='RH', tol=1e-12):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R

    metodo = ['RH','GS']
    usa reflectores de Householder (RH) o Gram Schmidt (GS)
    para realizar la factorizacion.

    Si el metodo no esta entre las opciones, retorna None.
    """
    return alc.calculaQR(A, metodo, tol)


# ============================================================
# TESTS L05-QR
# ============================================================


# --- Matrices de prueba ---

A2 = np.array([
    [1., 2.],
    [3., 4.]
])

A3 = np.array([
    [1., 0., 1.],
    [0., 1., 1.],
    [1., 1., 0.]
])

A4 = np.array([
    [2., 0., 1., 3.],
    [0., 1., 4., 1.],
    [1., 0., 2., 0.],
    [3., 1., 0., 2.]
])


# --- Funcion auxiliar para los tests ---

def check_QR(Q, R, A, tol=1e-10):
    """
    Comprueba:
        1) Q tiene columnas ortonormales
        2) Q @ R reconstruye A
    """

    assert np.allclose(
        Q.T @ Q,
        np.eye(Q.shape[1]),
        atol=tol
    )

    assert np.allclose(
        Q @ R,
        A,
        atol=tol
    )


# ============================================================
# TESTS QR_con_GS
# ============================================================

print("TESTS QR_con_GS")

Q2, R2 = QR_con_GS(A2)
check_QR(Q2, R2, A2)

Q3, R3 = QR_con_GS(A3)
check_QR(Q3, R3, A3)

Q4, R4 = QR_con_GS(A4)
check_QR(Q4, R4, A4)

print("----- EXITO QR_con_GS -----\n")


# ============================================================
# TEST retorna_nops
# ============================================================

print("TEST retorna_nops")

Q2, R2, nops = QR_con_GS(
    A2,
    retorna_nops=True
)

check_QR(Q2, R2, A2)

assert nops >= 0

print("----- EXITO retorna_nops -----\n")


# ============================================================
# TESTS QR_con_HH
# ============================================================

print("TESTS QR_con_HH")

Q2h, R2h = QR_con_HH(A2)
check_QR(Q2h, R2h, A2)

Q3h, R3h = QR_con_HH(A3)
check_QR(Q3h, R3h, A3)

Q4h, R4h = QR_con_HH(A4)
check_QR(Q4h, R4h, A4)

print("----- EXITO QR_con_HH -----\n")


# ============================================================
# TEST Householder rectangular
# ============================================================

print("TEST Householder rectangular")

Arect = np.array([
    [1., 1.],
    [1., 0.],
    [0., 1.]
])

Qrect, Rrect = QR_con_HH(Arect)

check_QR(Qrect, Rrect, Arect)

assert Qrect.shape == (3, 3)
assert Rrect.shape == (3, 2)

print("----- EXITO rectangular -----\n")


# ============================================================
# TEST extras=True
# ============================================================

print("TEST extras=True")

Qe, Re, extra_info = QR_con_HH(
    A3,
    extras=True
)

check_QR(Qe, Re, A3)

assert "R_matrices" in extra_info
assert "Q_matrices" in extra_info

assert len(extra_info["R_matrices"]) > 0
assert len(extra_info["Q_matrices"]) > 0

print("----- EXITO extras -----\n")


# ============================================================
# TESTS calculaQR
# ============================================================

print("TESTS calculaQR")

Q2c, R2c = calculaQR(
    A2,
    metodo='RH'
)

check_QR(Q2c, R2c, A2)


Q3c, R3c = calculaQR(
    A3,
    metodo='GS'
)

check_QR(Q3c, R3c, A3)


Q4c, R4c = calculaQR(
    A4,
    metodo='RH'
)

check_QR(Q4c, R4c, A4)

print("----- EXITO calculaQR -----\n")


# ============================================================
# TESTS DE ENTRADAS INVALIDAS
# ============================================================

print("TESTS entradas invalidas")

# Gram-Schmidt solo acepta matrices cuadradas

AnoCuadrada = np.array([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])

assert QR_con_GS(AnoCuadrada) is None


# Householder requiere m >= n

Amal = np.array([
    [1., 2., 3.],
    [4., 5., 6.]
])

assert QR_con_HH(Amal) is None


# Metodo incorrecto

assert calculaQR(
    A2,
    metodo='cualquier_cosa'
) is None

print("----- EXITO entradas invalidas -----\n")


# ============================================================
# FINAL
# ============================================================

print("====================================")
print("----- FINALIZADO LABO 5 QR! -----")
print("====================================")