# Test L06-metpot2k, Aval

import numpy as np
from alc import metpot2k, diagRH


def normaExacta(A, p=[1, 'inf']):
    A = np.asarray(A)

    if isinstance(p, list):
        return [normaExacta(A, norma_p) for norma_p in p]

    if p == 1:
        n = A.shape[1]
        m = A.shape[0]

    elif p == 'inf':
        n = A.shape[0]
        m = A.shape[1]

    else:
        return None

    mayor = 0

    for i in range(n):

        suma = 0

        for j in range(m):

            if p == 1:
                suma += np.abs(A[j, i])

            elif p == 'inf':
                suma += np.abs(A[i, j])

        if suma > mayor:
            mayor = suma

    return mayor


print("========================================")
print("         TESTS LABORATORIO 6")
print("========================================\n")


# ============================================================
# TEST 1 - metpot2k
# ============================================================

print("TEST 1 - metpot2k con autovalores positivos")

S = np.vstack([
    np.array([2, 1, 0]) / np.sqrt(5),
    np.array([-1, 2, 5]) / np.sqrt(30),
    np.array([1, -2, 1]) / np.sqrt(6)
]).T


# Pedimos que pase el 95% de los casos

exitos = 0

for i in range(100):

    D = np.diag(np.random.random(3) + 1) * 100

    A = S @ D @ S.T

    v, l, k = metpot2k(
        A,
        1e-15,
        1e5
    )

    if np.abs(l - np.max(D)) < 1e-8:
        exitos += 1


print("Casos exitosos:", exitos, "/ 100")

assert exitos > 95

print("----- EXITO TEST 1 -----\n")


# ============================================================
# TEST 2 - metpot2k con Householder
# ============================================================

print("TEST 2 - metpot2k con matrices construidas con Householder")

exitos = 0

for i in range(100):

    v = np.random.rand(9)

    ixv = np.argsort(-np.abs(v))

    D = np.diag(v[ixv])

    I = np.eye(9)

    H = (
        I
        - 2 * np.outer(v.T, v)
        / (np.linalg.norm(v) ** 2)
    )

    A = H @ D @ H.T

    v, l, k = metpot2k(
        A,
        1e-15,
        1e5
    )

    if abs(l - D[0, 0]) < 1e-8:
        exitos += 1


print("Casos exitosos:", exitos, "/ 100")

assert exitos > 95

print("----- EXITO TEST 2 -----\n")


# ============================================================
# TEST 3 - diagRH caso conocido
# ============================================================

print("TEST 3 - diagRH con matriz de autovalores conocidos")

D = np.diag([
    1,
    0.5,
    0.25
])

S = np.vstack([
    np.array([1, -1, 1]) / np.sqrt(3),
    np.array([1, 1, 0]) / np.sqrt(2),
    np.array([1, -1, -2]) / np.sqrt(6)
]).T


A = S @ D @ S.T


SRH, DRH = diagRH(
    A,
    tol=1e-15,
    K=1e5
)


print("D esperada:")
print(D)

print("\nD obtenida:")
print(DRH)

print()


assert np.allclose(
    D,
    DRH
)

assert np.allclose(
    np.abs(S.T @ SRH),
    np.eye(A.shape[0]),
    atol=1e-7
)

print("----- EXITO TEST 3 -----\n")


# ============================================================
# TEST 4 - diagRH matrices aleatorias
# ============================================================

print("TEST 4 - diagRH con matrices simetricas aleatorias")

# Pedimos que pase el 95% de los casos

exitos = 0

mayor_error = 0

for i in range(100):

    A = np.random.random((5, 5))

    A = 0.5 * (A + A.T)

    S, D = diagRH(
        A,
        tol=1e-15,
        K=1e5
    )

    ARH = S @ D @ S.T

    e = normaExacta(
        ARH - A,
        p='inf'
    )

    if e > mayor_error:
        mayor_error = e

    if e < 1e-5:
        exitos += 1


print("Casos exitosos:", exitos, "/ 100")
print("Mayor error encontrado:", mayor_error)

assert exitos >= 95

print("----- EXITO TEST 4 -----\n")


# ============================================================
# FINAL
# ============================================================

print("========================================")
print("      FINALIZADO LABORATORIO 6")
print("        TODOS LOS TESTS PASARON")
print("========================================")