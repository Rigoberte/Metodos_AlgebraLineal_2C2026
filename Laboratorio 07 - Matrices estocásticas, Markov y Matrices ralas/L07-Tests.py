import numpy as np
import alc


def transiciones_al_azar_continuas(n):
    return alc.transiciones_al_azar_continuas(n)


def transiciones_al_azar_uniformes(n, thres):
    return alc.transiciones_al_azar_uniformes(n, thres)


def nucleo(A, tol=1e-15):
    return alc.nucleo(A, tol)


def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    return alc.crea_rala(listado, m_filas, n_columnas, tol)


def multiplica_rala_vector(A, v):
    return alc.multiplica_rala_vector(A, v)


def es_markov(T, tol=1e-6):
    n = T.shape[0]

    for i in range(n):
        for j in range(n):
            if T[i, j] < 0:
                return False

    for j in range(n):
        suma_columna = sum(T[:, j])

        if np.abs(suma_columna - 1) > tol:
            return False

    return True


def es_markov_uniforme(T, thres=1e-6):
    if not es_markov(T, thres):
        return False

    m = T.shape[1]

    for j in range(m):

        non_zero = T[:, j][T[:, j] > thres]

        close = all(
            np.abs(non_zero - non_zero[0]) < thres
        )

        if not close:
            return False

    return True


def esNucleo(A, S, tol=1e-5):
    for col in S.T:

        res = A @ col

        if not np.allclose(
            res,
            np.zeros(A.shape[0]),
            atol=tol
        ):
            return False

    return True


print("========================================")
print("         TESTS LABORATORIO 7")
print("========================================\n")


# ============================================================
# TEST 1 - transiciones_al_azar_continuas
# ============================================================

print("TEST 1 - transiciones_al_azar_continuas")

for i in range(1, 100):

    T = transiciones_al_azar_continuas(i)

    assert es_markov(T), (
        f"transiciones_al_azar_continuas fallo para n={i}"
    )

print("Se probaron matrices de n=1 hasta n=99")
print("----- EXITO TEST 1 -----\n")


# ============================================================
# TEST 2 - transiciones_al_azar_uniformes
# ============================================================

print("TEST 2 - transiciones_al_azar_uniformes")

for i in range(1, 100):

    T = transiciones_al_azar_uniformes(i, 0.3)

    assert es_markov_uniforme(T), (
        f"transiciones_al_azar_uniformes fallo para "
        f"n={i}, thres=0.3"
    )

print("Caso thres=0.3 superado para n=1,...,99")


for i in range(1, 100):

    T = transiciones_al_azar_uniformes(i, 0.01)

    assert es_markov_uniforme(T), (
        f"transiciones_al_azar_uniformes fallo para "
        f"n={i}, thres=0.01"
    )

print("Caso borde thres=0.01 superado para n=1,...,99")


# repetimos para detectar columnas nulas por azar

for i in range(1, 100):

    T = transiciones_al_azar_uniformes(i, 0.01)

    assert es_markov_uniforme(T), (
        f"transiciones_al_azar_uniformes fallo en "
        f"segunda prueba para n={i}, thres=0.01"
    )

print("Segunda prueba con thres=0.01 superada")
print("----- EXITO TEST 2 -----\n")


# ============================================================
# TEST 3 - nucleo de la identidad
# ============================================================

print("TEST 3 - nucleo de la matriz identidad")

A = np.eye(3)

S = nucleo(A)

print("Forma obtenida para el nucleo:", S.shape)

assert S.shape[0] == 0, (
    "nucleo fallo para matriz identidad"
)

print("El nucleo de I es trivial")
print("----- EXITO TEST 3 -----\n")


# ============================================================
# TEST 4 - nucleo de matriz diagonal
# ============================================================

print("TEST 4 - nucleo de matriz diagonal con un cero")

A = np.eye(3)
A[1, 1] = 0

S = nucleo(A)

print("Matriz A:")
print(A)

print("\nBase del nucleo obtenida:")
print(S)

msg = "nucleo fallo para matriz con un cero en diagonal"

assert esNucleo(A, S), msg
assert S.shape == (3, 1), msg
assert abs(S[2, 0]) < 1e-2, msg
assert abs(S[0, 0]) < 1e-2, msg

print("Forma del nucleo:", S.shape)
print("----- EXITO TEST 4 -----\n")


# ============================================================
# TEST 5 - nucleo de proyeccion ortogonal
# ============================================================

print("TEST 5 - nucleo de matriz de proyeccion ortogonal")

v = np.random.random(5)

v = v / np.linalg.norm(v)

H = np.eye(5) - np.outer(v, v)

S = nucleo(H)

msg = "nucleo fallo para matriz de proyeccion ortogonal"

assert S.shape == (5, 1), msg

v_gen = S[:, 0]
v_gen = v_gen / np.linalg.norm(v_gen)

print("Vector original:")
print(v)

print("\nVector generado por nucleo:")
print(v_gen)

assert (
    np.allclose(v, v_gen)
    or np.allclose(v, -v_gen)
), msg

print("El vector del nucleo coincide salvo signo")
print("----- EXITO TEST 5 -----\n")


# ============================================================
# TEST 6 - crea_rala basico
# ============================================================

print("TEST 6 - crea_rala caso basico")

listado = [
    [0, 17],
    [3, 4],
    [0.5, 0.25]
]

A_rala_dict, dims = crea_rala(
    listado,
    32,
    89
)

print("Dimensiones:", dims)
print("Diccionario:", A_rala_dict)

assert dims == (32, 89), (
    "crea_rala fallo en dimensiones"
)

assert A_rala_dict[(0, 3)] == 0.5, (
    "crea_rala fallo"
)

assert A_rala_dict[(17, 4)] == 0.25, (
    "crea_rala fallo"
)

assert len(A_rala_dict) == 2, (
    "crea_rala fallo en cantidad de elementos"
)

print("----- EXITO TEST 6 -----\n")


# ============================================================
# TEST 7 - crea_rala segundo caso
# ============================================================

print("TEST 7 - crea_rala con mas elementos")

listado = [
    [32, 16, 5],
    [3, 4, 7],
    [7, 0.5, 0.25]
]

A_rala_dict, dims = crea_rala(
    listado,
    50,
    50
)

print("Dimensiones:", dims)
print("Diccionario:", A_rala_dict)

assert dims == (50, 50)

assert A_rala_dict.get((32, 3)) == 7
assert A_rala_dict[(16, 4)] == 0.5
assert A_rala_dict[(5, 7)] == 0.25

print("----- EXITO TEST 7 -----\n")


# ============================================================
# TEST 8 - crea_rala con tolerancia
# ============================================================

print("TEST 8 - crea_rala filtrando valores menores a tol")

listado = [
    [1, 2, 3],
    [4, 5, 6],
    [1e-20, 0.5, 0.25]
]

A_rala_dict, dims = crea_rala(
    listado,
    10,
    10
)

print("Dimensiones:", dims)
print("Diccionario filtrado:", A_rala_dict)

assert dims == (10, 10)

assert (1, 4) not in A_rala_dict

assert A_rala_dict[(2, 5)] == 0.5
assert A_rala_dict[(3, 6)] == 0.25

assert len(A_rala_dict) == 2

print("El valor 1e-20 fue descartado correctamente")
print("----- EXITO TEST 8 -----\n")


# ============================================================
# TEST 9 - crea_rala lista vacia
# ============================================================

print("TEST 9 - crea_rala con lista vacia")

listado = []

A_rala_dict, dims = crea_rala(
    listado,
    10,
    10
)

print("Dimensiones:", dims)
print("Diccionario:", A_rala_dict)

assert dims == (10, 10), (
    "crea_rala fallo en dimensiones con lista vacia"
)

assert len(A_rala_dict) == 0, (
    "crea_rala fallo en cantidad de elementos con lista vacia"
)

print("----- EXITO TEST 9 -----\n")


# ============================================================
# TEST 10 - multiplica_rala_vector diagonal
# ============================================================

print("TEST 10 - multiplica_rala_vector con matriz diagonal")

listado = [
    [0, 1, 2],
    [0, 1, 2],
    [1, 2, 3]
]

A_rala = crea_rala(
    listado,
    3,
    3
)

v = np.random.random(3)

v = v / np.linalg.norm(v)

res = multiplica_rala_vector(
    A_rala,
    v
)

A = np.array([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]
])

res_esperado = A @ v

print("Vector v:")
print(v)

print("\nResultado ralo:")
print(res)

print("\nResultado esperado:")
print(res_esperado)

assert np.allclose(
    res,
    res_esperado
), "multiplica_rala_vector fallo"

print("----- EXITO TEST 10 -----\n")


# ============================================================
# TEST 11 - multiplica_rala_vector aleatorio
# ============================================================

print("TEST 11 - multiplica_rala_vector con matriz aleatoria")

A = np.random.random((5, 5))

A = A * (A > 0.5)

listado = [
    [],
    [],
    []
]

for i in range(5):
    for j in range(5):

        listado[0].append(i)
        listado[1].append(j)
        listado[2].append(A[i, j])


A_rala = crea_rala(
    listado,
    5,
    5
)

v = np.random.random(5)

res_ralo = multiplica_rala_vector(
    A_rala,
    v
)

res_normal = A @ v

print("Matriz original:")
print(A)

print("\nVector:")
print(v)

print("\nResultado ralo:")
print(res_ralo)

print("\nResultado normal:")
print(res_normal)

assert np.allclose(
    res_ralo,
    res_normal
), "multiplica_rala_vector fallo para matriz aleatoria"

print("----- EXITO TEST 11 -----\n")


# ============================================================
# FINAL
# ============================================================

print("========================================")
print("      FINALIZADO LABORATORIO 7")
print("        TODOS LOS TESTS PASARON")
print("========================================")