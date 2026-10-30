import numpy as np
from alc6 import diagRH

def transiciones_al_azar_continuas(n):
    """
    Genera una matriz de transicion aleatoria de n x n.

    Las entradas son numeros aleatorios entre 0 y 1
    y cada columna suma 1.
    """

    if n <= 0:
        return None

    T = np.random.random((n, n))

    # Normalizamos columna por columna
    for j in range(n):

        suma = 0

        for i in range(n):
            suma = suma + T[i, j]

        for i in range(n):
            T[i, j] = T[i, j] / suma

    return T


def transiciones_al_azar_uniformes(n, thres):
    """
    Genera una matriz de transicion uniforme de n x n.

    Una entrada es distinta de cero si el numero aleatorio
    generado es menor o igual a thres.

    Todas las entradas no nulas de una misma columna
    tienen el mismo valor.
    """

    if n <= 0:
        return None

    T = np.zeros((n, n))

    aleatorios = np.random.random((n, n))

    for j in range(n):

        cantidad = 0

        # Primero decidimos cuales entradas quedan activas
        for i in range(n):

            if aleatorios[i, j] <= thres:
                T[i, j] = 1
                cantidad = cantidad + 1

        # Caso borde:
        # no puede quedar una columna toda de ceros,
        # porque una matriz de Markov debe tener suma 1
        if cantidad == 0:

            # Elegimos una fila aleatoria
            fila = np.random.randint(0, n)

            T[fila, j] = 1
            cantidad = 1

        # Ahora todas las entradas activas tienen
        # probabilidad uniforme
        for i in range(n):

            if T[i, j] != 0:
                T[i, j] = 1 / cantidad

    return T


def nucleo(A, tol=1e-15):
    """
    Calcula una base del nucleo de A.

    Para ello calcula:

        B = A^T A

    y diagonaliza B con diagRH.

    Los autovectores cuyos autovalores tienen
    modulo menor o igual a tol forman el nucleo.

    Retorna una matriz n x k, donde k es la
    dimension del nucleo.

    Para compatibilidad con los tests, si el nucleo
    es trivial retorna np.array([]).
    """

    if A is None:
        return np.array([])

    if np.ndim(A) != 2:
        return np.array([])

    m, n = np.shape(A)

    # --------------------------------------------
    # Construimos AtA sin usar @ ni np.matmul
    # --------------------------------------------

    AtA = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            suma = 0

            for k in range(m):
                suma = suma + A[k, i] * A[k, j]

            AtA[i, j] = suma

    # --------------------------------------------
    # Diagonalizamos AtA
    #
    # AtA es simetrica
    # --------------------------------------------

    S, D = diagRH(
        AtA,
        tol=tol,
        K=100000
    )

    # --------------------------------------------
    # Contamos cuantos autovalores son cero
    # --------------------------------------------

    cantidad = 0

    for i in range(n):

        if abs(D[i, i]) <= tol:
            cantidad = cantidad + 1

    # Caso nucleo trivial.
    #
    # OJO:
    # el test espera S.shape[0] == 0.
    if cantidad == 0:
        return np.array([])

    # --------------------------------------------
    # Guardamos los autovectores correspondientes
    # --------------------------------------------

    N = np.zeros((n, cantidad))

    columna = 0

    for i in range(n):

        if abs(D[i, i]) <= tol:

            for j in range(n):
                N[j, columna] = S[j, i]

            columna = columna + 1

    return N


def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Crea una matriz rala.

    listado tiene la forma:

        [indices_i, indices_j, valores]

    Retorna:

        diccionario, (m_filas, n_columnas)

    donde el diccionario tiene claves:

        (i,j)

    y valores:

        A_ij
    """

    A = {}

    # Caso lista vacia
    if listado is None or len(listado) == 0:
        return A, (m_filas, n_columnas)

    filas = listado[0]
    columnas = listado[1]
    valores = listado[2]

    cantidad = len(valores)

    for k in range(cantidad):

        valor = valores[k]

        # Ignoramos elementos aproximadamente nulos
        if abs(valor) >= tol:

            i = filas[k]
            j = columnas[k]

            A[(i, j)] = valor

    return A, (m_filas, n_columnas)


def multiplica_rala_vector(A, v):
    """
    Multiplica una matriz rala A por un vector v.

    A debe ser el resultado de crea_rala:

        (diccionario, dimensiones)

    Retorna w = A v.
    """

    if A is None:
        return None

    A_dict = A[0]
    dimensiones = A[1]

    m = dimensiones[0]
    n = dimensiones[1]

    if len(v) != n:
        return None

    # Si v es np.array, queremos devolver np.array.
    if isinstance(v, np.ndarray):

        w = np.zeros(m)

        for posicion in A_dict:

            i = posicion[0]
            j = posicion[1]

            w[i] = w[i] + A_dict[posicion] * v[j]

        return w

    # Si v es una lista, devolvemos una lista.
    else:

        w = [0] * m

        for posicion in A_dict:

            i = posicion[0]
            j = posicion[1]

            w[i] = w[i] + A_dict[posicion] * v[j]

        return w