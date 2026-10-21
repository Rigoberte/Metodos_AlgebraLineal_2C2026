import numpy as np

def error(x, y):
    x_float64 = np.asarray(x, dtype=np.float64)
    y_float64 = np.asarray(y, dtype=np.float64)

    return np.abs(x_float64 - y_float64)

def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x,y), 0, atol=atol)

def norma2(v):
    """
    Calcula la norma 2 de un vector.
    """
    n = np.shape(v)[0]

    suma = 0

    for i in range(n):
        suma = suma + v[i] * v[i]

    return suma ** 0.5

def productoInterno(v, w):
    """
    Calcula el producto interno entre dos vectores.
    """
    n = np.shape(v)[0]

    suma = 0

    for i in range(n):
        suma = suma + v[i] * w[i]

    return suma

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    Calcula la factorizacion QR de A usando Gram-Schmidt.

    Si A no es cuadrada, retorna None.

    Si hay columnas linealmente dependientes, Q contiene
    solamente las columnas independientes.

    Si retorna_nops=True:
        retorna Q, R, nops

    Si retorna_nops=False:
        retorna Q, R
    """

    if A is None:
        return None

    if np.ndim(A) != 2:
        return None

    n, m = np.shape(A)

    if n != m:
        return None

    Qaux = np.zeros((n, n))
    Raux = np.zeros((n, n))

    r = 0

    nops = 0

    for j in range(n):

        v = np.copy(A[:, j])

        for k in range(r):

            rij = 0

            for i in range(n):
                rij = rij + Qaux[i, k] * v[i]

                nops = nops + 2

            Raux[k, j] = rij

            for i in range(n):
                v[i] = v[i] - rij * Qaux[i, k]

                # producto + resta
                nops = nops + 2

        suma = 0

        for i in range(n):
            suma = suma + v[i] * v[i]

            nops = nops + 2

        norma = suma ** 0.5

        if not sonIguales(norma, 0, atol=tol):

            Raux[r, j] = norma

            for i in range(n):
                Qaux[i, r] = v[i] / norma

                # division
                nops = nops + 1

            r = r + 1

    Q = np.copy(Qaux[:, :r])
    R = np.copy(Raux[:r, :])

    for i in range(r):
        for j in range(n):
            if sonIguales(R[i, j], 0, atol=tol):
                R[i, j] = 0

    if retorna_nops:
        return Q, R, nops

    return Q, R


def QR_con_HH(A, tol=1e-12, extras=False):
    """
    Calcula la factorizacion QR usando reflexiones
    de Householder.

    A debe ser m x n con m >= n.

    Si extras=False:
        retorna Q, R

    Si extras=True:
        retorna Q, R, extra_info
    """

    if A is None:
        return None

    if np.ndim(A) != 2:
        return None

    m, n = np.shape(A)

    if m < n:
        return None

    R = np.array(A, dtype=float)
    Q = np.eye(m)

    R_matrices = []
    Q_matrices = []

    for k in range(n):

        # x = R[k:m, k]
        x = np.copy(R[k:m, k])

        norma_x = norma2(x)

        if not sonIguales(norma_x, 0, atol=tol):

            if x[0] < 0:
                signo = -1
            else:
                signo = 1

            alpha = -signo * norma_x

            u = np.copy(x)

            u[0] = u[0] - alpha

            norma_u = norma2(u)

            if not sonIguales(norma_u, 0, atol=tol):

                largo = np.shape(u)[0]

                # Normalizamos u
                for i in range(largo):
                    u[i] = u[i] / norma_u

                for j in range(k, n):

                    prod = 0

                    for i in range(largo):
                        prod = prod + u[i] * R[k + i, j]

                    for i in range(largo):
                        R[k + i, j] = (
                            R[k + i, j]
                            - 2 * u[i] * prod
                        )

                for i in range(m):

                    prod = 0

                    for j in range(largo):
                        prod = (
                            prod
                            + Q[i, k + j] * u[j]
                        )

                    for j in range(largo):
                        Q[i, k + j] = (
                            Q[i, k + j]
                            - 2 * prod * u[j]
                        )

        for i in range(k + 1, m):

            if sonIguales(R[i, k], 0, atol=tol):
                R[i, k] = 0

        R_matrices.append(np.copy(R))
        Q_matrices.append(np.copy(Q))

    for i in range(m):
        for j in range(n):

            if sonIguales(R[i, j], 0, atol=tol):
                R[i, j] = 0

    if extras:

        extra_info = {
            "R_matrices": R_matrices,
            "Q_matrices": Q_matrices
        }

        return Q, R, extra_info

    return Q, R

def calculaQR(A, metodo='RH', tol=1e-12):
    """
    Calcula QR usando el metodo indicado.

    metodo='RH' -> Householder
    metodo='GS' -> Gram-Schmidt

    Si el metodo no es valido retorna None.
    """

    if metodo == 'GS':
        return QR_con_GS(A, tol=tol)

    if metodo == 'RH':
        return QR_con_HH(A, tol=tol)

    return None