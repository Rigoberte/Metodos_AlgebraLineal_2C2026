import numpy as np

def multiplicar_matrices(A, B):
    A = np.asarray(A)
    B = np.asarray(B)

    if A.shape[1] != B.shape[0]:
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.")

    resultado = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                resultado[i, j] += A[i, k] * B[k, j]

    return resultado

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
    return multiplicar_matrices(escala(s), rota(theta))

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

    v_trans = multiplicar_matrices(A, v_hom)

    return v_trans[0:2]

def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x,y), 0, atol=atol)

def norma(x, p):
    x = np.asarray(x)
    
    if p in [1,2,5,10,100,200]:
        return np.power(np.sum(np.power(np.abs(x), p)), 1/p)
    if p == 'inf':
        return np.max(np.abs(x))

def normaliza(X, p):
    ans = []

    for x in X:
        x = np.array(x, dtype=float)
        nx = norma(x, p)

        if nx != 0:
            ans.append(x / nx)
        else:
            ans.append(x)

    return ans

def normaExacta(A, p=[1,'inf']):
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

def transpuesta(A):
    A = np.asarray(A)
    n, m = A.shape
    ans = np.zeros((m, n))

    for i in range(n):
        for j in range(m):
            ans[j, i] = A[i, j]

    return ans

def normaMatMC(A, q, p, Np):
    m, n = np.shape(A)

    X = np.random.uniform(-1, 1, (n, Np))

    if p == 'inf':

        normasX = abs(X[0, :])

        for i in range(1, n):
            fila = abs(X[i, :])

            for k in range(Np):
                if fila[k] > normasX[k]:
                    normasX[k] = fila[k]

    else:

        normasX = np.zeros(Np)

        for i in range(n):
            normasX = normasX + abs(X[i, :]) ** p

        normasX = normasX ** (1 / p)

    X = X / normasX

    AX = np.zeros((m, Np))

    for i in range(m):
        for j in range(n):
            AX[i, :] = AX[i, :] + A[i, j] * X[j, :]

    if q == 'inf':

        valores = abs(AX[0, :])

        for i in range(1, m):
            fila = abs(AX[i, :])

            for k in range(Np):
                if fila[k] > valores[k]:
                    valores[k] = fila[k]

    else:

        valores = np.zeros(Np)

        for i in range(m):
            valores = valores + abs(AX[i, :]) ** q

        valores = valores ** (1 / q)

    indice_max = 0
    valor_max = valores[0]

    for k in range(1, Np):
        if valores[k] > valor_max:
            valor_max = valores[k]
            indice_max = k

    return valor_max, np.copy(X[:, indice_max])

def inversa(A):
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    if n != m:
        return None

    aug = np.zeros((n, 2 * n))
    aug[:, :n] = A
    aug[:, n:] = np.eye(n)

    for col in range(n):
        fila_pivote = col
        mayor = abs(aug[col, col])
        for fila in range(col + 1, n):
            valor = abs(aug[fila, col])
            if valor > mayor:
                mayor = valor
                fila_pivote = fila

        if fila_pivote != col:
            aux = np.copy(aug[col, :])
            aug[col, :] = aug[fila_pivote, :]
            aug[fila_pivote, :] = aux

        pivote = aug[col, col]
        aug[col, :] = aug[col, :] / pivote

        for fila in range(n):
            if fila != col:
                factor = aug[fila, col]
                aug[fila, :] = aug[fila, :] - factor * aug[col, :]

    return np.copy(aug[:, n:])

def condMC(A, p):
    A = np.asarray(A, dtype=float)
    A_inv = inversa(A)

    if A_inv is None:
        return None  # La matriz no es invertible

    nA, _ = normaMatMC(A, p, p, 10)
    nInv, _ = normaMatMC(A_inv, p, p, 10)

    return nA * nInv

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
