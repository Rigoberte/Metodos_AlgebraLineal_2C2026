import numpy as np

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
