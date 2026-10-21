#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def error(x, y):
    x_float64 = np.asarray(x, dtype=np.float64)
    y_float64 = np.asarray(y, dtype=np.float64)

    return np.abs(x_float64 - y_float64)

def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x,y), 0, atol=atol)

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return None, None, 0
    
    ## desde aqui -- CODIGO A COMPLETAR

    for k in range(n - 1):
        print("Solucion hasta k: ", k, ":\n", Ac)
        if sonIguales(Ac[k, k], 0):
            print("Pivote 0 en la diagonal")
            return None, None, 0

        for i in range(k+1, n):

            ik = Ac[i, k] / Ac[k, k]

            cant_op += 1

            Ac[i, k] = ik

            for j in range(k+1, n):
                Ac[i, j] = Ac[i, j] - ik * Ac[k, j]

                cant_op += 2

    print("Solucion intermedia:\n", Ac)

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

    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
    
    return L, U, cant_op


def main():
    n = 3
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()